import numpy as np
import logging
import csv
import os
from collections import defaultdict
import pandas as pd

class RLRewardCalculator:
    def __init__(self,
                 config,
                 max_instances=100,
                 price_ratio_token=0.6,
                 mode: str = "joint"):
        """
        :param config: 包含权重参数的配置对象 (DictConfig)
        :param max_instances: 集群最大机器数 (用于归一化成本)
        :param price_ratio_token: Token 机器相对于 Prompt 机器的成本比率
                                  (例如 H100=1.0, A100=0.6)
        :param mode: "prompt" / "token" / "joint"，用于区分奖励逻辑
        """
        assert mode in ("prompt", "token", "joint")
        self.mode = mode
        # 1. 权重参数 (需要通过超参数搜索微调)
        # 建议初始值: w_cost=0.5, w_slo=2.0, w_switch=0.1, w_util=0.2
        self.w_cost = config.get("w_cost", 0.5)
        self.w_slo = config.get("w_slo", 2.0)
        self.w_switch = config.get("w_switch", 0.1)
        self.w_util = config.get("w_util", 0.2)

        # 2. 硬件成本参数
        self.price_p = 1.0  # Prefill 机器 (基准价格)
        # self.price_t = price_ratio_token  # Decoding 机器 (通常较便宜)
        self.price_t = 1.0  # 假设都是用同样的机器
        self.max_instances = max_instances
        self.is_first_step = True

        # 3. 状态记忆 (用于计算切换成本)
        self.last_instances = {'p': 0, 't': 0}
        self.last_action_sign = 0 # 记录上一次是加还是减

    #     new
        self.BASE_SLO_PENALTY = 10.0
        self.ACTION_COST = 0.2
        self.HYSTERESIS_PENALTY = 2.0
        self.max_instances = max_instances
        self.last_action_sign = 0

        # SLO 阈值 (单位: 秒)
        self.TARGET_TTFT = 1.0  # 1秒
        self.TARGET_TBT = 0.05  # 50ms

    def calculate_reward(self, cluster, applications, raw_stats, instance_num, action_executed=True, step=0):
        """
        基于排队论估算的即时奖励，修复马尔可夫性破坏问题。
        
        :param avg_queue_time: 平均队列时间
        :param avg_nth_token_overhead: 平均 nth_token_overhead
        """

        # -------------------------------------------------------------
        # 1. 获取即时状态 (Leading Indicators)
        # -------------------------------------------------------------

        # A. 队列堆积情况
        # 这是“正在发生的灾难”,总prompt数
        q_prompt = raw_stats[2]
        q_decoding = raw_stats[3]

        # B. 当前系统的处理能力 (Service Rate)
        # 我们需要知道当前 1秒 能消化多少请求。
        # 可以用过去一个小窗口的平均吞吐量来近似当前的处理能力。
        # raw_stats 需要包含 'processed_prompt_reqs_per_sec' 和 'processed_token_reqs_per_sec'
        # max(0.1, ...) 防止除以零
        throughput_p = max(0.1, raw_stats[0])
        throughput_d = max(0.1, raw_stats[1])

        # -------------------------------------------------------------
        # 2. 计算“即时估算延迟” (Instantaneous Estimated Latency)
        # -------------------------------------------------------------

        # 估算 TTFT：如果在 Prefill 队列排队，要排多久？
        # 公式：排队数 / 消化速度
        est_ttft = q_prompt / throughput_p

        # 估算 TBT 压力：这里比较特殊。
        # TBT 变差通常是因为 Decoding 机器显存满了，请求进不去 Decoding Pool，
        # 或者 Decoding Pool 并发过高导致显存带宽争抢。
        # 我们可以用 (Decoding队列 / Decoding消化速度) 来衡量“等待进入 Decoding 的延迟”。
        # 如果 Decoding 队列在堆积，说明 TBT 风险极大 (因为前面的请求卡住了)。
        est_decoding_wait = q_decoding / throughput_d

        request_data = []
        request_data.append({
            'prompt_sizes': raw_stats[6],
            'ttft': est_ttft,
            'tbt': est_decoding_wait
        })
        request_df = pd.DataFrame(request_data)
        normalized_df = applications[0].scheduler.perf_model.add_baseline_perf(request_df, model="bloom-176b", hardware="h100-80gb",
                                                          tensor_parallel=8)


        # 归一化为 Ratio (相对于 SLO 阈值)
        # 注意：prompt agent 主要看 TTFT，token agent 主要看 TBT
        # ratio_ttft = est_ttft / self.TARGET_TTFT
        # ratio_tbt = est_decoding_wait / (self.TARGET_TBT * 10)  # 容忍度稍微放宽
        normalized_df['normalized_ttft'] = normalized_df['ttft'] / normalized_df['baseline_ttft']
        normalized_df['normalized_tbt'] = normalized_df['tbt'] / normalized_df['baseline_tbt']
        ratio_ttft = normalized_df['normalized_ttft'][0] / 6
        ratio_tbt = normalized_df['normalized_tbt'][0] / 5

        if self.mode == "prompt":
            max_ratio = ratio_ttft
        elif self.mode == "token":
            max_ratio = ratio_tbt
        else:
            max_ratio = max(ratio_ttft, ratio_tbt)

        # -------------------------------------------------------------
        # 3. 计算奖励 (逻辑与之前相同，但输入变了)
        # -------------------------------------------------------------

        # 计算成本分数
        n_p, n_t = raw_stats[4:6]
        if self.mode == "prompt":
            cost_score = n_p
        elif self.mode == "token":
            cost_score = n_t
        else:
            cost_score = (n_p + n_t)
        # 最大实例数*interval step
        # 计算 prompt 实例的总使用时间（自上次调用以来）
        # if self.mode == "prompt":
        #     cost_score = applications[0].scaling_manager.calculate_prompt_instance_time_since_last()
        # elif self.mode == "token":
        #     cost_score = applications[0].scaling_manager.calculate_token_instance_time_since_last()

        reward = 0.0
        # 奖励1.0
        # if max_ratio > 1.0:
        #     # === 🔴 危险区 ===
        #     # 队列堆积导致预估延迟超标，立刻重罚！
        #     # 这样 Agent 在队列刚开始堆积（t时刻）就会收到负反馈，不用等请求跑完。
        #     reward = -self.BASE_SLO_PENALTY * ((max_ratio - 1.0) ** 2) - 2.0
        #
        # elif max_ratio > 0.8:
        #     # === 🟡 缓冲区 ===
        #     reward = (1.0 - cost_score) + 0.5
        #
        # else:
        #     # === 🟢 安全区 ===
        #     reward = 1.0 - cost_score

        # 奖励2.0
        # reward_stats = [prompt_rate, token_rate, p_queue, d_queue, n_p, n_t, avg_prompt_size, ttft_rate, tbt_rate, p_queue_len, d_queue_len]
        # Agent 会为了让 penalty_slo 保持为 0 而不敢越线
        # 在 penalty_slo 为 0 的前提下，它会尽可能减小 cost_term
        queue_len = raw_stats[2] if self.mode == "prompt" else raw_stats[3]
        if self.mode == "prompt":
            use_time = applications[0].scaling_manager.calculate_prompt_instance_time_since_last()
        else :
            use_time = applications[0].scaling_manager.calculate_token_instance_time_since_last()
        reward = -3 * (queue_len/10000)
        # - self.w_cost * use_time
        # print(-self.w_slo * np.log1p(q_prompt),- self.w_cost * cost_score)

        # 奖励3.0
        #   不对，现在的排队应该是越少越好，适当的
        # 1. 悬崖：违约 (Ratio > 1.0)
        #     reward =  -self.w_slo * (max_ratio ** 2)  # 重罚
        # # 2. 甜头：黄金区间 (0.8 < Ratio <= 1.0)
        # # 这就是让 Agent 勇敢的原因！只要保持在这里，不仅不罚，还给额外的正反馈
        # elif max_ratio > 0.8:
        #     reward = 2 - cost_score + max_ratio * 10
        # # 3. 浪费：离悬崖太远 (Ratio <= 0.8) 即使没违约，但因为离得太远，没有“甜头”拿，只有省钱的微薄奖励
        # else:
        #     reward = - cost_score * ((1-max_ratio)*100)**2

        # 奖励4.0

        # -------------------------------------------------------------
        # 4. 稳定性惩罚
        # -------------------------------------------------------------
        # (保持原有的迟滞惩罚逻辑)
        # ...
        info = {
            'step':step,
            'reward':reward,
            'cost_score': cost_score,
            'ttft_p50':raw_stats[7][0],
            'ttft_p90':raw_stats[7][1],
            'ttft_p99':raw_stats[7][2],
            'tbt_p50':raw_stats[8][0],
            'tbt_p90':raw_stats[8][1],
            'tbt_p99':raw_stats[8][2],
            'p_queue_len':raw_stats[2], #sch_queue
            't_queue_len':raw_stats[3],
            'instance_p_queue_len':raw_stats[9],
            'instance_t_queue_len':raw_stats[10],
            'use_time':use_time,
            'avg_queue_time': raw_stats[11],
            'avg_nth_token_overhead': raw_stats[12]
        }
        return reward,info

    def calculate_reward_(self, cluster, applications, interval_stats, instance_num, action_executed=True):
        """
        计算单步奖励
        :param cluster: Cluster 对象
        :param applications: App 对象
        :param interval_stats: 包含 TTFT和TBT的P50、P90、P99值的列表
                               格式：[[ttft_p50, ttft_p90, ttft_p99], [tbt_p50, tbt_p90, tbt_p99], [ttft_vio, tbt_vio]]
        :param instance_num: 实例数量和利用率
        :param action_executed: 是否执行了扩缩容动作
        :return: (total_reward, info_dict)
        """

        # --- A. 运营成本项 (OpEx) ---
        # 目标：最小化租金
        n_p, n_t = instance_num[0], instance_num[1]

        # 计算加权成本 (Normalized by max budget)
        current_cost = (n_p * self.price_p + n_t * self.price_t)
        max_possible_cost = self.max_instances * 1.0

        cost_penalty = -current_cost

        # --- B. SLO 奖励项 (Performance) ---
        # 使用 P50、P90、P99 的 TTFT 和 TBT 计算 SLO 合规性
        # SLO 阈值：
        # TTFT: P50=2, P90=3, P99=6
        # TBT: P50=1.25, P90=1.5, P99=5
        
        ttft_values = interval_stats[0]  # [p50, p90, p99]
        tbt_values = interval_stats[1]   # [p50, p90, p99]
        
        # TTFT SLO 阈值
        ttft_slo_thresholds = [2.0, 3.0, 6.0]  # P50, P90, P99
        # TBT SLO 阈值
        tbt_slo_thresholds = [1.25, 1.5, 5.0]  # P50, P90, P99
        
        # 计算每个分位数的合规率（值越小于阈值越好）
        ttft_compliance_scores = []
        tbt_compliance_scores = []
        
        # 奖励逻辑：SLO 符合 (≤ 阈值) 给小奖励，不符合 (> 阈值) 给惩罚
        ttft_compliance_scores = []
        tbt_compliance_scores = []
        for i in range(3):
            # TTFT
            if ttft_values[i] <= ttft_slo_thresholds[i]:
                ttft_score = 10  # 符合 SLO 给小奖励
            else:
                ttft_score = -1 * (ttft_values[i] - ttft_slo_thresholds[i]) *10 # 超出 SLO 给惩罚，按超出比例线性
            ttft_compliance_scores.append(ttft_score)
            # TBT
            if tbt_values[i] <= tbt_slo_thresholds[i]:
                tbt_score = 10
            else:
                tbt_score = -1 * (tbt_values[i] - tbt_slo_thresholds[i]) *10
            tbt_compliance_scores.append(tbt_score)
        # 加权平均
        weights = [0.2, 0.3, 0.5]
        ttft_weighted_score = sum(w * s for w, s in zip(weights, ttft_compliance_scores))
        tbt_weighted_score = sum(w * s for w, s in zip(weights, tbt_compliance_scores))
        
        # 综合 SLO 奖励（TTFT 和 TBT 各占一半）
        slo_reward = 0.5 * ttft_weighted_score + 0.5 * tbt_weighted_score
        

        # --- C. 切换成本项 & 稳定性奖励 (Stability) ---
        # 目标：抑制机器数量剧烈抖动，奖励稳定状态
        stability_bonus = 0.0
        
        if self.is_first_step:
            self.is_first_step = False
            switch_penalty = 0.0
        else:
            # 如果没有执行扩缩容动作（delta_total == 0 或 action_executed == False）
            if not action_executed:
                # 给予稳定性奖励
                stability_bonus = 5  # 鼓励保持稳定
                switch_penalty = 0.0
            else:
                switch_penalty = -5

    
        delta_p = abs(n_p - self.last_instances['p'])
        delta_t = abs(n_t - self.last_instances['t'])
        delta_total = delta_p + delta_t
            
        # 更新历史
        self.last_instances = {'p': n_p, 't': n_t}

        # --- D. 利用率塑形 (Reward Shaping - Optional) ---
        # 目标：引导 Agent 将利用率维持在 "Sweet Spot" (例如 60% - 80%)
        # 避免 0% (浪费) 也不要 100% (容易排队)
        util_p,util_d = instance_num[2],instance_num[3]

        def utilization_bonus(u):
            # 一个倒 U 型函数，在 0.7 处达到峰值 1.0
            # 改为指数形式以增强敏感度
            return np.exp(1.0 - abs(u - 0.7)) / np.e

        util_reward = 0.3 * utilization_bonus(util_p) + 0.3 * utilization_bonus(util_d)

        # --- E. 总奖励聚合 ---
        # 注意：Cost 和 Switch 是负值，SLO、Util 和 Stability 是正值
        # 标准化各组件值到相似范围，保持符号不变
        total_reward = (
                self.w_cost * cost_penalty +
                self.w_slo * slo_reward +
                self.w_switch * (switch_penalty + stability_bonus)
                # self.w_util * util_reward
        )

        # 返回详细信息用于 Debug (这对 RL 调参至关重要！)
        info = {
            "reward_total": total_reward,
            "raw_cost": current_cost,
            "pen_cost": self.w_cost * cost_penalty,
            "rew_slo": self.w_slo * slo_reward,
            "pen_switch": self.w_switch * switch_penalty,
            "rew_stability": self.w_switch * stability_bonus,
            "rew_util": self.w_util * util_reward,
            "ttft_weighted": ttft_weighted_score,
            "tbt_weighted": tbt_weighted_score,
            "ttft_p50": ttft_values[0] if len(ttft_values) > 0 else 0,
            "ttft_p90": ttft_values[1] if len(ttft_values) > 1 else 0,
            "ttft_p99": ttft_values[2] if len(ttft_values) > 2 else 0,
            "tbt_p50": tbt_values[0] if len(tbt_values) > 0 else 0,
            "tbt_p90": tbt_values[1] if len(tbt_values) > 1 else 0,
            "tbt_p99": tbt_values[2] if len(tbt_values) > 2 else 0,
            "delta_total": delta_total,
            "action_executed": action_executed,
            "util_avg": (util_p + util_d) / 2
        }

        return total_reward, info

    def reset(self):
        """重置内部状态 (每个 Episode 开始时调用)"""
        self.last_instances = {'p': 0, 't': 0}

class RewardRecorder:


    def __init__(self, filename="reward.csv", clear_file=True):
        self.filename = filename
        self.fieldnames = None  # 将在第一次调用record_reward时初始化
        self._initialize_csv(clear_file)

    def _initialize_csv(self, clear_file=True):
        """Initialize the CSV file with headers"""
        # 如果需要清空文件或者文件不存在，则重新创建文件并写入表头
        if clear_file or not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                pass  # 只创建空文件

    def record_reward(self, step, info_dict):
        """
        Record reward components to CSV file

        :param step: Current decision step
        :param info_dict: Dictionary containing reward components from RLRewardCalculator
        """

        # 如果是第一次调用，初始化fieldnames并写入表头
        if self.fieldnames is None:
            self.fieldnames = list(info_dict.keys())
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

        # 使用追加模式写入数据
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(info_dict)