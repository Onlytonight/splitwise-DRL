import numpy as np
import logging
import csv
import os
from collections import defaultdict

class RLRewardCalculator:
    def __init__(self,
                 config,
                 max_instances=100,
                 price_ratio_token=0.6):
        """
        :param config: 包含权重参数的配置对象 (DictConfig)
        :param max_instances: 集群最大机器数 (用于归一化成本)
        :param price_ratio_token: Token 机器相对于 Prompt 机器的成本比率
                                  (例如 H100=1.0, A100=0.6)
        """
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
        self.price_m = 1.0  # Mixed 机器 (通常假设等同于昂贵机器)
        self.max_instances = max_instances
        self.is_first_step = True

        # 3. 状态记忆 (用于计算切换成本)
        self.last_instances = {'p': 0, 't': 0, 'm': 0}
        self.last_action_sign = 0 # 记录上一次是加还是减

    def calculate_reward(self, cluster, applications, interval_stats, instance_num, action_executed=True):
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
        n_p, n_t, n_m = instance_num[0], instance_num[1], instance_num[2]

        # 计算加权成本 (Normalized by max budget)
        current_cost = (n_p * self.price_p + n_t * self.price_t + n_m * self.price_m)
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
        delta_m = abs(n_m - self.last_instances['m'])
        delta_total = delta_p + delta_t + delta_m
        
            
        # 更新历史
        self.last_instances = {'p': n_p, 't': n_t, 'm': n_m}

        # --- D. 利用率塑形 (Reward Shaping - Optional) ---
        # 目标：引导 Agent 将利用率维持在 "Sweet Spot" (例如 60% - 80%)
        # 避免 0% (浪费) 也不要 100% (容易排队)
        util_p,util_d,util_m = instance_num[3],instance_num[4],instance_num[5]

        def utilization_bonus(u):
            # 一个倒 U 型函数，在 0.7 处达到峰值 1.0
            # 改为指数形式以增强敏感度
            return np.exp(1.0 - abs(u - 0.7)) / np.e

        util_reward = 0.3 * utilization_bonus(util_p) + 0.3 * utilization_bonus(util_d)+ 0.3 * utilization_bonus(util_m)

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
            "util_avg": (util_p + util_d + util_m) / 3
        }

        return total_reward, info

    def reset(self):
        """重置内部状态 (每个 Episode 开始时调用)"""
        self.last_instances = {'p': 0, 't': 0, 'm': 0}

class RewardRecorder:


    def __init__(self, filename="reward.csv", clear_file=True):
        self.filename = filename
        self.fieldnames = [
            "step", "total_reward", "cost_penalty", "slo_reward", "switch_penalty", 
            "stability_bonus", "util_reward", "raw_cost", 
            "ttft_weighted", "tbt_weighted",
            "ttft_p50", "ttft_p90", "ttft_p99",
            "tbt_p50", "tbt_p90", "tbt_p99",
            "delta_total", "action_executed", "util_avg"
        ]
        self._initialize_csv(clear_file)

    def _initialize_csv(self, clear_file=True):
        """Initialize the CSV file with headers"""
        # 如果需要清空文件或者文件不存在，则重新创建文件并写入表头
        if clear_file or not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def record_reward(self, step, info_dict):
        """
        Record reward components to CSV file

        :param step: Current decision step
        :param info_dict: Dictionary containing reward components from RLRewardCalculator
        """
        row_data = {
            "step": step,
            "total_reward": info_dict.get("reward_total", 0),
            "cost_penalty": info_dict.get("pen_cost", 0),
            "slo_reward": info_dict.get("rew_slo", 0),
            "switch_penalty": info_dict.get("pen_switch", 0),
            "stability_bonus": info_dict.get("rew_stability", 0),
            "util_reward": info_dict.get("rew_util", 0),
            "raw_cost": info_dict.get("raw_cost", 0),
            "ttft_weighted": info_dict.get("ttft_weighted", 0),
            "tbt_weighted": info_dict.get("tbt_weighted", 0),
            "ttft_p50": info_dict.get("ttft_p50", 0),
            "ttft_p90": info_dict.get("ttft_p90", 0),
            "ttft_p99": info_dict.get("ttft_p99", 0),
            "tbt_p50": info_dict.get("tbt_p50", 0),
            "tbt_p90": info_dict.get("tbt_p90", 0),
            "tbt_p99": info_dict.get("tbt_p99", 0),
            "delta_total": info_dict.get("delta_total", 0),
            "action_executed": info_dict.get("action_executed", True),
            "util_avg": info_dict.get("util_avg", 0),
        }

        # 使用追加模式写入数据
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row_data)