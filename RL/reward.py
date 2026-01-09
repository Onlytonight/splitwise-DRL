import numpy as np
import logging
import csv
import os
import time
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
        # 1. 权重参数
        # 简化设计：只考虑队列惩罚 + 成本惩罚
        # 目标：队列数为0的情况下成本最低
        self.w_cost = config.get("w_cost", 0.1)  # 成本惩罚权重
        self.w_queue = config.get("w_queue", 1.0)  # 队列惩罚权重
        # 保留旧参数以兼容（但不再使用）
        self.w_congestion = config.get("w_congestion", 1.0)  # 已废弃，使用w_queue
        self.w_stability = config.get("w_stability", 0.0)  # 已废弃
        self.w_slo = config.get("w_slo", 0.0)  # 已废弃
        self.w_switch = config.get("w_switch", 0.0)  # 已废弃
        self.w_util = config.get("w_util", 0.0)  # 利用率惩罚权重

        # 2. 硬件成本参数
        self.price_p = 1.0  # Prefill 机器 (基准价格)
        self.price_t = 1.0  # 假设都是用同样的机器
        self.max_instances = max_instances

        # 3. 状态记忆
        self.last_instances = {'p': 0, 't': 0}
        self.last_queue_delta = {'p': 0.0, 't': 0.0}  # 记录上一次的队列差值，用于计算导数

        # 4. 拥堵惩罚阈值
        self.PENDING_TOKEN_THRESHOLD = 100.0  # prompt实例待处理token的阈值
        self.UTIL_MEM_THRESHOLD = 0.8  # 内存利用率过载预警阈值

    def calculate_reward(self, cluster, applications, raw_stats, instance_num, action_executed=True, step=0):
        """
        简化的奖励函数：只考虑队列和成本
        目标：队列数为0的情况下成本最低
        
        :param action_executed:
        :param cluster:
        :param applications:
        :param instance_num:
        :param raw_stats: [prompt_rate, token_rate, sch_p_queue, sch_d_queue, n_p, n_t,
                          avg_prompt_size, ttft, tbt, ins_p_queue, ins_d_queue, 
                          avg_queue_time, avg_nth_token_overhead, use_time, rps]
        """
        # -------------------------------------------------------------
        # 1. 成本惩罚 (Cost Penalty)
        # -------------------------------------------------------------
        n_p, n_t = raw_stats[4:6]
        if self.mode == "prompt":
            cost_score = n_p
        elif self.mode == "token":
            cost_score = n_t
        else:
            cost_score = (n_p + n_t)
        
        cost_penalty = -self.w_cost * cost_score

        # -------------------------------------------------------------
        # 2. 队列惩罚 (Queue Penalty) - 核心信号
        # -------------------------------------------------------------
        # A. 队列存在惩罚（使用队列的绝对值）
        q_prompt = raw_stats[2]  # 调度器中的prompt队列
        q_decoding = raw_stats[3]  # 调度器中的decoding队列

        # # 修复bug：原来两次都用了 q_decoding
        # queue_absolute_penalty = -(np.log1p(q_prompt) + np.log1p(q_decoding)) * self.w_queue

        # B. 队列差值惩罚（使用队列的变化量）
        # raw_stats 格式已更新，包含队列差值（在最后两个位置）
        p_queue_delta = raw_stats[15] if len(raw_stats) > 15 else 0.0  # prompt队列差值
        d_queue_delta = raw_stats[16] if len(raw_stats) > 16 else 0.0  # decoding队列差值

        # 队列差值惩罚：正值表示队列增长（需要惩罚），负值表示队列减少（给予奖励）
        # 改用 log(1+abs(x))，并保留符号
        queue_delta_penalty = -(
            np.sign(p_queue_delta) * np.log1p(abs(p_queue_delta)) + 
            np.sign(d_queue_delta) * np.log1p(abs(d_queue_delta))
        ) * self.w_queue

        # C. 队列导数惩罚（队列变化的加速度）
        # 导数 = 当前差值 - 上一次差值
        p_queue_derivative = p_queue_delta - self.last_queue_delta['p']
        d_queue_derivative = d_queue_delta - self.last_queue_delta['t']

        # 队列导数惩罚：正值表示队列增长加速（更严重），负值表示队列增长减速（好转）
        # 改用 log(1+abs(x))，并保留符号
        queue_derivative_penalty = -(
            np.sign(p_queue_derivative) * np.log1p(abs(p_queue_derivative)) + 
            np.sign(d_queue_derivative) * np.log1p(abs(d_queue_derivative))
        ) * self.w_queue

        # 更新上一次的队列差值
        self.last_queue_delta = {'p': p_queue_delta, 't': d_queue_delta}

        # D. 综合队列惩罚
        queue_penalty =  queue_delta_penalty + queue_derivative_penalty

        # E. 利用率过载预警（从instance_num获取）
        # instance_num格式: [n_p, n_t, util_p, util_d, util_mem_p, util_mem_t]
        util_p, util_d = instance_num[2], instance_num[3]
        util_mem_p = instance_num[4] if len(instance_num) > 4 else 0.0
        util_mem_t = instance_num[5] if len(instance_num) > 5 else 0.0

        # 如果内存利用率过高，给予惩罚
        overload_penalty = 0.0
        if util_mem_p > self.UTIL_MEM_THRESHOLD:
            overload_penalty -= self.w_util * (util_mem_p - self.UTIL_MEM_THRESHOLD) * 10.0
        if util_mem_t > self.UTIL_MEM_THRESHOLD:
            overload_penalty -= self.w_util * (util_mem_t - self.UTIL_MEM_THRESHOLD) * 10.0

        congestion_penalty = queue_penalty + overload_penalty

        # -------------------------------------------------------------
        # 4. 总奖励
        # -------------------------------------------------------------
        reward = cost_penalty + congestion_penalty

        # # ======== 打印详细信息 ========
        # print(f"Step详细信息：")
        # print(f"  模式(mode): {self.mode}")
        # print(f"  实例数 n_p: {n_p}, n_t: {n_t}")
        # print(f"  调度器队列 q_prompt: {q_prompt}, q_decoding: {q_decoding}")
        # print(f"  队列差值 p_delta: {p_queue_delta:.2f}, d_delta: {d_queue_delta:.2f}")
        # print(f"  队列导数 p_deriv: {p_queue_derivative:.2f}, d_deriv: {d_queue_derivative:.2f}")
        # print(f"  cost_score: {cost_score}")
        # print(f"  成本惩罚 cost_penalty: {cost_penalty:.4f}")
        # print(f"  队列差值惩罚 queue_delta_penalty: {queue_delta_penalty:.4f}")
        # print(f"  队列导数惩罚 queue_derivative_penalty: {queue_derivative_penalty:.4f}")
        # print(f"  队列总惩罚 queue_penalty: {queue_penalty:.4f}")
        # print(f"  util_p: {util_p:.2f}, util_d: {util_d:.2f}, util_mem_p: {util_mem_p:.2f}, util_mem_t: {util_mem_t:.2f}")
        # print(f"  overload_penalty: {overload_penalty:.4f}")
        # print(f"  总惩罚 congestion_penalty: {congestion_penalty:.4f}")
        # print(f"  ==> 总奖励 reward: {reward:.4f}")

        
        # 更新状态记忆
        self.last_instances = {'p': n_p, 't': n_t}

        # -------------------------------------------------------------
        # 4. 构建info字典（保留SLO信息用于外部评估）
        # -------------------------------------------------------------
        ins_p_queue = raw_stats[9] if len(raw_stats) > 9 else 0
        ins_d_queue = raw_stats[10] if len(raw_stats) > 10 else 0
        
        info = {
            'step': step,
            'reward': reward,
            'cost_score': cost_score,
            'cost_penalty': cost_penalty,
            'queue_delta_penalty': queue_delta_penalty,
            'queue_derivative_penalty': queue_derivative_penalty,
            'queue_penalty': queue_penalty,
            'congestion_penalty': congestion_penalty,
            'overload_penalty': overload_penalty,
            'n_p': n_p,
            'n_t': n_t,
            # 保留SLO信息用于外部评估（不参与奖励计算）
            'ttft_p50': raw_stats[7][0] if len(raw_stats) > 7 else 0,
            'ttft_p90': raw_stats[7][1] if len(raw_stats) > 7 else 0,
            'ttft_p99': raw_stats[7][2] if len(raw_stats) > 7 else 0,
            'tbt_p50': raw_stats[8][0] if len(raw_stats) > 8 else 0,
            'tbt_p90': raw_stats[8][1] if len(raw_stats) > 8 else 0,
            'tbt_p99': raw_stats[8][2] if len(raw_stats) > 8 else 0,
            'p_queue_len': q_prompt,
            't_queue_len': q_decoding,
            'p_queue_delta': p_queue_delta,
            't_queue_delta': d_queue_delta,
            'p_queue_derivative': p_queue_derivative,
            't_queue_derivative': d_queue_derivative,
            'instance_p_queue_len': ins_p_queue,
            'instance_t_queue_len': ins_d_queue,
            'use_time': raw_stats[13] if len(raw_stats) > 13 else 0,
            'avg_queue_time': raw_stats[11] if len(raw_stats) > 11 else 0,
            'avg_nth_token_overhead': raw_stats[12] if len(raw_stats) > 12 else 0,
        }
        
        return reward, info

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
        self.last_queue_delta = {'p': 0.0, 't': 0.0}

    def get_slo_reward(self, ttft, tbt):
        """
        根据 TTFT/TBT 与 SLO 的相对关系计算 SLO 奖励：
        - 以 2 * SLO 为分界线，对应 0 奖励
        - 小于 2 * SLO 时：按 (2 - 实际值/SLO) 线性正奖励（越小越好，线性增大）
        - 大于 2 * SLO 时：按 (实际值/SLO - 2) 线性惩罚（越大惩罚越重）
        """
        slo_reward = 0.0
        TTFT_SLO = [2, 3, 6]
        TBT_SLO = [1.25, 1.5, 5]

        for i in range(len(TTFT_SLO)):
            # ----- TTFT 奖励/惩罚 -----
            tt = ttft[i]
            tt_slo = TTFT_SLO[i]
            ratio_tt = tt / tt_slo if tt_slo > 0 else 1.0
            base_tt_pos = 10.0  # 满足或远优于 SLO 时的正奖励尺度

            if ratio_tt <= 2.0:
                # [0, 2*SLO]：从 2*SLO 处 0 奖励，向左线性增加
                factor = 2.0 - ratio_tt  # 1 -> 0
                slo_reward += max(0.0, base_tt_pos * factor)
            else:
                # > 2*SLO：线性惩罚（随超标程度线性增加负值）
                penalty = -base_tt_pos * (ratio_tt - 2.0)
                slo_reward += penalty

            # ----- TBT 奖励/惩罚 -----
            tb = tbt[i]
            tb_slo = TBT_SLO[i]
            ratio_tb = tb / tb_slo if tb_slo > 0 else 1.0
            base_tb_pos = 10.0  # 满足或远优于 SLO 时的正奖励尺度

            if ratio_tb <= 2.0:
                # [0, 2*SLO]：从 2*SLO 处 0 奖励，向左线性增加
                factor = 2.0 - ratio_tb
                slo_reward += max(0.0, base_tb_pos * factor)
            else:
                penalty = -base_tb_pos * (ratio_tb - 2.0)
                slo_reward += penalty

        return slo_reward

class RewardRecorder:

    def __init__(self, filename="reward.csv", clear_file=True, buffer_size=100):
        self.filename = filename
        self.fieldnames = None  # 将在第一次调用record_reward时初始化
        self.buffer_size = buffer_size  # 缓冲区大小（增加到100以减少文件打开频率）
        self.write_buffer = []  # 写入缓冲区
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
        try:
            # 如果是第一次调用，初始化fieldnames并写入表头
            if self.fieldnames is None:
                self.fieldnames = list(info_dict.keys())
                try:
                    with open(self.filename, 'w', newline='') as csvfile:
                        writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                        writer.writeheader()
                        csvfile.flush()
                except (IOError, OSError) as e:
                    logging.error(f"Failed to write header to {self.filename}: {e}")
                    return

            # 将数据添加到缓冲区
            self.write_buffer.append(info_dict)
            
            # 当缓冲区达到指定大小时，批量写入文件
            if len(self.write_buffer) >= self.buffer_size:
                self._flush_buffer()
        except Exception as e:
            logging.error(f"Unexpected error in record_reward: {e}")
    
    def _flush_buffer(self, max_retries=3, retry_delay=0.1):
        """将缓冲区中的数据写入文件，带重试机制"""
        if not self.write_buffer:
            return
        
        for attempt in range(max_retries):
            try:
                with open(self.filename, 'a', newline='') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                    for row in self.write_buffer:
                        writer.writerow(row)
                    csvfile.flush()
                self.write_buffer.clear()
                return  # 成功写入，退出
            except (IOError, OSError) as e:
                if attempt < max_retries - 1:
                    # 等待后重试
                    time.sleep(retry_delay * (attempt + 1))  # 指数退避
                    logging.warning(f"Retry {attempt + 1}/{max_retries} flushing buffer to {self.filename}")
                else:
                    # 最后一次尝试失败，记录错误但不清空缓冲区（保留数据）
                    logging.error(f"Failed to flush buffer to {self.filename} after {max_retries} attempts: {e}")
                    # 不清空缓冲区，保留数据以便下次尝试
                    # 但如果缓冲区太大，清空一部分以避免内存问题
                    if len(self.write_buffer) > self.buffer_size * 2:
                        logging.warning(f"Buffer too large ({len(self.write_buffer)}), clearing half")
                        self.write_buffer = self.write_buffer[len(self.write_buffer)//2:]
    
    def close(self):
        """关闭记录器，确保所有缓冲数据都被写入"""
        if self.write_buffer:
            self._flush_buffer()