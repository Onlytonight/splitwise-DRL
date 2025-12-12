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

    def calculate_reward(self, cluster, applications, interval_stats,instance_num):
        """
        计算单步奖励
        :param cluster: Cluster 对象
        :param applications: App 对象
        :param interval_stats: 这是一个字典，包含 RLStateCollector 计算出的区间统计值
                               (如 SLO 违约率, RPS 等)
        :return: (total_reward, info_dict)
        """

        # --- A. 运营成本项 (OpEx) ---
        # 目标：最小化租金
        n_p, n_t, n_m = instance_num[0],instance_num[1],instance_num[2]

        # 计算加权成本 (Normalized by max budget)
        current_cost = (n_p * self.price_p + n_t * self.price_t + n_m * self.price_m)
        max_possible_cost = self.max_instances * 1.0

        # 成本惩罚 (0 ~ -1 之间) - 改为指数衰减
        cost_penalty = -(np.exp(current_cost / max_possible_cost) - 1) / (np.e - 1)

        # --- B. SLO 奖励项 (Performance) ---
        # 目标：合规率越高越好。使用指数奖励，对高合规率给予更高奖励。低于99%才开始扣分
        # interval_stats 来自 StateCollector，包含 'ttft_rate' 和 'tbt_rate'
        ttft_compliance_rate = max(0, 0.99 - interval_stats[0]) / 0.99  # 0.99为基准合规率
        tbt_compliance_rate = max(0, 0.99 - interval_stats[1]) / 0.99   # 0.99为基准合规率

        # 指数奖励 TBT (因为 Splitwise 中 Token Phase 是长尾)
        # 合规率越高，奖励越高 (0~1)
        # 标准化 SLO 奖励到 0~1 范围内
        slo_reward = (np.exp(0.4 * ttft_compliance_rate + 0.6 * tbt_compliance_rate) - 1) / (np.e - 1)
        
        

        # --- C. 切换成本项 (Stability) ---
        # 目标：抑制机器数量剧烈抖动
        delta_total = 0
        if self.is_first_step:
            self.is_first_step = False
            switch_penalty = 0.0
        else:
            delta_p = abs(n_p - self.last_instances['p'])
            delta_t = abs(n_t - self.last_instances['t'])
            delta_m = abs(n_m - self.last_instances['m'])
            
            delta_total = delta_p + delta_t + delta_m
            current_action_sign = np.sign(delta_total)
            hysteresis_penalty = 0
            # 如果上一步不为0，这一步也不为0，且方向相反
            if self.last_action_sign != 0 and current_action_sign != 0:
                if self.last_action_sign != current_action_sign:
                    # 触发重罚！罚分是普通 switch 的 10 倍
                    hysteresis_penalty = -5.0         
            self.last_action_sign = current_action_sign
            
            # 归一化切换量 (假设一次最多变动 10 台) - 改为指数衰减
            total_delta = delta_p + delta_t + delta_m
            switch_penalty = -(np.exp(total_delta / 10.0) - 1) / (np.e - 1) + hysteresis_penalty

            
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
        # 注意：Cost 和 Switch 是负值，SLO 和 Util 是正值
        # 标准化各组件值到相似范围，保持符号不变
        total_reward = (
                self.w_cost * cost_penalty +
                self.w_slo * slo_reward 
                # self.w_switch * switch_penalty +
                # self.w_util * util_reward
        )

        # 返回详细信息用于 Debug (这对 RL 调参至关重要！)
        info = {
            "reward_total": total_reward,
            "raw_cost": current_cost,
            "pen_cost": self.w_cost * cost_penalty,
            "rew_slo": self.w_slo * slo_reward,
            "pen_switch": self.w_switch * switch_penalty,
            "rew_util": self.w_util * util_reward,
            "ttft_compliance": ttft_compliance_rate,
            "tbt_compliance": tbt_compliance_rate,
            "delta_total": delta_total,
            "util_avg": (util_p + util_d + util_m) / 3
        }

        return total_reward, info

    def reset(self):
        """重置内部状态 (每个 Episode 开始时调用)"""
        self.last_instances = {'p': 0, 't': 0, 'm': 0}

class RewardRecorder:


    def __init__(self, filename="reward.csv", clear_file=True):
        self.filename = filename
        self.fieldnames = ["step", "total_reward", "cost_penalty", "slo_reward", "switch_penalty", "util_reward", "raw_cost", "ttft_compliance", "tbt_compliance", "delta_total", "util_avg"]
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
            "util_reward": info_dict.get("rew_util", 0),
            "raw_cost": info_dict.get("raw_cost", 0),
            "ttft_compliance": info_dict.get("ttft_compliance", 0),
            "tbt_compliance": info_dict.get("tbt_compliance", 0),
            "delta_total": info_dict.get("delta_total", 0),
            "util_avg": info_dict.get("util_avg", 0),
        }

        # 使用追加模式写入数据
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow(row_data)