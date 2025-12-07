import numpy as np
import logging


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
        self.price_t = price_ratio_token  # Decoding 机器 (通常较便宜)
        self.price_m = 1.0  # Mixed 机器 (通常假设等同于昂贵机器)
        self.max_instances = max_instances

        # 3. 状态记忆 (用于计算切换成本)
        self.last_instances = {'p': 0, 't': 0, 'm': 0}

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

        # 成本惩罚 (0 ~ -1 之间)
        cost_penalty = -(current_cost / max_possible_cost)

        # --- B. SLO 惩罚项 (Performance) ---
        # 目标：违约率越低越好。使用平方惩罚，对严重违约重拳出击。
        # interval_stats 来自 StateCollector，包含 'ttft_rate' 和 'tbt_rate'
        ttft_vio_rate = interval_stats[0]
        tbt_vio_rate = interval_stats[1]

        # 重点惩罚 TBT (因为 Splitwise 中 Token Phase 是长尾)
        # 示例: 10% 违约率 -> 0.1^2 = 0.01; 50% 违约率 -> 0.25 (惩罚激增)
        slo_penalty = -(0.4 * (ttft_vio_rate ** 2) + 0.6 * (tbt_vio_rate ** 2))

        # --- C. 切换成本项 (Stability) ---
        # 目标：抑制机器数量剧烈抖动
        delta_p = abs(n_p - self.last_instances['p'])
        delta_t = abs(n_t - self.last_instances['t'])
        delta_m = abs(n_m - self.last_instances['m'])

        # 归一化切换量 (假设一次最多变动 10 台)
        total_delta = delta_p + delta_t + delta_m
        switch_penalty = -(total_delta / 10.0)

        # 更新历史
        self.last_instances = {'p': n_p, 't': n_t, 'm': n_m}

        # --- D. 利用率塑形 (Reward Shaping - Optional) ---
        # 目标：引导 Agent 将利用率维持在 "Sweet Spot" (例如 60% - 80%)
        # 避免 0% (浪费) 也不要 100% (容易排队)
        util_p,util_d,util_m = instance_num[3],instance_num[4],instance_num[5]

        def utilization_bonus(u):
            # 一个倒 U 型函数，在 0.7 处达到峰值 1.0
            # 简单实现：1 - |u - 0.7|
            return 1.0 - abs(u - 0.7)

        util_reward = 0.3 * utilization_bonus(util_p) + 0.3 * utilization_bonus(util_d)+ 0.3 * utilization_bonus(util_m)

        # --- E. 总奖励聚合 ---
        # 注意：Switch 和 SLO 是负值，Util 是正值
        total_reward = (
                self.w_cost * cost_penalty +
                self.w_slo * slo_penalty +
                self.w_switch * switch_penalty +
                self.w_util * util_reward
        )

        # 返回详细信息用于 Debug (这对 RL 调参至关重要！)
        info = {
            "reward_total": total_reward,
            "raw_cost": current_cost,
            "pen_cost": self.w_cost * cost_penalty,
            "pen_slo": self.w_slo * slo_penalty,
            "pen_switch": self.w_switch * switch_penalty,
            "rew_util": self.w_util * util_reward,
            "ttft_vio": ttft_vio_rate,
            "tbt_vio": tbt_vio_rate
        }

        return total_reward, info

    def reset(self):
        """重置内部状态 (每个 Episode 开始时调用)"""
        self.last_instances = {'p': 0, 't': 0, 'm': 0}