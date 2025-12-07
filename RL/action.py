import logging
import math


class RLActionExecutor:
    def __init__(self, cluster, config):
        """
        :param cluster: 仿真器的 cluster 对象
        :param config: 包含动作步长限制的配置
        """
        self.cluster = cluster

        # --- 超参数设置 ---
        # 定义 PPO 输出 1.0 对应多少台机器的变化
        self.scale_step_size = config.get("action_scale_step", 5)  # 扩缩容步长 (e.g., ±5台)
        self.mig_step_size = config.get("action_mig_step", 3)  # 迁移步长 (e.g., ±3台)

        # 安全边界
        self.min_instances = config.get("min_instances_per_pool", 1)
        self.max_total_instances = config.get("max_total_instances", 100)

    def execute(self, action_vector):
        """
        解析并执行三维解耦动作
        action_vector: [alpha_p, alpha_t, alpha_mig] 范围通常在 [-1, 1]
        """
        # 1. 解包动作 (PPO 输出通常是 numpy array)
        alpha_p, alpha_t, alpha_mig = action_vector

        # 2. 映射为整数 (Rounding)
        # 使用 int() 或 round()，这里用 round 确保 0.1 也能有机会变成 1 (如果步长够大)
        delta_p = int(round(alpha_p * self.scale_step_size))
        delta_t = int(round(alpha_t * self.scale_step_size))
        delta_mig = int(round(alpha_mig * self.mig_step_size))

        logging.debug(f"RL Raw Action: {action_vector} -> Deltas: P={delta_p}, T={delta_t}, Mig={delta_mig}")

        self._handle_scaling(delta_p, "prefill")
        self._handle_scaling(delta_t, "decoding")

    def _handle_scaling(self, delta, role):
        """
        重写
        处理云端资源的申请与释放
        """
        if delta == 0:
            return

        current_count = self.cluster.count(role)
        total_count = self.cluster.total_count()

        if delta > 0:
            # --- 扩容 (Scale Up) ---
            # [安全约束] 检查预算上限
            remaining_quota = self.max_total_instances - total_count
            actual_add = min(delta, remaining_quota)

            if actual_add > 0:
                # 假设 cluster.add_instances 会模拟启动延迟
                self.cluster.add_instances(role=role, count=actual_add)
                logging.info(f"[Action] Added {actual_add} {role} instances")
            else:
                logging.warning(f"[Action Blocked] Max cluster size reached ({self.max_total_instances})")

        elif delta < 0:
            # --- 缩容 (Scale Down) ---
            want_remove = abs(delta)
            # [安全约束] 保证不缩减到 0
            actual_remove = min(want_remove, current_count - self.min_instances)

            if actual_remove > 0:
                # 假设 cluster.remove_instances 会处理 Graceful Shutdown
                self.cluster.remove_instances(role=role, count=actual_remove)
                logging.info(f"[Action] Removed {actual_remove} {role} instances")
            else:
                logging.warning(f"[Action Blocked] Cannot remove {role}, min limit reached")