import logging
import math


class RLActionExecutor:
    def __init__(self, application, config):
        """
        :param application: 应用对象（包含 scaling_manager 和 scheduler）
        :param config: 包含动作步长限制的配置
        """
        self.application = application
        self.scaling_manager = application.scaling_manager
        self.scheduler = application.scheduler

        # --- 超参数设置 ---
        # 定义 PPO 输出 1.0 对应多少台机器的变化
        self.scale_step_size = config.get("action_scale_step", 5)  # 扩缩容步长 (e.g., ±5台)
        self.mig_step_size = config.get("action_mig_step", 3)  # 迁移步长 (e.g., ±3台)

        # 安全边界
        self.min_instances = config.get("min_instances_per_pool", 1)
        self.max_total_instances = config.get("max_total_instances", 100)

    def execute(self, action_vector):
        """
        解析并执行二维解耦动作
        action_vector: [alpha_p, alpha_t] 范围通常在 [-1, 1]
        """
        # 1. 解包动作 (PPO 输出通常是 numpy array)
        alpha_p, alpha_t = action_vector

        # 2. 映射为整数 (Rounding)
        # 使用 int() 或 round()，这里用 round 确保 0.1 也能有机会变成 1 (如果步长够大)
        # delta_p = int(round(alpha_p * self.scale_step_size))
        # delta_t = int(round(alpha_t * self.scale_step_size))
        threshold = 0.3  # 这个阈值就是你想要的“选择维度”

        if abs(alpha_p) < threshold:
            delta_p = 0
        else:
            # 重新映射剩余区间，保持线性
            # (val - thresh) 用于确保刚过阈值时是从 1 开始
            sign = 1 if alpha_p > 0 else -1
            magnitude = (abs(alpha_p) - threshold) / (1 - threshold)
            delta_p = int(round(sign * magnitude * self.scale_step_size))

        if abs(alpha_t) < threshold:
            delta_t = 0
        else:
            # 重新映射剩余区间，保持线性
            # (val - thresh) 用于确保刚过阈值时是从 1 开始
            sign = 1 if alpha_t > 0 else -1
            magnitude = (abs(alpha_t) - threshold) / (1 - threshold)
            delta_t = int(round(sign * magnitude * self.scale_step_size))

        logging.debug(f"RL Raw Action: {action_vector} -> Deltas: P={delta_p}, T={delta_t}")

        self._handle_scaling(delta_p, "prompt")
        self._handle_scaling(delta_t, "token")
        
        return True  # 返回 True 表示执行了动作

    def _handle_scaling(self, delta, tag):
        """
        使用扩缩容管理器处理资源的申请与释放
        
        Args:
            delta: 实例数量变化（正数为扩容，负数为缩容）
            tag: 实例标签（"prompt" 或 "token"）
        """
        if delta == 0:
            return

        # 获取当前实例数量
        if tag == "prompt":
            current_instances = self.scheduler.prompt_instances
        elif tag == "token":
            current_instances = self.scheduler.token_instances
        else:
            logging.warning(f"[Action] Unknown tag: {tag}")
            return
        
        # 获取活跃实例（排除扩缩容中的实例）
        active_instances = self.scaling_manager.get_active_instances(current_instances)
        current_count = len(active_instances)
        
        # 获取总实例数（包括所有状态）
        total_count = len(self.scheduler.instances)

        if delta > 0:
            # --- 扩容 (Scale Up) ---
            # [安全约束] 检查预算上限
            remaining_quota = self.max_total_instances - total_count
            actual_add = min(delta, remaining_quota)

            if actual_add > 0:
                for _ in range(actual_add):
                    try:
                        # 使用全流程扩容：自动创建服务器 + 实例
                        instance_cfg = self._get_instance_config(tag)
                        parallelism = self._get_parallelism(tag)
                        
                        if instance_cfg is None or parallelism is None:
                            logging.warning(f"[Action] No configuration found for {tag} instance")
                            break


                        server, instance = self.scaling_manager.scale_up_full(
                            instance_cfg=instance_cfg,
                            parallelism=parallelism,
                            tag=tag,
                            server_sku="dgx-h100"  # 使用默认 SKU
                        )
                        # logging.info(f"[Action] Added {tag} instance {instance.instance_id} on server {server.server_id}")
                    except Exception as e:
                        logging.error(f"[Action] Failed to add {tag} instance: {e}")
                        break
            else:
                # logging.warning(f"[Action Blocked] Max cluster size reached ({self.max_total_instances})")
                pass

        elif delta < 0:
            # --- 缩容 (Scale Down) ---
            want_remove = abs(delta)
            # [安全约束] 保证不缩减到最小值
            actual_remove = min(want_remove, current_count - self.min_instances)

            if actual_remove > 0:
                # 选择负载最低的实例进行缩容
                for _ in range(actual_remove):
                    if len(active_instances) <= self.min_instances:
                        break
                    
                    # 根据标签选择负载最低的实例
                    if tag == "prompt":
                        least_loaded = min(active_instances, key=lambda i: i.sched_pending_tokens)
                    else:  # token
                        least_loaded = min(active_instances, key=lambda i: i.sched_memory)
                    
                    try:
                        # 使用全流程缩容：自动排空实例 + 移除服务器
                        self.scaling_manager.scale_down_full(least_loaded)
                        active_instances.remove(least_loaded)
                        # logging.info(f"[Action] Removed {tag} instance {least_loaded.instance_id}")
                    except Exception as e:
                        logging.error(f"[Action] Failed to remove {tag} instance: {e}")
                        break
            else:
                # logging.warning(f"[Action Blocked] Cannot remove {tag}, min limit reached")
                pass
    
    def _get_instance_config(self, tag):
        """从启动状态配置获取实例配置"""
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_instance_config(tag)
        
        # 回退到默认配置
        logging.warning(f"[Action] No start_state_manager found, using default config")
        class InstanceConfig:
            def __init__(self):
                self.instance_type = "Splitwise"
                self.max_batch_size = 512
                self.max_batch_tokens = 2048
                self.max_preemptions = 4
        
        return InstanceConfig()
    
    def _get_parallelism(self, tag):
        """从启动状态配置获取并行度"""
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_parallelism(tag)
        
        # 回退到默认并行度
        logging.warning(f"[Action] No start_state_manager found, using default parallelism")
        from model import ModelParallelism
        return ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)