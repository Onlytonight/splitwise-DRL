"""
示例：带自动扩缩容功能的调度器

展示如何在调度器中集成扩缩容管理器来实现自动扩缩容。
"""

import logging
from scheduler import KVScheduler
from scaling_manager import InstanceStatus
from simulator import clock


class AutoScalingScheduler(KVScheduler):
    """
    带自动扩缩容的调度器
    
    功能：
    1. 监控实例负载
    2. 当负载过高时自动扩容
    3. 当负载过低时自动缩容
    4. 确保最小实例数
    """
    
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 # 自动扩缩容参数
                 min_prompt_instances=1,
                 max_prompt_instances=100,
                 min_token_instances=1,
                 max_token_instances=100,
                 scale_up_threshold=0.8,    # 负载超过 80% 时扩容
                 scale_down_threshold=0.3,  # 负载低于 30% 时缩容
                 check_interval=100,         # 每 100 个请求检查一次
                 debug=False):
        super().__init__(application,
                        router,
                        overheads,
                        executor_overheads,
                        prompt_processors,
                        token_processors,
                        debug)
        
        self.transfer_bandwidth = transfer_bandwidth * 1024**3
        
        # 扩缩容参数
        self.min_prompt_instances = min_prompt_instances
        self.max_prompt_instances = max_prompt_instances
        self.min_token_instances = min_token_instances
        self.max_token_instances = max_token_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.check_interval = check_interval
        
        # 计数器
        self.schedule_counter = 0
        self.last_scale_time = 0
        self.scale_cooldown = 5.0  # 扩缩容冷却时间（秒）
        
        # 日志
        if self.debug:
            self.scaling_logger = logging.getLogger(f"autoscaling_{application.application_id}")
    
    def schedule(self, request, *args, **kwargs):
        """
        调度请求，并检查是否需要扩缩容
        """
        # 每隔一定数量的请求检查扩缩容
        self.schedule_counter += 1
        if self.schedule_counter % self.check_interval == 0:
            self.check_and_scale()
        
        # 执行正常的调度逻辑
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        
        # 获取可调度的实例
        prompt_instances = self.get_schedulable_prompt_instances()
        token_instances = self.get_schedulable_token_instances()
        
        if len(prompt_instances) == 0 or len(token_instances) == 0:
            print(f"[AutoScaling] No instances available at {clock():.2f}")
            raise ValueError("No instances available")
        
        # 选择负载最低的实例
        prompt_instance = min(prompt_instances,
                             key=lambda i: i.sched_pending_tokens)
        token_instance = min(token_instances,
                           key=lambda i: i.sched_memory)
        
        # 传输 KV 缓存
        self.add_kv_cache_transfer(request,
                                   prompt_instance,
                                   token_instance,
                                   self.transfer_bandwidth)
        
        # 更新调度器内存计数
        prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance)
        token_instance.sched_memory += (prompt_task.max_memory(token_instance) + 
                                       token_task.max_memory(token_instance))
        
        # 更新待处理令牌数
        prompt_instance.sched_pending_tokens += prompt_task.prompt_size
        token_instance.sched_pending_tokens += 1
    
    def check_and_scale(self):
        """
        检查负载并决定是否扩缩容
        """
        current_time = clock()
        
        # 冷却时间检查
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        # 检查是否有扩缩容管理器
        if not hasattr(self.application, 'scaling_manager') or \
           self.application.scaling_manager is None:
            return
        
        scaling_manager = self.application.scaling_manager
        
        # 获取活跃实例（排除扩缩容中的实例）
        active_prompt_instances = scaling_manager.get_active_instances(self.prompt_instances)
        active_token_instances = scaling_manager.get_active_instances(self.token_instances)
        
        # 计算负载
        prompt_load = self._calculate_prompt_load(active_prompt_instances)
        token_load = self._calculate_token_load(active_token_instances)
        
        print(f"[AutoScaling] Check at {current_time:.2f} - "
              f"Prompt load: {prompt_load:.2%}, Token load: {token_load:.2%}")
        
        # 决策：Prompt 实例
        if prompt_load > self.scale_up_threshold and \
           len(self.prompt_instances) < self.max_prompt_instances:
            self._scale_up_prompt_instance()
            self.last_scale_time = current_time
        elif prompt_load < self.scale_down_threshold and \
             len(active_prompt_instances) > self.min_prompt_instances:
            self._scale_down_prompt_instance(active_prompt_instances)
            self.last_scale_time = current_time
        
        # 决策：Token 实例
        if token_load > self.scale_up_threshold and \
           len(self.token_instances) < self.max_token_instances:
            self._scale_up_token_instance()
            self.last_scale_time = current_time
        elif token_load < self.scale_down_threshold and \
             len(active_token_instances) > self.min_token_instances:
            self._scale_down_token_instance(active_token_instances)
            self.last_scale_time = current_time
    
    def _calculate_prompt_load(self, instances):
        """
        计算 Prompt 实例的平均负载
        使用待处理令牌数作为负载指标
        """
        if not instances:
            return 0.0
        
        # 假设每个实例的最大容量是某个值（例如 10000 tokens）
        max_capacity_per_instance = 10000
        
        total_load = sum(inst.sched_pending_tokens for inst in instances)
        total_capacity = len(instances) * max_capacity_per_instance
        
        return total_load / total_capacity if total_capacity > 0 else 0.0
    
    def _calculate_token_load(self, instances):
        """
        计算 Token 实例的平均负载
        使用内存使用率作为负载指标
        """
        if not instances:
            return 0.0
        
        total_usage = sum(inst.sched_memory for inst in instances)
        total_capacity = sum(inst.max_memory for inst in instances)
        
        return total_usage / total_capacity if total_capacity > 0 else 0.0
    
    def _scale_up_prompt_instance(self):
        """
        扩容一个 Prompt 实例（使用全流程方法）
        """
        print(f"[AutoScaling] Scaling up prompt instance at {clock():.2f}")
        
        # 从配置获取实例配置和并行度
        instance_cfg = self._get_instance_config("prompt")
        parallelism = self._get_parallelism("prompt")
        
        if instance_cfg is None or parallelism is None:
            print(f"[AutoScaling] No configuration found for prompt instance")
            return
        
        # 使用全流程扩容：自动创建服务器 + 实例
        try:
            server, instance = self.application.scaling_manager.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="prompt",
                server_sku=None  # 使用默认 SKU
            )
            print(f"[AutoScaling] Created server {server.server_id} with prompt instance {instance.instance_id}")
        except Exception as e:
            print(f"[AutoScaling] Failed to scale up: {e}")
    
    def _scale_down_prompt_instance(self, instances):
        """
        缩容一个 Prompt 实例（选择负载最低的，使用全流程方法）
        """
        if not instances:
            return
        
        # 选择负载最低的实例
        least_loaded = min(instances, key=lambda i: i.sched_pending_tokens)
        
        print(f"[AutoScaling] Scaling down prompt instance {least_loaded.instance_id} at {clock():.2f}")
        
        # 使用全流程缩容：自动排空实例 + 移除服务器
        self.application.scaling_manager.scale_down_full(least_loaded)
    
    def _scale_up_token_instance(self):
        """
        扩容一个 Token 实例（使用全流程方法）
        """
        print(f"[AutoScaling] Scaling up token instance at {clock():.2f}")
        
        # 从配置获取实例配置和并行度
        instance_cfg = self._get_instance_config("token")
        parallelism = self._get_parallelism("token")
        
        if instance_cfg is None or parallelism is None:
            print(f"[AutoScaling] No configuration found for token instance")
            return
        
        # 使用全流程扩容：自动创建服务器 + 实例
        try:
            server, instance = self.application.scaling_manager.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="token",
                server_sku=None  # 使用默认 SKU
            )
            print(f"[AutoScaling] Created server {server.server_id} with token instance {instance.instance_id}")
        except Exception as e:
            print(f"[AutoScaling] Failed to scale up: {e}")
    
    def _scale_down_token_instance(self, instances):
        """
        缩容一个 Token 实例（选择负载最低的，使用全流程方法）
        """
        if not instances:
            return
        
        # 选择负载最低的实例
        least_loaded = min(instances, key=lambda i: i.sched_memory)
        
        print(f"[AutoScaling] Scaling down token instance {least_loaded.instance_id} at {clock():.2f}")
        
        # 使用全流程缩容：自动排空实例 + 移除服务器
        self.application.scaling_manager.scale_down_full(least_loaded)
    
    
    def _get_instance_config(self, tag):
        """
        从启动状态配置获取实例配置
        
        Args:
            tag: 实例标签（"prompt" 或 "token"）
            
        Returns:
            instance_cfg: 实例配置对象
        """
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_instance_config(tag)
        
        # 如果没有配置管理器，返回默认配置
        print(f"[AutoScaling] Warning: No start_state_manager found, using default config")
        class InstanceConfig:
            def __init__(self):
                self.instance_type = "Splitwise"
                self.max_batch_size = 64
                self.max_batch_tokens = 4096
                self.max_preemptions = 3
        
        return InstanceConfig()
    
    def _get_parallelism(self, tag):
        """
        从启动状态配置获取并行度
        
        Args:
            tag: 实例标签（"prompt" 或 "token"）
            
        Returns:
            ModelParallelism: 并行度对象
        """
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_parallelism(tag)
        
        # 如果没有配置管理器，返回默认并行度
        print(f"[AutoScaling] Warning: No start_state_manager found, using default parallelism")
        from model import ModelParallelism
        return ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)


