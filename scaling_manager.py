"""
Scaling Manager for managing instance and server scaling operations.
"""

import logging
from enum import Enum
from itertools import count
from typing import Optional, Dict

import utils
from simulator import clock, schedule_event


class InstanceStatus(Enum):
    """Instance status for scaling management"""
    ACTIVE = "active"  # 正常运行，可以接收新任务
    SCALING_UP = "scaling_up"  # 正在扩容启动中，不可接收任务
    DRAINING = "draining"  # 排空中，不接收新任务，等待现有任务完成
    SCALING_DOWN = "scaling_down"  # 准备缩容，已排空


class ServerStatus(Enum):
    """Server status for scaling management"""
    ACTIVE = "active"  # 正常运行
    SCALING_UP = "scaling_up"  # 正在启动中
    SCALING_DOWN = "scaling_down"  # 准备关闭


class ScalingManager:
    """
    Scaling Manager manages instance and server scaling operations.
    
    Responsibilities:
    - Handle scale-up with startup delay
    - Handle scale-down with graceful draining
    - Track instance and server status
    - Prevent scheduling to draining/scaling instances
    """
    
    def __init__(self, 
                 application,
                 cluster,
                 scale_up_delay=10.0,  # 扩容启动延迟（秒）
                 drain_check_interval=1.0,  # 排空检查间隔（秒）
                 autoscaling_policy=None,  # 自动扩缩容策略（可选）
                 decision_interval=30.0,  # 自动扩缩容决策间隔（秒）
                 enable_autoscaling=False,  # 是否启用自动扩缩容
                 debug=False):
        self.application = application
        self.cluster = cluster
        self.scale_up_delay = scale_up_delay
        self.drain_check_interval = drain_check_interval
        self.debug = debug
        
        # 跟踪实例状态
        self.instance_status = {}  # instance_id -> InstanceStatus
        
        # 跟踪服务器状态
        self.server_status = {}  # server_id -> ServerStatus
        
        # 跟踪扩容中的实例和服务器
        self.scaling_up_instances = []
        self.scaling_up_servers = []
        
        # 跟踪缩容中的实例和服务器
        self.draining_instances = []
        self.scaling_down_servers = []
        
        # 跟踪等待缩容的服务器（实例ID -> 服务器列表）
        self._pending_server_scale_downs = {}
        
        # 自动扩缩容相关
        self.autoscaling_policy = autoscaling_policy
        self.decision_interval = decision_interval
        self.enable_autoscaling = enable_autoscaling
        self._last_decision_time = -float('inf')
        
        # 用于计算 TPS 等速率指标的累积计数器
        self._last_metrics = {
            'completed_tokens': 0,
            'completed_requests': 0,
            'timestamp': 0.0,
        }
        
        # 日志
        logger_name = f"scaling/{self.application.application_id}"
        level = logging.DEBUG if self.debug else logging.INFO
        self.logger = utils.file_logger(logger_name, level=level)
        self.logger.info("time,action,target,status")
        
        # 注意：不在 __init__ 中调度事件，因为此时模拟器还没有初始化
        # 需要在模拟器启动后调用 start_autoscaling() 方法
    
    def can_schedule_to_instance(self, instance):
        """
        检查实例是否可以接收新的调度任务
        
        Returns:
            bool: True if instance can accept new tasks
        """
        status = self.instance_status.get(instance.instance_id, InstanceStatus.ACTIVE)
        return status == InstanceStatus.ACTIVE
    
    def get_active_instances(self, instances):
        """
        从实例列表中过滤出可以接收任务的活跃实例
        
        Args:
            instances: 实例列表
            
        Returns:
            list: 活跃实例列表
        """
        return [inst for inst in instances if self.can_schedule_to_instance(inst)]
    
    def scale_up_full(self, instance_cfg, parallelism, tag=None, server_sku=None):
        """
        完整的扩容流程：先扩容服务器，再扩容实例
        一个服务器对应一个实例
        
        Args:
            instance_cfg: 实例配置
            parallelism: 模型并行配置
            tag: 实例标签（如 "prompt" 或 "token"）
            server_sku: 服务器 SKU 名称，如果为 None 则使用配置中的第一个
            
        Returns:
            tuple: (server, instance) 新创建的服务器和实例
        """
        # 1. 扩容服务器
        if server_sku is None:
            # 使用配置中的第一个服务器类型
            if self.cluster.cluster_cfg and self.cluster.cluster_cfg.servers:
                server_sku = self.cluster.cluster_cfg.servers[0].sku
            else:
                raise ValueError("No server configuration found")
        
        server_cfg = self.cluster.get_server_config(server_sku)
        if server_cfg is None:
            raise ValueError(f"Server configuration for {server_sku} not found")
        
        # 创建服务器
        server = self.scale_up_server(server_cfg)
        
        # 2. 在新服务器上扩容实例（使用服务器的所有处理器）
        instance = self.scale_up_instance(
            instance_cfg=instance_cfg,
            processors=server.processors,
            parallelism=parallelism,
            tag=tag
        )
        
        # print(f"[ScalingManager] Full scale-up: server {server.server_id} + instance {instance.instance_id}")
        
        return server, instance
    
    def scale_down_full(self, instance):
        """
        完整的缩容流程：先缩容实例，再缩容服务器
        一个服务器对应一个实例
        
        Args:
            instance: 要缩容的实例
        """
        # 记录实例所在的服务器
        instance_servers = list(instance.servers)
        
        # 1. 开始缩容实例（异步等待排空）
        self.scale_down_instance(instance)
        
        # 2. 缩容流程会在实例完全排空后自动缩容服务器
        # 注册回调，在实例缩容完成后缩容服务器
        self._pending_server_scale_downs[instance.instance_id] = instance_servers
        
        # print(f"[ScalingManager] Full scale-down initiated: instance {instance.instance_id}")
    
    def scale_up_instance(self, instance_cfg, processors, parallelism, tag=None):
        """
        扩容：添加新实例（内部方法，通常应使用 scale_up_full）
        
        Args:
            instance_cfg: 实例配置
            processors: 处理器列表
            parallelism: 模型并行配置
            tag: 实例标签（如 "prompt" 或 "token"）
            
        Returns:
            instance: 新创建的实例（处于 SCALING_UP 状态）
        """
        # 使用 allocator 创建实例
        instance = self.application.allocator.start_spin_up_instance(
            instance_cfg=instance_cfg,
            processors=processors,
            parallelism=parallelism,
            pre_start=False,  # 不立即启动
            tag=tag
        )
        
        # 标记实例为扩容中
        self.instance_status[instance.instance_id] = InstanceStatus.SCALING_UP
        self.scaling_up_instances.append(instance)
        
        # 日志
        self.logger.info("%s,scale_up_start,instance_%s,%s", 
                        clock(), instance.instance_id, InstanceStatus.SCALING_UP.value)
        
        # 安排延迟后激活实例
        schedule_event(self.scale_up_delay,
                      lambda inst=instance: self._complete_scale_up(inst))
        
        return instance
    
    def _complete_scale_up(self, instance):
        """
        完成实例扩容，将实例状态设置为 ACTIVE
        并通知调度器尝试调度等待的请求
        """
        self.instance_status[instance.instance_id] = InstanceStatus.ACTIVE
        if instance in self.scaling_up_instances:
            self.scaling_up_instances.remove(instance)
        
        self.logger.info("%s,scale_up_complete,instance_%s,%s", 
                        clock(), instance.instance_id, InstanceStatus.ACTIVE.value)
        
        # print(f"[ScalingManager] Instance {instance.instance_id} scaled up at {clock():.2f}")
        
        # 通知调度器有新实例可用，尝试调度等待的请求
        if hasattr(self.application, 'scheduler'):
            scheduler = self.application.scheduler
            # 检查调度器是否有 on_instance_available 方法
            if hasattr(scheduler, 'on_instance_available'):
                scheduler.on_instance_available()
                if self.debug:
                    print(f"[ScalingManager] Notified scheduler of new instance {instance.instance_id} at {clock():.2f}")
    
    def scale_down_instance(self, instance):
        """
        缩容：开始排空实例
        
        Args:
            instance: 要缩容的实例
        """
        current_status = self.instance_status.get(instance.instance_id, InstanceStatus.ACTIVE)
        
        if current_status != InstanceStatus.ACTIVE:
            self.logger.warning("%s,scale_down_invalid,instance_%s,%s", 
                              clock(), instance.instance_id, current_status.value)
            print(f"[ScalingManager] Cannot scale down instance {instance.instance_id} - status: {current_status}")
            return
        
        # 标记实例为排空中
        self.instance_status[instance.instance_id] = InstanceStatus.DRAINING
        self.draining_instances.append(instance)
        
        self.logger.info("%s,scale_down_start,instance_%s,%s", 
                        clock(), instance.instance_id, InstanceStatus.DRAINING.value)
        
        # print(f"[ScalingManager] Instance {instance.instance_id} draining started at {clock():.2f}")
        
        # 开始检查排空状态
        self._check_drain_status(instance)
    
    def _check_drain_status(self, instance):
        """
        检查实例是否已排空（所有任务完成）
        """
        # 检查实例是否还有任务
        has_tasks = (len(instance.pending_queue) > 0 or 
                    len(instance.batch) > 0 or
                    len(getattr(instance, 'blocked_queue', [])) > 0)
        
        if has_tasks:
            # 还有任务，继续等待
            schedule_event(self.drain_check_interval,
                          lambda inst=instance: self._check_drain_status(inst))
        else:
            # 已排空，完成缩容
            self._complete_scale_down(instance)
    
    def _complete_scale_down(self, instance):
        """
        完成实例缩容，移除实例，并在需要时缩容服务器
        """
        instance_id = instance.instance_id
        
        self.instance_status[instance_id] = InstanceStatus.SCALING_DOWN
        if instance in self.draining_instances:
            self.draining_instances.remove(instance)
        
        # 从应用和调度器中移除实例
        if instance in self.application.instances:
            self.application.instances.remove(instance)
        
        if instance in self.application.scheduler.instances:
            self.application.scheduler.instances.remove(instance)
        
        # 从 prompt_instances 或 token_instances 中移除（如果适用）
        if hasattr(self.application.scheduler, 'prompt_instances'):
            if instance in self.application.scheduler.prompt_instances:
                self.application.scheduler.prompt_instances.remove(instance)
        
        if hasattr(self.application.scheduler, 'token_instances'):
            if instance in self.application.scheduler.token_instances:
                self.application.scheduler.token_instances.remove(instance)
        
        if hasattr(self.application.scheduler, 'mixed_instances'):
            if instance in self.application.scheduler.mixed_instances:
                self.application.scheduler.mixed_instances.remove(instance)
        
        # 从处理器中移除实例
        for processor in instance.processors:
            if instance in processor.instances:
                processor.instances.remove(instance)
        
        self.logger.info("%s,scale_down_complete,instance_%s,%s", 
                        clock(), instance_id, InstanceStatus.SCALING_DOWN.value)
        
        # print(f"[ScalingManager] Instance {instance_id} scaled down at {clock():.2f}")
        
        # 如果是全流程缩容，检查是否需要缩容服务器
        if instance_id in self._pending_server_scale_downs:
            servers = self._pending_server_scale_downs[instance_id]
            del self._pending_server_scale_downs[instance_id]
            
            # 缩容所有相关服务器（一个实例对应一个服务器）
            for server in servers:
                # 检查服务器是否还有其他实例
                if len(server.instances) == 0:
                    self.scale_down_server(server)
                    # print(f"[ScalingManager] Auto scale-down server {server.server_id} after instance removal")
                else:
                    print(f"[ScalingManager] Server {server.server_id} still has {len(server.instances)} instances, skipping scale-down")
    
    def scale_up_server(self, server_cfg):
        """
        扩容：添加新服务器
        
        Args:
            server_cfg: 服务器配置
            
        Returns:
            server: 新创建的服务器
        """
        import hardware_repo
        from server import Server
        
        # 生成新的 server_id
        max_id = -1
        for servers_at_sku in self.cluster.servers.values():
            for server in servers_at_sku:
                if server.server_id > max_id:
                    max_id = server.server_id
        server_id = max_id + 1
        
        # 创建服务器
        sku_cfg = hardware_repo.get_sku_config(server_cfg.sku)
        server = Server.from_config(sku_cfg, server_id=server_id)
        
        # 添加到集群
        sku_name = server_cfg.sku
        if sku_name not in self.cluster.servers:
            self.cluster.servers[sku_name] = []
        self.cluster.servers[sku_name].append(server)
        
        # 标记服务器为扩容中
        self.server_status[server.server_id] = ServerStatus.SCALING_UP
        self.scaling_up_servers.append(server)
        
        self.logger.info("%s,scale_up_server,server_%s,%s", 
                        clock(), server.server_id, ServerStatus.SCALING_UP.value)
        
        # 服务器启动（运行初始化）
        server.run()
        
        # 标记为活跃
        self.server_status[server.server_id] = ServerStatus.ACTIVE
        self.scaling_up_servers.remove(server)
        
        # print(f"[ScalingManager] Server {server.server_id} scaled up at {clock():.2f}")
        
        return server
    
    def scale_down_server(self, server):
        """
        缩容：移除服务器（需要先排空所有实例）
        
        Args:
            server: 要缩容的服务器
        """
        # 检查服务器上是否还有实例
        if len(server.instances) > 0:
            self.logger.warning("%s,scale_down_server_invalid,server_%s,has_instances", 
                              clock(), server.server_id)
            print(f"[ScalingManager] Cannot scale down server {server.server_id} - has {len(server.instances)} instances")
            return
        
        # 标记服务器为缩容中
        self.server_status[server.server_id] = ServerStatus.SCALING_DOWN
        
        # 从集群中移除
        for sku_name in self.cluster.servers:
            if server in self.cluster.servers[sku_name]:
                self.cluster.servers[sku_name].remove(server)
                break
        
        self.logger.info("%s,scale_down_server,server_%s,%s", 
                        clock(), server.server_id, ServerStatus.SCALING_DOWN.value)
        
        # print(f"[ScalingManager] Server {server.server_id} scaled down at {clock():.2f}")
    
    def get_status_summary(self):
        """
        获取扩缩容状态摘要
        
        Returns:
            dict: 状态摘要
        """
        return {
            "active_instances": sum(1 for s in self.instance_status.values() if s == InstanceStatus.ACTIVE),
            "scaling_up_instances": len(self.scaling_up_instances),
            "draining_instances": len(self.draining_instances),
            "active_servers": sum(1 for s in self.server_status.values() if s == ServerStatus.ACTIVE),
            "scaling_up_servers": len(self.scaling_up_servers),
            "pending_server_scale_downs": len(self._pending_server_scale_downs),
            "total_servers": sum(len(servers) for servers in self.cluster.servers.values()),
            "total_instances": len(self.application.instances),
        }
    
    # ==================== 自动扩缩容相关方法 ====================
    
    def collect_metrics(self) -> Dict:
        """
        收集系统指标用于自动扩缩容决策
        参考 RL/state.py 的实现方式
        
        Returns:
            dict: 包含各种系统指标的字典
        """
        current_time = clock()
        
        # 计算时间间隔
        if self._last_metrics['timestamp'] > 0:
            interval = current_time - self._last_metrics['timestamp']
        else:
            interval = self.decision_interval
        
        # 防止除零
        if interval <= 0:
            interval = self.decision_interval
        
        metrics = {
            'timestamp': current_time,
        }
        
        # 统计当前 Prefill 和 Decode 实例数
        prompt_instances = []
        token_instances = []
        mixed_instances = []
        
        scheduler = self.application.scheduler
        if hasattr(scheduler, 'prompt_instances'):
            prompt_instances = [inst for inst in scheduler.prompt_instances 
                              if self.can_schedule_to_instance(inst)]
        if hasattr(scheduler, 'token_instances'):
            token_instances = [inst for inst in scheduler.token_instances 
                             if self.can_schedule_to_instance(inst)]
        if hasattr(scheduler, 'mixed_instances'):
            mixed_instances = [inst for inst in scheduler.mixed_instances 
                             if self.can_schedule_to_instance(inst)]
        
        metrics['current_prompt_instances'] = len(prompt_instances)
        metrics['current_token_instances'] = len(token_instances)
        metrics['current_mixed_instances'] = len(mixed_instances)
        
        # 收集 Decode TPS（每秒解码 Token 数）
        # 参考 RL/state.py: 从 router.total_complete_token 获取累积值，计算增量
        if hasattr(self.application, 'router') and hasattr(self.application.router, 'total_complete_token'):
            curr_tokens = self.application.router.total_complete_token
            delta_tokens = curr_tokens - self._last_metrics['completed_tokens']
            decode_tps_total = delta_tokens / interval
            
            # 更新累积状态
            self._last_metrics['completed_tokens'] = curr_tokens
        else:
            # 降级方案：如果 router 不可用，设为 0
            decode_tps_total = 0.0
        
        metrics['decode_tps'] = decode_tps_total
        
        # 收集 Prefill TPS（每秒完成的 Prompt 请求数）
        # 从 router.total_arrivals 获取累积值，计算增量
        if hasattr(self.application, 'router') and hasattr(self.application.router, 'total_arrivals'):
            curr_arrivals = self.application.router.total_arrivals
            delta_arrivals = curr_arrivals - self._last_metrics['completed_requests']
            prefill_tps_total = delta_arrivals / interval
            
            # 更新累积状态
            self._last_metrics['completed_requests'] = curr_arrivals
        else:
            # 降级方案：如果 router 不可用，设为 0
            prefill_tps_total = 0.0
        
        metrics['prefill_tps'] = prefill_tps_total
        
        # 收集 GPU 利用率（基于显存使用率）
        prefill_gpu_util = 0.0
        decode_gpu_util = 0.0
        
        # Prefill GPU 利用率（显存使用率）
        if len(prompt_instances) > 0:
            for instance in prompt_instances:
                # 使用实例的显存利用率：memory / max_memory
                if instance.max_memory > 0:
                    prefill_gpu_util += instance.memory / instance.max_memory
            prefill_gpu_util /= len(prompt_instances)
        
        # Decode GPU 利用率（显存使用率）
        if len(token_instances) > 0:
            for instance in token_instances:
                # 使用实例的显存利用率：memory / max_memory
                if instance.max_memory > 0:
                    decode_gpu_util += instance.memory / instance.max_memory
            decode_gpu_util /= len(token_instances)
        
        metrics['prefill_gpu_util'] = prefill_gpu_util
        metrics['decode_gpu_util'] = decode_gpu_util
        
        # 收集延迟指标（TTFT 和 TBT）
        # 使用原始数据（未归一化）用于扩缩容决策
        # 参考 RL/state.py: 从 scheduler 获取
        if hasattr(scheduler, 'get_period_raw_result'):
            try:
                # 获取原始的（未归一化的）延迟数据
                ttft_raw, tbt_raw = scheduler.get_period_raw_result()
                # ttft_raw 和 tbt_raw 是列表 [p50, p90, p99]，使用 p50 作为代表值
                metrics['ttft'] = ttft_raw[0] if len(ttft_raw) > 0 else 0.0
                metrics['tbt'] = tbt_raw[0] if len(tbt_raw) > 0 else 0.0
                # 也保存完整的分位数信息，供高级策略使用
                metrics['ttft_percentiles'] = ttft_raw  # [p50, p90, p99]
                metrics['tbt_percentiles'] = tbt_raw    # [p50, p90, p99]
                
                # 如果需要归一化数据和 SLO 违规率，也可以获取
                if hasattr(scheduler, 'get_period_normalized_result'):
                    try:
                        ttft_norm, tbt_norm, vio_slo_rate = scheduler.get_period_normalized_result()
                        metrics['ttft_normalized'] = ttft_norm[0] if len(ttft_norm) > 0 else 0.0
                        metrics['tbt_normalized'] = tbt_norm[0] if len(tbt_norm) > 0 else 0.0
                        metrics['ttft_normalized_percentiles'] = ttft_norm
                        metrics['tbt_normalized_percentiles'] = tbt_norm
                        metrics['slo_violation_rate'] = vio_slo_rate  # [ttft_vio, tbt_vio]
                    except:
                        pass
            except Exception as e:
                if self.debug:
                    self.logger.warning(f"Failed to get period raw result: {e}")
                metrics['ttft'] = 0.0
                metrics['tbt'] = 0.0
                metrics['ttft_percentiles'] = [0.0, 0.0, 0.0]
                metrics['tbt_percentiles'] = [0.0, 0.0, 0.0]
        else:
            # 降级方案：如果 scheduler 没有 get_period_raw_result 方法
            metrics['ttft'] = 0.0
            metrics['tbt'] = 0.0
            metrics['ttft_percentiles'] = [0.0, 0.0, 0.0]
            metrics['tbt_percentiles'] = [0.0, 0.0, 0.0]
        
        # 队列长度
        total_pending = sum(len(inst.pending_queue) for inst in self.application.instances)
        metrics['pending_queue_length'] = total_pending
        
        # 更新时间戳
        self._last_metrics['timestamp'] = current_time
        
        return metrics
    
    def _make_autoscaling_decision(self):
        """
        执行自动扩缩容决策
        
        这个方法会被周期性调用，使用配置的策略做出扩缩容决策
        """
        if not self.enable_autoscaling or self.autoscaling_policy is None:
            return
        
        # 收集指标
        metrics = self.collect_metrics()
        
        # 调用策略做决策
        action = self.autoscaling_policy.decide(metrics)
        
        if self.debug:
            print(f"[ScalingManager] Autoscaling decision at {clock():.2f}: {action}")
        
        # 执行扩缩容动作
        self._execute_scaling_action(action, metrics)
        
        # 安排下一次决策
        schedule_event(self.decision_interval, lambda: self._make_autoscaling_decision())
    
    def _execute_scaling_action(self, action, metrics: Dict):
        """
        执行扩缩容动作
        
        Args:
            action: ScalingAction 对象
            metrics: 当前系统指标
        """
        from autoscaling_policies import ScalingDecision
        
        if action.decision == ScalingDecision.NO_CHANGE:
            return
        
        current_prompt = metrics['current_prompt_instances']
        current_token = metrics['current_token_instances']
        target_prompt = action.target_prompt_instances
        target_token = action.target_token_instances
        
        self.logger.info("%s,autoscaling_decision,%s,prompt:%d->%d_token:%d->%d,%s",
                        clock(), action.decision.value,
                        current_prompt, target_prompt,
                        current_token, target_token,
                        action.reason)
        
        if self.debug:
            print(f"[ScalingManager] Executing action: {action}")
        
        # 执行 Prefill 实例扩缩容
        if target_prompt > current_prompt:
            # 扩容 Prefill 实例
            num_to_add = target_prompt - current_prompt
            for _ in range(num_to_add):
                self._autoscale_add_prompt_instance()
        elif target_prompt < current_prompt:
            # 缩容 Prefill 实例
            num_to_remove = current_prompt - target_prompt
            self._autoscale_remove_instances('prompt', num_to_remove)
        
        # 执行 Decode 实例扩缩容
        if target_token > current_token:
            # 扩容 Decode 实例
            num_to_add = target_token - current_token
            for _ in range(num_to_add):
                self._autoscale_add_token_instance()
        elif target_token < current_token:
            # 缩容 Decode 实例
            num_to_remove = current_token - target_token
            self._autoscale_remove_instances('token', num_to_remove)
    
    def _autoscale_add_prompt_instance(self):
        """自动扩容添加一个 Prefill 实例"""
        # 获取配置（参考 action.py 的实现）
        instance_cfg = self._get_instance_config("prompt")
        parallelism = self._get_parallelism("prompt")
        
        if instance_cfg is None or parallelism is None:
            self.logger.warning("No configuration found for prompt instance")
            return
        
        try:
            server, instance = self.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="prompt"
            )
            
            # 添加到调度器的 prompt_instances
            if hasattr(self.application.scheduler, 'prompt_instances'):
                # 等待实例启动完成后再添加到调度器
                # （_complete_scale_up 中已经有通知调度器的逻辑）
                pass
            
            if self.debug:
                print(f"[ScalingManager] Auto-added prompt instance {instance.instance_id}")
        except Exception as e:
            self.logger.error(f"Failed to add prompt instance: {e}")
            print(f"[ScalingManager] Error adding prompt instance: {e}")
    
    def _autoscale_add_token_instance(self):
        """自动扩容添加一个 Decode 实例"""
        # 获取配置（参考 action.py 的实现）
        instance_cfg = self._get_instance_config("token")
        parallelism = self._get_parallelism("token")
        
        if instance_cfg is None or parallelism is None:
            self.logger.warning("No configuration found for token instance")
            return
        
        try:
            server, instance = self.scale_up_full(
                instance_cfg=instance_cfg,
                parallelism=parallelism,
                tag="token"
            )
            
            if self.debug:
                print(f"[ScalingManager] Auto-added token instance {instance.instance_id}")
        except Exception as e:
            self.logger.error(f"Failed to add token instance: {e}")
            print(f"[ScalingManager] Error adding token instance: {e}")
    
    def _autoscale_remove_instances(self, instance_type: str, num_to_remove: int):
        """
        自动缩容移除指定数量的实例
        
        Args:
            instance_type: "prompt" 或 "token"
            num_to_remove: 要移除的实例数量
        """
        scheduler = self.application.scheduler
        
        if instance_type == "prompt":
            if not hasattr(scheduler, 'prompt_instances'):
                return
            candidates = [inst for inst in scheduler.prompt_instances
                         if self.can_schedule_to_instance(inst)]
        else:  # token
            if not hasattr(scheduler, 'token_instances'):
                return
            candidates = [inst for inst in scheduler.token_instances
                         if self.can_schedule_to_instance(inst)]
        
        # 选择要移除的实例：优先移除负载最低的实例
        # 负载定义为：待处理队列长度 + 当前批次大小
        def get_instance_load(inst):
            """计算实例负载"""
            queue_load = len(inst.pending_queue)
            batch_load = len(inst.batch) if hasattr(inst, 'batch') else 0
            # 如果有 blocked_queue，也计入负载
            blocked_load = len(inst.blocked_queue) if hasattr(inst, 'blocked_queue') else 0
            return queue_load + batch_load + blocked_load
        
        # 按负载从低到高排序，优先移除负载低的实例
        candidates.sort(key=get_instance_load)
        
        num_removed = 0
        for instance in candidates:
            if num_removed >= num_to_remove:
                break
            
            # 检查实例是否可以缩容（没有正在扩容或缩容中）
            status = self.instance_status.get(instance.instance_id, InstanceStatus.ACTIVE)
            if status == InstanceStatus.ACTIVE:
                load = get_instance_load(instance)
                self.scale_down_full(instance)
                num_removed += 1
                
                if self.debug:
                    print(f"[ScalingManager] Auto-removed {instance_type} instance {instance.instance_id} "
                          f"(load={load}: pending={len(instance.pending_queue)}, "
                          f"batch={len(instance.batch) if hasattr(instance, 'batch') else 0})")
        
        if num_removed < num_to_remove:
            self.logger.warning(
                f"Could only remove {num_removed}/{num_to_remove} {instance_type} instances"
            )
    
    def set_autoscaling_policy(self, policy):
        """
        设置自动扩缩容策略
        
        Args:
            policy: AutoscalingPolicy 的实例
        """
        self.autoscaling_policy = policy
        if self.debug:
            print(f"[ScalingManager] Set autoscaling policy to {policy.name}")
    
    def start_autoscaling(self):
        """
        启动自动扩缩容循环
        
        应该在模拟器初始化完成后调用（例如在 init_start_state 之后）
        """
        if self.enable_autoscaling and self.autoscaling_policy is not None:
            # 安排第一次决策
            schedule_event(self.decision_interval, lambda: self._make_autoscaling_decision())
            if self.debug:
                print(f"[ScalingManager] Started autoscaling loop with interval {self.decision_interval}s")
        else:
            if self.debug:
                print("[ScalingManager] Autoscaling not started (disabled or no policy)")
    
    def enable_autoscaling_loop(self):
        """启用自动扩缩容循环（如果已经在运行中则不重复启动）"""
        if not self.enable_autoscaling:
            self.enable_autoscaling = True
            # 如果模拟器已经启动，立即调度
            try:
                schedule_event(self.decision_interval, lambda: self._make_autoscaling_decision())
                if self.debug:
                    print(f"[ScalingManager] Enabled autoscaling loop with interval {self.decision_interval}s")
            except:
                # 如果 schedule_event 失败（模拟器未初始化），等待 start_autoscaling() 调用
                if self.debug:
                    print("[ScalingManager] Autoscaling enabled, waiting for simulator initialization")
    
    def disable_autoscaling_loop(self):
        """禁用自动扩缩容循环"""
        self.enable_autoscaling = False
        if self.debug:
            print("[ScalingManager] Disabled autoscaling loop")
    
    def _get_instance_config(self, tag):
        """
        从启动状态配置获取实例配置
        参考 action.py 的实现
        
        Args:
            tag: 实例标签（"prompt" 或 "token"）
        
        Returns:
            实例配置对象
        """
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_instance_config(tag)
        
        # 回退到默认配置
        self.logger.warning(f"No start_state_manager found, using default config for {tag}")
        
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
        参考 action.py 的实现
        
        Args:
            tag: 实例标签（"prompt" 或 "token"）
        
        Returns:
            ModelParallelism 对象
        """
        if hasattr(self.application, 'start_state_manager') and \
           self.application.start_state_manager is not None:
            return self.application.start_state_manager.get_parallelism(tag)
        
        # 回退到默认并行度
        self.logger.warning(f"No start_state_manager found, using default parallelism for {tag}")
        from model import ModelParallelism
        return ModelParallelism(pipeline_parallelism=1, tensor_parallelism=1)

