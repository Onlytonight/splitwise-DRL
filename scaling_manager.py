"""
Scaling Manager for managing instance and server scaling operations.
"""

import logging
from enum import Enum
from itertools import count

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
        
        # 跟踪实例状态变化时间（用于计算使用时间）
        # instance_id -> [(status, start_time), ...]
        self.instance_status_history = {}  # 实例状态历史记录
        self.instance_status_start_time = {}  # 当前状态的开始时间 instance_id -> start_time
        self.instance_tag = {}  # 实例标签映射 instance_id -> tag ("prompt" 或 "token")
        self._last_calculation_time = None  # 上次计算总使用时间的时间戳
        self._last_calculation_token_time =None
        self._last_calculation_prompt_time = None
        # 记录排空完成和启动成功的时间戳
        self.instance_drain_complete_time = {}  # instance_id -> drain_complete_time
        self.instance_startup_complete_time = {}  # instance_id -> startup_complete_time
        # 日志
        logger_name = f"scaling/{self.application.application_id}"
        level = logging.DEBUG if self.debug else logging.INFO
        self.logger = utils.file_logger(logger_name, level=level)
        self.logger.info("time,action,target,status")
    
    def _record_status_change(self, instance_id, new_status):
        """
        记录实例状态变化的时间戳
        
        Args:
            instance_id: 实例ID
            new_status: 新状态
        """
        current_time = clock()
        
        # 如果实例已有状态历史，记录上一个状态的结束时间
        if instance_id in self.instance_status_start_time:
            old_status = self.instance_status.get(instance_id)
            if old_status is not None:
                start_time = self.instance_status_start_time[instance_id]
                duration = current_time - start_time
                
                # 记录到历史
                if instance_id not in self.instance_status_history:
                    self.instance_status_history[instance_id] = []
                self.instance_status_history[instance_id].append({
                    'status': old_status,
                    'start_time': start_time,
                    'end_time': current_time,
                    'duration': duration
                })
        
        # 更新当前状态和开始时间
        self.instance_status[instance_id] = new_status
        self.instance_status_start_time[instance_id] = current_time
    
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
        
        # 记录实例的 tag
        if tag is not None:
            self.instance_tag[instance.instance_id] = tag
        elif hasattr(instance, 'tag') and instance.tag:
            self.instance_tag[instance.instance_id] = instance.tag
        
        # 标记实例为扩容中
        self._record_status_change(instance.instance_id, InstanceStatus.SCALING_UP)
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
        self._record_status_change(instance.instance_id, InstanceStatus.ACTIVE)
        if instance in self.scaling_up_instances:
            self.scaling_up_instances.remove(instance)
        
        # 记录启动成功时间戳
        current_time = clock()
        self.instance_startup_complete_time[instance.instance_id] = current_time
        
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
        self._record_status_change(instance.instance_id, InstanceStatus.DRAINING)
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
            # 已排空，记录排空完成时间戳
            current_time = clock()
            self.instance_drain_complete_time[instance.instance_id] = current_time
            # 完成缩容
            self._complete_scale_down(instance)
    
    def _complete_scale_down(self, instance):
        """
        完成实例缩容，移除实例，并在需要时缩容服务器
        """
        instance_id = instance.instance_id
        
        self._record_status_change(instance_id, InstanceStatus.SCALING_DOWN)
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
    
    def calculate_total_instance_time(self, start_time=None, end_time=None, interval=None, tag=None):
        """
        计算指定时间间隔内所有实例的总使用时间
        包括活跃时间、扩容启动时间和缩容排空时间
        
        Args:
            start_time: 时间间隔开始时间（如果为 None，则使用上次调用时间或 0）
            end_time: 时间间隔结束时间（如果为 None，则使用当前时间）
            interval: 时间间隔（秒），如果提供，则 start_time = end_time - interval
            tag: 实例标签过滤（"prompt" 或 "token"），如果为 None 则计算所有实例
            
        Returns:
            float: 总使用时间（秒），包括：
                - ACTIVE 状态的时间
                - SCALING_UP 状态的时间（扩容启动时间）
                - DRAINING 状态的时间（缩容排空时间）
        """
        current_time = clock()
        
        # 处理参数：支持 interval 参数
        if interval is not None:
            if end_time is None:
                end_time = current_time
            if start_time is None:
                start_time = end_time - interval
        else:
            if end_time is None:
                end_time = current_time
            if start_time is None:
                # 如果没有提供 start_time，使用上次记录的最早时间或 0
                start_time = 0.0
                if self.instance_status_history:
                    for records in self.instance_status_history.values():
                        if records:
                            earliest = min(r['start_time'] for r in records)
                            start_time = min(start_time, earliest) if start_time > 0 else earliest
        
        total_time = 0.0
        
        # 遍历所有实例（包括历史记录中的、当前状态中的，以及 application.instances 中的所有实例）
        # 这样可以确保包括启动时通过 pre_start=True 创建的、未通过 ScalingManager 记录的实例
        all_instance_ids = set(self.instance_status_history.keys())
        all_instance_ids.update(self.instance_status.keys())
        # 添加所有当前存在的实例ID
        for inst in self.application.instances:
            all_instance_ids.add(inst.instance_id)
        
        for instance_id in all_instance_ids:
            # 如果指定了 tag，则过滤实例
            instance_tag = self.instance_tag.get(instance_id)
            # 如果实例没有 tag 记录，尝试从 application.instances 中查找
            if instance_tag is None:
                for inst in self.application.instances:
                    if inst.instance_id == instance_id:
                        instance_tag = getattr(inst, 'tag', None)
                        if instance_tag:
                            self.instance_tag[instance_id] = instance_tag
                        break
            
            if tag is not None and instance_tag != tag:
                continue
            
            instance_total_time = 0.0
            
            # 检查实例是否在 ScalingManager 的状态记录系统中
            has_status_record = (instance_id in self.instance_status or 
                                instance_id in self.instance_status_history)
            
            if has_status_record:
                # 实例在 ScalingManager 中有状态记录，使用原有的计算逻辑
                # 1. 计算历史记录中在时间间隔内的部分
                if instance_id in self.instance_status_history:
                    for record in self.instance_status_history[instance_id]:
                        # 计算记录与时间间隔的重叠部分
                        record_start = max(record['start_time'], start_time)
                        record_end = min(record['end_time'], end_time)
                        
                        if record_start < record_end:
                            # 只计算 ACTIVE、SCALING_UP、DRAINING 状态的时间
                            if record['status'] in [InstanceStatus.ACTIVE, 
                                                    InstanceStatus.SCALING_UP, 
                                                    InstanceStatus.DRAINING]:
                                instance_total_time += (record_end - record_start)
                
                # 2. 计算当前状态在时间间隔内的部分
                if instance_id in self.instance_status_start_time:
                    status = self.instance_status.get(instance_id)
                    if status is not None:
                        status_start = self.instance_status_start_time[instance_id]
                        
                        # 计算当前状态与时间间隔的重叠部分
                        overlap_start = max(status_start, start_time)
                        overlap_end = min(current_time, end_time)
                        
                        if overlap_start < overlap_end:
                            # 只计算 ACTIVE、SCALING_UP、DRAINING 状态的时间
                            if status in [InstanceStatus.ACTIVE, 
                                         InstanceStatus.SCALING_UP, 
                                         InstanceStatus.DRAINING]:
                                instance_total_time += (overlap_end - overlap_start)
            else:
                # 实例不在 ScalingManager 的状态记录系统中（通常是启动时通过 pre_start=True 创建的）
                # 将其视为从创建时开始就处于 ACTIVE 状态
                for inst in self.application.instances:
                    if inst.instance_id == instance_id:
                        # 使用实例的 spin_up_timestamp 作为开始时间
                        if hasattr(inst, 'metrics') and hasattr(inst.metrics, 'spin_up_timestamp'):
                            instance_start_time = inst.metrics.spin_up_timestamp
                            if instance_start_time is not None and instance_start_time >= 0:
                                # 计算实例运行时间与时间间隔的重叠部分
                                overlap_start = max(instance_start_time, start_time)
                                overlap_end = min(current_time, end_time)
                                
                                if overlap_start < overlap_end:
                                    # 视为 ACTIVE 状态的时间
                                    instance_total_time += (overlap_end - overlap_start)
                        break
            
            total_time += instance_total_time
        
        # 更新上次计算时间（仅当没有指定 tag 时，避免影响其他计算）
        # 一个周期内连续调用两次
        if tag =="token":
            self._last_calculation_token_time = end_time
        elif tag=="prompt":
            self._last_calculation_prompt_time = end_time
        else:
            self._last_calculation_time = end_time


        return total_time
    
    def calculate_total_instance_time_since_last(self):
        """
        计算自上次调用以来的所有实例总使用时间
        这是一个便捷方法，用于定期调用
        
        Returns:
            float: 自上次调用以来的总使用时间（秒）
        """
        current_time = clock()
        
        if self._last_calculation_time is None:
            # 第一次调用，计算从 0 到当前时间
            return self.calculate_total_instance_time(start_time=2.0, end_time=current_time)
        else:
            # 计算从上一次调用到当前时间
            return self.calculate_total_instance_time(
                start_time=self._last_calculation_time, 
                end_time=current_time
            )
    
    def calculate_prompt_instance_time(self, start_time=None, end_time=None, interval=None):
        """
        计算指定时间间隔内所有 prompt 实例的总使用时间
        
        Args:
            start_time: 时间间隔开始时间（如果为 None，则使用上次调用时间或 0）
            end_time: 时间间隔结束时间（如果为 None，则使用当前时间）
            interval: 时间间隔（秒），如果提供，则 start_time = end_time - interval
            
        Returns:
            float: prompt 实例的总使用时间（秒）
        """
        return self.calculate_total_instance_time(
            start_time=start_time, 
            end_time=end_time, 
            interval=interval, 
            tag="prompt"
        )
    
    def calculate_token_instance_time(self, start_time=None, end_time=None, interval=None):
        """
        计算指定时间间隔内所有 token 实例的总使用时间
        
        Args:
            start_time: 时间间隔开始时间（如果为 None，则使用上次调用时间或 0）
            end_time: 时间间隔结束时间（如果为 None，则使用当前时间）
            interval: 时间间隔（秒），如果提供，则 start_time = end_time - interval
            
        Returns:
            float: token 实例的总使用时间（秒）
        """
        return self.calculate_total_instance_time(
            start_time=start_time, 
            end_time=end_time, 
            interval=interval, 
            tag="token"
        )
    
    def calculate_prompt_instance_time_since_last(self):
        """
        计算自上次调用以来的所有 prompt 实例总使用时间
        
        Returns:
            float: 自上次调用以来的 prompt 实例总使用时间（秒）
        """
        current_time = clock()
        
        if self._last_calculation_prompt_time is None:
            return self.calculate_prompt_instance_time(start_time=2.0, end_time=current_time)
        else:
            return self.calculate_prompt_instance_time(
                start_time=self._last_calculation_prompt_time,
                end_time=current_time
            )
    
    def calculate_token_instance_time_since_last(self):
        """
        计算自上次调用以来的所有 token 实例总使用时间
        
        Returns:
            float: 自上次调用以来的 token 实例总使用时间（秒）
        """
        current_time = clock()
        
        if self._last_calculation_token_time is None:
            return self.calculate_token_instance_time(start_time=2.0, end_time=current_time)
        else:
            return self.calculate_token_instance_time(
                start_time=self._last_calculation_token_time,
                end_time=current_time
            )

