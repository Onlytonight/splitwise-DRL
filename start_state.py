"""
Utility functions to initialize the Cluster with a starting state.
"""

import logging

from model import ModelParallelism
from simulator import clock, schedule_event, cancel_event, reschedule_event


class StartStateManager:
    """
    管理启动状态配置，提供获取实例配置和并行度的方法
    """
    def __init__(self, start_state_cfg):
        self.start_state_cfg = start_state_cfg
        self.state_type = start_state_cfg.state_type
        
        # 保存 prompt 和 token 配置
        if "splitwise" in self.state_type:
            self.prompt_cfg = start_state_cfg.prompt if hasattr(start_state_cfg, 'prompt') else None
            self.token_cfg = start_state_cfg.token if hasattr(start_state_cfg, 'token') else None
            
            # 保存并行度配置
            if self.prompt_cfg:
                self.prompt_parallelism = ModelParallelism(
                    pipeline_parallelism=self.prompt_cfg.pipeline_parallelism,
                    tensor_parallelism=self.prompt_cfg.tensor_parallelism
                )
            else:
                self.prompt_parallelism = None
            
            if self.token_cfg:
                self.token_parallelism = ModelParallelism(
                    pipeline_parallelism=self.token_cfg.pipeline_parallelism,
                    tensor_parallelism=self.token_cfg.tensor_parallelism
                )
            else:
                self.token_parallelism = None
        elif self.state_type in ["orca", "baseline"]:
            self.instance_cfg = start_state_cfg.instance if hasattr(start_state_cfg, 'instance') else None
            if self.instance_cfg:
                self.parallelism = ModelParallelism(
                    pipeline_parallelism=self.instance_cfg.pipeline_parallelism,
                    tensor_parallelism=self.instance_cfg.tensor_parallelism
                )
            else:
                self.parallelism = None
        else:
            self.prompt_cfg = None
            self.token_cfg = None
            self.instance_cfg = None
            self.prompt_parallelism = None
            self.token_parallelism = None
            self.parallelism = None
    
    def get_instance_config(self, tag=None):
        """
        获取实例配置
        
        Args:
            tag: 实例标签（"prompt" 或 "token"），如果为 None 则返回默认配置
            
        Returns:
            instance_cfg: 实例配置对象
        """
        if "splitwise" in self.state_type:
            if tag == "prompt":
                return self.prompt_cfg
            elif tag == "token":
                return self.token_cfg
            else:
                # 默认返回 prompt 配置
                return self.prompt_cfg if self.prompt_cfg else self.token_cfg
        else:
            return self.instance_cfg
    
    def get_parallelism(self, tag=None):
        """
        获取并行度配置
        
        Args:
            tag: 实例标签（"prompt" 或 "token"），如果为 None 则返回默认并行度
            
        Returns:
            ModelParallelism: 并行度对象
        """
        if "splitwise" in self.state_type:
            if tag == "prompt":
                return self.prompt_parallelism
            elif tag == "token":
                return self.token_parallelism
            else:
                # 默认返回 prompt 并行度
                return self.prompt_parallelism if self.prompt_parallelism else self.token_parallelism
        else:
            return self.parallelism


def load_start_state(start_state_cfg, **kwargs):
    """
    Load the start state configuration and initialize the cluster.
    """
    # 创建 StartStateManager 并保存到应用中
    if 'applications' in kwargs:
        applications = kwargs['applications']
        start_state_manager = StartStateManager(start_state_cfg)
        
        # 将配置管理器保存到每个应用中
        for app in applications.values():
            app.start_state_manager = start_state_manager
    
    state_type = start_state_cfg.state_type
    if state_type == "unallocated":
        pass
    elif state_type == "orca":
        uniform(start_state_cfg, **kwargs)
    elif state_type == "baseline":
        uniform(start_state_cfg, **kwargs)
    elif "splitwise" in state_type:
        splitwise(start_state_cfg, **kwargs)
    else:
        raise ValueError(f"Unknown start state type: {state_type}")


def uniform(start_state_cfg, cluster, applications, **kwargs):
    """
    Initialize all servers with a single instance of the application.
    """
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers
    print("servers are", servers)
    
    instance_cfg = start_state_cfg.instance
    print("instance cfg is", instance_cfg.num_instances)
    parallelism = ModelParallelism(pipeline_parallelism=instance_cfg.pipeline_parallelism,
                                   tensor_parallelism=instance_cfg.tensor_parallelism)

    for sku_name in servers:
        for server in servers[sku_name]:
            allocator.start_spin_up_instance(instance_cfg=instance_cfg,
                                             processors=server.processors,
                                             parallelism=parallelism,
                                             pre_start=True)


def splitwise(start_state_cfg, cluster, applications, **kwargs):
    """
    Initialize all servers with a single instance of the application.
    Separate prompt and token instances with different kinds of parallelism.
    TODO: use preferences and constraints within scheduler instead
    """
    application = applications[start_state_cfg.application_id]
    allocator = application.allocator
    servers = cluster.servers

    prompt_cfg = start_state_cfg.prompt
    token_cfg = start_state_cfg.token
    # print("prompt instance is",prompt_cfg.num_instances,"token instance is", token_cfg.num_instances)
    prompt_parallelism = ModelParallelism(pipeline_parallelism=prompt_cfg.pipeline_parallelism,
                                          tensor_parallelism=prompt_cfg.tensor_parallelism)
    token_parallelism = ModelParallelism(pipeline_parallelism=token_cfg.pipeline_parallelism,
                                         tensor_parallelism=token_cfg.tensor_parallelism)

    split_type = start_state_cfg.split_type

    if split_type == "homogeneous":
        n_prompts = prompt_cfg.num_instances
        n_tokens = token_cfg.num_instances
        # allocate n_prompt instance of prompt
        all_servers = [server for sku_name in servers for server in servers[sku_name]]
        for server in all_servers[:n_prompts]:
            for proc_id in range(0, len(server.processors), prompt_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                 processors=server.processors[proc_id:proc_id+prompt_parallelism.tensor_parallelism],
                                                 parallelism=prompt_parallelism,
                                                 pre_start=True,
                                                 tag="prompt")
        for server in all_servers[n_prompts:n_prompts+n_tokens]:
            for proc_id in range(0, len(server.processors), token_parallelism.tensor_parallelism):
                allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                 processors=server.processors[proc_id:proc_id+token_parallelism.tensor_parallelism],
                                                 parallelism=token_parallelism,
                                                 pre_start=True,
                                                 tag="token")

    if split_type == "heterogeneous":
        prompt_instances = prompt_cfg.instance_names
        token_instances = token_cfg.instance_names
        for sku_name in servers:
            for server in servers[sku_name]:
                if sku_name in prompt_instances:
                    # allocate as many prompt instances as possible
                    for proc_id in range(0, len(server.processors), prompt_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=prompt_cfg,
                                                         processors=server.processors[proc_id:proc_id+prompt_parallelism.tensor_parallelism],
                                                         parallelism=prompt_parallelism,
                                                         pre_start=True,
                                                         tag="prompt")
                elif sku_name in token_instances:
                    # allocate as many token instances as possible
                    for proc_id in range(0, len(server.processors), token_parallelism.tensor_parallelism):
                        allocator.start_spin_up_instance(instance_cfg=token_cfg,
                                                         processors=server.processors[proc_id:proc_id+token_parallelism.tensor_parallelism],
                                                         parallelism=token_parallelism,
                                                         pre_start=True,
                                                         tag="token")
                else:
                    raise ValueError(f"Unsupported sku_name: {sku_name}")
