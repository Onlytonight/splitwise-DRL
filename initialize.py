"""
Utility functions for initializing the simulation environment.
"""

import logging
import os

from hydra.utils import instantiate
from hydra.utils import get_original_cwd

from application import Application
from cluster import Cluster
from hardware_repo import HardwareRepo
from model_repo import ModelRepo
from orchestrator_repo import OrchestratorRepo
from start_state import load_start_state
from trace import Trace


def init_trace(cfg):
    trace_path = os.path.join(get_original_cwd(), cfg.trace.path)
    print("trace path is", trace_path)
    trace = Trace.from_csv(trace_path)
    return trace


def init_hardware_repo(cfg):
    processors_path = os.path.join(get_original_cwd(),
                                   cfg.hardware_repo.processors)
    interconnects_path = os.path.join(get_original_cwd(),
                                      cfg.hardware_repo.interconnects)
    skus_path = os.path.join(get_original_cwd(),
                             cfg.hardware_repo.skus)
    hardware_repo = HardwareRepo(processors_path,
                                 interconnects_path,
                                 skus_path)
    return hardware_repo


def init_model_repo(cfg):
    model_architectures_path = os.path.join(get_original_cwd(),
                                            cfg.model_repo.architectures)
    model_sizes_path = os.path.join(get_original_cwd(),
                                    cfg.model_repo.sizes)
    model_repo = ModelRepo(model_architectures_path, model_sizes_path)
    return model_repo


def init_orchestrator_repo(cfg):
    allocators_path = os.path.join(get_original_cwd(),
                                   cfg.orchestrator_repo.allocators)
    schedulers_path = os.path.join(get_original_cwd(),
                                   cfg.orchestrator_repo.schedulers)
    orchestrator_repo = OrchestratorRepo(allocators_path, schedulers_path)
    return orchestrator_repo


def init_performance_model(cfg):
    performance_model = instantiate(cfg.performance_model)
    return performance_model


def init_power_model(cfg):
    power_model = instantiate(cfg.power_model)
    return power_model


def init_cluster(cfg):
    cluster = Cluster.from_config(cfg.cluster)
    return cluster


def init_router(cfg, cluster):
    router = instantiate(cfg.router, cluster=cluster)
    return router


def init_arbiter(cfg, cluster):
    arbiter = instantiate(cfg.arbiter, cluster=cluster)
    return arbiter


def init_autoscaling_policy(cfg):
    """
    初始化自动扩缩容策略
    Args:
        cfg: 配置对象
    Returns:
        AutoscalingPolicy 实例或 None
    """
    if hasattr(cfg, 'autoscaling_policy') and cfg.autoscaling_policy is not None:
        # 使用 Hydra instantiate 创建策略对象
        policy = instantiate(cfg.autoscaling_policy)
        return policy
    return None

def init_applications(cfg, cluster, router, arbiter):
    applications = {}

    # 初始化自动扩缩容策略（全局共享）
    autoscaling_policy = init_autoscaling_policy(cfg)
    
    for application_cfg in cfg.applications:
        application = Application.from_config(application_cfg,
                                              cluster=cluster,
                                              router=router,
                                              arbiter=arbiter)
        
        # 初始化扩缩容管理器
        if hasattr(application_cfg, 'scaling_manager') and application_cfg.scaling_manager is not None:
            from scaling_manager import ScalingManager
            scaling_manager = ScalingManager(
                application=application,
                cluster=cluster,
                scale_up_delay=getattr(application_cfg.scaling_manager, 'scale_up_delay', 10.0),
                drain_check_interval=getattr(application_cfg.scaling_manager, 'drain_check_interval', 1.0),
                autoscaling_policy=autoscaling_policy,  # 传入自动扩缩容策略
                decision_interval=getattr(application_cfg.scaling_manager, 'decision_interval', 30.0),
                enable_autoscaling=getattr(application_cfg.scaling_manager, 'enable_autoscaling', False),
                debug=getattr(application_cfg, 'debug', False)
            )
            application.scaling_manager = scaling_manager
        
        applications[application_cfg.application_id] = application
    return applications


def init_start_state(cfg, **kwargs):
    load_start_state(cfg.start_state, **kwargs)


def start_autoscaling(applications):
    """
    启动所有应用的自动扩缩容循环
    
    应该在模拟器初始化完成后调用（在 init_start_state 之后）
    
    Args:
        applications: 应用字典
    """
    for app_id, application in applications.items():
        if hasattr(application, 'scaling_manager') and application.scaling_manager is not None:
            application.scaling_manager.start_autoscaling()


if __name__ == "__main__":
    pass
