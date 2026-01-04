import logging
import os
import random
import sys
import time
import hydra

from hydra.utils import instantiate
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf

from simulator import TraceSimulator, TraceRLSimulator,TraceSACSimulator
from initialize import *


# register custom hydra resolver
OmegaConf.register_new_resolver("eval", eval)


def init_trace_from_path(trace_path):
    """
    从给定的 trace 路径初始化 Trace 对象。
    
    Args:
        trace_path: trace 文件的路径（相对于原始工作目录）
    
    Returns:
        Trace 对象
    """
    from hydra.utils import get_original_cwd
    from trace import Trace
    import os
    
    full_path = os.path.join(get_original_cwd(), trace_path)
    # print(f"Loading trace from: {full_path}")
    trace = Trace.from_csv(full_path)
    return trace


def run_simulation(cfg):
    hardware_repo = init_hardware_repo(cfg)
    model_repo = init_model_repo(cfg)
    orchestrator_repo = init_orchestrator_repo(cfg)
    performance_model = init_performance_model(cfg)
    power_model = init_power_model(cfg)
    cluster = init_cluster(cfg)
    router = init_router(cfg, cluster)
    arbiter = init_arbiter(cfg, cluster)
    applications = init_applications(cfg, cluster, router, arbiter)
    
    for application in applications.values():
        router.add_application(application)
        arbiter.add_application(application)
    
    # 获取 trace_epochs 配置（如果设置了，就循环使用同一个 trace）
    trace_epochs = getattr(cfg, 'trace_epochs', 1)
    # 获取 trace_list_repeats 配置（trace_list 的循环次数）
    trace_list_repeats = getattr(cfg, 'trace_list_repeats', 1)
    
    # 检查是否有 trace_list 配置，如果有则循环训练多个 trace
    if hasattr(cfg, 'trace_list') and cfg.trace_list is not None and len(cfg.trace_list) > 0:
        # 使用 trace_list 进行循环训练
        # 将 ListConfig 转换为普通列表
        trace_paths = list(cfg.trace_list)
        
        # 如果设置了 trace_list_repeats > 1，将整个 trace_list 重复指定次数
        if trace_list_repeats > 1:
            original_trace_paths = trace_paths.copy()
            trace_paths = original_trace_paths * trace_list_repeats
            logging.info(f"Repeating trace_list {trace_list_repeats} times: {len(original_trace_paths)} traces × {trace_list_repeats} = {len(trace_paths)} total traces")
        # 如果只有一个 trace 且设置了 trace_epochs > 1，则循环使用它
        elif len(trace_paths) == 1 and trace_epochs > 1:
            trace_paths = trace_paths * trace_epochs
            logging.info(f"Using single trace with {trace_epochs} epochs: {trace_paths[0]}")
        else:
            logging.info(f"Starting multi-trace training with {len(trace_paths)} traces")
        
        # 初始化第一个 trace
        first_trace = init_trace_from_path(trace_paths[0])
        print(f"First trace: {trace_paths[0]}")
        
        # 创建模拟器（只创建一次，后续重用）
        sim = TraceSACSimulator(trace=first_trace,
                             cluster=cluster,
                             applications=applications,
                             router=router,
                             arbiter=arbiter,
                             end_time=cfg.end_time)
        
        # 初始化起始状态
        init_start_state(cfg,
                         cluster=cluster,
                         applications=applications,
                         router=router,
                         arbiter=arbiter)
        
        # 运行第一个 trace
        logging.info(f"Running trace {1}/{len(trace_paths)}: {trace_paths[0]}")
        sim.run()
        
        # 循环处理剩余的 trace
        for epoch, trace_path in enumerate(trace_paths[1:], start=2):
            logging.info(f"Running trace {epoch}/{len(trace_paths)}: {trace_path}")
            
            # 在开始新 trace 之前，确保上一个 trace 的所有文件流都已关闭（双重保险）
            # reset_for_new_trace 中也会关闭，但这里确保即使 run() 异常退出也能关闭
            if hasattr(sim, 'reward_recorder') and sim.reward_recorder is not None:
                sim.reward_recorder.close()
            if hasattr(sim, 'prompt_reward_recorder') and sim.prompt_reward_recorder is not None:
                sim.prompt_reward_recorder.close()
            if hasattr(sim, 'token_reward_recorder') and sim.token_reward_recorder is not None:
                sim.token_reward_recorder.close()
            
            # 加载新的 trace（如果路径相同，可以重用同一个 trace 对象）
            new_trace = init_trace_from_path(trace_path)
            
            # 重新初始化所有组件（彻底清理状态）
            new_cluster = init_cluster(cfg)
            new_router = init_router(cfg, new_cluster)
            new_arbiter = init_arbiter(cfg, new_cluster)
            new_applications = init_applications(cfg, new_cluster, new_router, new_arbiter)
            
            for application in new_applications.values():
                new_router.add_application(application)
                new_arbiter.add_application(application)
            
            # 重置模拟器状态（保持 PPO agents），并传入新初始化的组件
            sim.reset_for_new_trace(new_trace, 
                                   new_cluster=new_cluster,
                                   new_applications=new_applications,
                                   new_router=new_router,
                                   new_arbiter=new_arbiter)
            
            # 重新初始化起始状态
            init_start_state(cfg,
                             cluster=new_cluster,
                             applications=new_applications,
                             router=new_router,
                             arbiter=new_arbiter)
            
            # 运行新的 trace
            sim.run()
        
        logging.info(f"Completed multi-trace training with {len(trace_paths)} traces")
    else:
        # 使用单个 trace（原有逻辑）
        trace = init_trace(cfg)
        print("trace is", trace)
        
        # 创建模拟器
        sim = TraceSACSimulator(trace=trace,
                             cluster=cluster,
                             applications=applications,
                             router=router,
                             arbiter=arbiter,
                             end_time=cfg.end_time)
        
        # 如果设置了 trace_epochs > 1，循环使用同一个 trace
        for epoch in range(1, trace_epochs + 1):
            logging.info(f"Running trace epoch {epoch}/{trace_epochs}")
            
            # 在开始新 epoch 之前，确保上一个 epoch 的所有文件流都已关闭（双重保险）
            if epoch > 1:
                if hasattr(sim, 'reward_recorder') and sim.reward_recorder is not None:
                    sim.reward_recorder.close()
                if hasattr(sim, 'prompt_reward_recorder') and sim.prompt_reward_recorder is not None:
                    sim.prompt_reward_recorder.close()
                if hasattr(sim, 'token_reward_recorder') and sim.token_reward_recorder is not None:
                    sim.token_reward_recorder.close()
            
            if epoch > 1:
                # 重新初始化所有组件（彻底清理状态）
                new_cluster = init_cluster(cfg)
                new_router = init_router(cfg, new_cluster)
                new_arbiter = init_arbiter(cfg, new_cluster)
                new_applications = init_applications(cfg, new_cluster, new_router, new_arbiter)
                
                for application in new_applications.values():
                    new_router.add_application(application)
                    new_arbiter.add_application(application)
                
                # 重置模拟器状态（保持 PPO agents），并传入新初始化的组件
                sim.reset_for_new_trace(trace,
                                       new_cluster=new_cluster,
                                       new_applications=new_applications,
                                       new_router=new_router,
                                       new_arbiter=new_arbiter)
                
                # 重新初始化起始状态
                init_start_state(cfg,
                                 cluster=new_cluster,
                                 applications=new_applications,
                                 router=new_router,
                                 arbiter=new_arbiter)
            else:
                # 第一次运行，初始化起始状态
                init_start_state(cfg,
                                 cluster=cluster,
                                 applications=applications,
                                 router=router,
                                 arbiter=arbiter)
            
            # 运行 trace
            sim.run()
        
        if trace_epochs > 1:
            logging.info(f"Completed training with {trace_epochs} epochs")


@hydra.main(config_path="configs", config_name="config", version_base=None)
def run(cfg: DictConfig) -> None:
    # print config
    #print(OmegaConf.to_yaml(cfg, resolve=False))
    #hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
    #print(OmegaConf.to_yaml(hydra_cfg, resolve=False))

    # initialize random number generator
    start_time = time.time()
    random.seed(cfg.seed)

    # delete existing oom.csv if any
    if os.path.exists("oom.csv"):
        os.remove("oom.csv")

    run_simulation(cfg)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")

if __name__ == "__main__":
    run()
