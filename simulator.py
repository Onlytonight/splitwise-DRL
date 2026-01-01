import heapq
import logging
import os
import random
from collections import defaultdict

import utils
from RL.state import RLStateCollector
from RL.reward import RLRewardCalculator, RewardRecorder
from RL.action import RLActionExecutor
from RL.PPO import PPO
# global simulator that drives the simulation
# bad practice, but it works for now
sim = None


class Event:
    """
    Events are scheduled actions in the simulator.
    """
    # 定义事件状态常量
    PENDING = "pending"      # 等待执行
    EXECUTING = "executing"  # 正在执行
    COMPLETED = "completed"  # 已完成
    CANCELLED = "cancelled"  # 已取消

    def __init__(self, time, action):
        self.time = time
        self.action = action
        self.status = self.PENDING  # 初始化状态为等待执行

    def __str__(self):
        return f"Event with time {self.time}, action {self.action}, status {self.status}"

    def __lt__(self, other):
        return self.time < other.time

    def mark_executing(self):
        """标记事件为正在执行"""
        self.status = self.EXECUTING

    def mark_completed(self):
        """标记事件为已完成"""
        self.status = self.COMPLETED

    def mark_cancelled(self):
        """标记事件为已取消"""
        self.status = self.CANCELLED



class Simulator:
    """
    A discrete event simulator that schedules and runs Events.
    """
    def __init__(self, end_time):
        global sim
        sim = self
        self.time = 0
        self.end_time = end_time
        self.events = []           # 待执行事件队列
        self.deleted_events = []   # 已删除事件列表
        self.completed_events = []  # 已完成事件列表
        logging.info("Simulator initialized")

        # logger for simulator events
        self.logger = utils.file_logger("simulator")
        self.logger.info("time,event,status")

    def schedule(self, delay, action):
        """
        Schedule an event by specifying delay and an action function.
        """
        # run immediately if delay is 0
        if delay == 0:
            action()
            return None
        event = Event(self.time + delay, action)
        heapq.heappush(self.events, event)
        return event

    def cancel(self, event):
        """
        Cancel an event.
        """
        if event:
            event.mark_cancelled()
            self.deleted_events.append(event)

    def reschedule(self, event, delay):
        """
        Reschedule an event by cancelling and scheduling it again.
        """
        self.cancel(event)
        return self.schedule(delay, event.action)

    def run(self):
        """
        Run the simulation until the end time.
        """
        while self.events and self.time < self.end_time:
            event = heapq.heappop(self.events)
            if event in self.deleted_events:
                self.deleted_events.remove(event)
                continue
            self.time = event.time
            event.mark_executing()
            event.action()
            event.mark_completed()
            self.completed_events.append(event)
            # self.logger.debug(f"{event.time},{event.action},{event.status}")



class TraceSimulator(Simulator):
    """
    A discrete event simulator that processes Request arrivals from a Trace.
    """
    def __init__(self,
                 trace,
                 cluster,
                 applications,
                 router,
                 arbiter,
                 end_time):
        super().__init__(end_time)
        self.trace = trace
        self.cluster = cluster
        self.applications = applications
        self.router = router
        self.arbiter = arbiter
        logging.info("TraceSimulator initialized")
        self.load_trace()

    def load_trace(self):
        """
        Load requests from the trace as arrival events.
        """
        for request in self.trace.requests:
            self.schedule(request.arrival_timestamp,
                          lambda request=request: self.router.request_arrival(request))

    def run(self):
        # start simulation by scheduling a cluster run
        self.schedule(0, self.cluster.run)
        self.schedule(0, self.router.run)
        self.schedule(0, self.arbiter.run)

        # run simulation
        super().run()
        self.logger.info(f"{self.time},end")
        logging.info(f"TraceSimulator completed at {self.time}")

        self.save_results()

    def save_results(self, detailed=True):
        """
        Save results at the end of the simulation.
        """
        self.router.save_results()

        sched_results = {}
        alloc_results = {}
        for application_id, application in self.applications.items():
            allocator_results, scheduler_results = application.get_results()
            alloc_results[application_id] = allocator_results
            sched_results[application_id] = scheduler_results

        # summary sched results
        summary_results = defaultdict(list)
        for application_id, results_dict in sched_results.items():
            summary_results["application_id"].append(application_id)
            for key, values in results_dict.items():
                summary = utils.get_statistics(values)
                # merge summary into summary_results
                for metric, value in summary.items():
                    summary_results[f"{key}_{metric}"].append(value)

        # save summary results
        utils.save_dict_as_csv(summary_results, "summary.csv")

        if detailed:
            # create a dataframe of all requests, save as csv
            for application_id, result in sched_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}.csv")
            for application_id, result in alloc_results.items():
                utils.save_dict_as_csv(result, f"detailed/{application_id}_alloc.csv")


# Convenience functions for simulator object

def clock():
    """
    Return the current time of the simulator.
    """
    return sim.time

def schedule_event(*args):
    """
    Schedule an event in the simulator at desired delay.
    """
    return sim.schedule(*args)

def cancel_event(*args):
    """
    Cancel existing event in the simulator.
    """
    return sim.cancel(*args)

def reschedule_event(*args):
    """
    Reschedule existing event in the simulator.
    Equivalent to cancelling and scheduling a new event.
    """
    return sim.reschedule(*args)


class TraceRLSimulator(Simulator):
    # 类级别的标志，用于跟踪是否是第一次保存（整个程序运行期间）
    _first_save = True
    
    def __init__(self,
                 trace,
                 cluster,
                 applications,
                 router,
                 arbiter,
                 end_time):  # <--- 新增参数：决策间隔（例如10秒）
        super().__init__(end_time)
        self.trace = trace
        self.cluster = cluster
        self.applications = applications
        self.router = router
        self.arbiter = arbiter
        self.decision_interval = 2  # 保存间隔
        # self.enabled_features =["rate", "length", "queue", "instance_count", "utilization", "scaling","slo"]
        self.enabled_features=["queue","none_count", "instance_count"]
        self.rl_config = {
            "w_cost": 0.5,
            "w_slo": 10,
            "w_switch": 0.1,
            "w_util": 0.2,
            "action_scale_step": 5,
            "action_mig_step": 3,
            "min_instances_per_pool": 1,
            "max_total_instances": 100,
            "stack_size": 4  # 状态堆叠的时间窗大小
        }
        rl_config = self.rl_config
        
        # 从配置中获取 stack_size，默认为 4
        self.stack_size = self.rl_config.get("stack_size", 4)

        # 用于保存上一次决策时的统计快照，用于计算区间内的速率（Rate）
        # prompt / token 两个 RL 代理分别使用各自的状态收集器
        self.prompt_collector = RLStateCollector(
            cluster=cluster,
            router=router,
            applications=applications,
            stack_size=self.stack_size,
            mode="prompt",
            enabled_features=self.enabled_features
        )
        self.token_collector = RLStateCollector(
            cluster=cluster,
            router=router,
            applications=applications,
            stack_size=self.stack_size,
            mode="token",
            enabled_features=self.enabled_features
        )

        # 初始化两个奖励计算器
        self.prompt_reward_calculator = RLRewardCalculator(
            config=rl_config,
            max_instances=rl_config.get("max_total_instances", 100),
            mode="prompt",
        )
        self.token_reward_calculator = RLRewardCalculator(
            config=rl_config,
            max_instances=rl_config.get("max_total_instances", 100),
            mode="token",
        )

        # 获取第一个应用（假设只有一个应用）
        self.application = list(applications.values())[0]

        self.action_executor = RLActionExecutor(
            application=self.application,
            config=rl_config,  # 从配置中读取步长等参数
        )

        # --- PPO Hyperparameters (从你的代码中提取并简化) ---
        self.has_continuous_action_space = True
        self.action_std = 0.4  # 初始动作方差
        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_std_decay_freq = 1000

        self.update_timestep = 10  # 每多少个决策步更新一次网络
        self.K_epochs = 40
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lr_actor = 0.001
        self.lr_critic = 0.001
        self.hidden_dim = 128  # 神经网络隐藏层维度

        # --- 初始化两个 PPO Agent ---
        prompt_state_dim = self.prompt_collector.feature_dim * self.prompt_collector.stack_size
        token_state_dim = self.token_collector.feature_dim * self.token_collector.stack_size
        
        # 添加调试信息
        logging.info(f"Prompt feature_dim: {self.prompt_collector.feature_dim}, stack_size: {self.prompt_collector.stack_size}, state_dim: {prompt_state_dim}")
        logging.info(f"Token feature_dim: {self.token_collector.feature_dim}, stack_size: {self.token_collector.stack_size}, state_dim: {token_state_dim}")

        self.prompt_agent = PPO(
            prompt_state_dim,
            1,  # 单一动作：只负责 prompt 池
            self.lr_actor,
            self.lr_critic,
            self.gamma,
            self.K_epochs,
            self.eps_clip,
            self.has_continuous_action_space,
            self.action_std,
            self.hidden_dim,
        )
        self.token_agent = PPO(
            token_state_dim,
            1,  # 单一动作：只负责 token 池
            self.lr_actor,
            self.lr_critic,
            self.gamma,
            self.K_epochs,
            self.eps_clip,
            self.has_continuous_action_space,
            self.action_std,
            self.hidden_dim,
        )

        # --- 训练状态追踪 ---
        self.decision_step = 0  # 相当于 time_step
        self.last_prompt_state = None
        self.last_token_state = None
        self.last_prompt_action_executed = True
        self.last_token_action_executed = True
        self.last_prompt_action = 0.0  # 上一次 prompt 动作值
        self.last_token_action = 0.0  # 上一次 token 动作值
        self.save_model_freq = 1000  # 保存模型频率
        self.finish_training = False

        logging.info("TraceRLSimulator initialized with dual RL agents")
        self.load_trace()

    def load_trace(self):
        """
        Load requests from the trace as arrival events.
        """
        for request in self.trace.requests:
            self.schedule(request.arrival_timestamp,
                          lambda request=request: self.router.request_arrival(request))

    def reset_for_new_trace(self, new_trace, new_cluster=None, new_applications=None, 
                            new_router=None, new_arbiter=None):
        """
        重置模拟器状态以加载新的 trace，但保持 PPO agents 和训练状态。
        如果提供了新的组件，则重新初始化所有组件（彻底清理状态）。
        
        Args:
            new_trace: 新的 Trace 对象
            new_cluster: 新初始化的 Cluster 对象（可选）
            new_applications: 新初始化的 Applications 字典（可选）
            new_router: 新初始化的 Router 对象（可选）
            new_arbiter: 新初始化的 Arbiter 对象（可选）
        """
        # 如果提供了新的组件，则重新初始化所有组件
        if new_cluster is not None and new_applications is not None and \
           new_router is not None and new_arbiter is not None:
            # print('reset')
            # 更新组件引用
            self.cluster = new_cluster
            self.applications = new_applications
            self.router = new_router
            self.arbiter = new_arbiter
            
            # 重新创建状态收集器（因为它们持有对 cluster/router/applications 的引用）
            # 第一个 collector 创建时重置共享状态，第二个保持共享
            from RL.state import RLStateCollector
            self.prompt_collector = RLStateCollector(
                cluster=new_cluster,
                router=new_router,
                applications=new_applications,
                stack_size=self.stack_size,
                mode="prompt",
                reset_shared_stats=True,  # 重置共享统计状态
                enabled_features=self.enabled_features
            )
            self.token_collector = RLStateCollector(
                cluster=new_cluster,
                router=new_router,
                applications=new_applications,
                stack_size=self.stack_size,
                mode="token",
                reset_shared_stats=False,  # 不重复重置（已经在 prompt_collector 中重置）
                enabled_features=self.enabled_features
            )
            
            # 重新创建 action executor（因为它持有对 application 的引用）
            from RL.action import RLActionExecutor
            self.application = list(new_applications.values())[0]
            self.action_executor = RLActionExecutor(
                application=self.application,
                config=self.rl_config,
            )
            
            # logging.info("All components reinitialized (cluster, applications, router, arbiter)")
        
        # 重置模拟器基础状态
        self.time = 0
        self.events = []
        self.deleted_events = []
        self.completed_events = []
        
        # 更新 trace
        self.trace = new_trace
        
        # 重新加载新 trace 的事件
        self.load_trace()
        
        # 重置状态收集器的快照缓存和共享统计状态
        from RL.state import RLStateCollector
        RLStateCollector.clear_snapshot_cache()

        # 重置 scheduler 的状态
        # 如果提供了新的 applications，使用新的 applications；否则使用现有的 applications
        applications_to_reset = new_applications if new_applications is not None else self.applications
        for application in applications_to_reset.values():
            if hasattr(application, 'scheduler') and application.scheduler is not None:
                application.scheduler.reset()

        # 重置上一次的状态（新 trace 开始时没有上一状态）
        self.last_prompt_state = None
        self.last_token_state = None
        self.last_prompt_action_executed = True
        self.last_token_action_executed = True
        self.last_prompt_action = 0.0
        self.last_token_action = 0.0
        self.finish_training = False
        
        # logging.info(f"Simulator reset for new trace with {len(new_trace.requests)} requests")

    def run(self):
        # start simulation by scheduling a cluster run
        self.schedule(0, self.cluster.run)
        self.schedule(0, self.router.run)
        self.schedule(0, self.arbiter.run)

        # [新增] 启动决策心跳循环
        # 如果设置了间隔，且大于0，则调度第一次决策
        if self.decision_interval > 0:
            logging.info(f"Starting decision cycle with interval {self.decision_interval}")
            self.schedule(self.decision_interval, self.run_decision_cycle)

        # run simulation
        super().run()
        self.logger.info(f"{self.time},end")
        logging.info(f"TraceSimulator completed at {self.time}")

        self.save_results()

    # [核心] 定义决策周期函数
    def run_decision_cycle(self):
        """
        这是 RL Agent 的核心介入点。
        """
        current_time = self.time

        # ---------------------------------------------------------
        # 1. 状态收集 (State Collection)
        # ---------------------------------------------------------
        # 获取上一次的动作值（如果存在）
        last_prompt_action = getattr(self, 'last_prompt_action', 0.0)
        last_token_action = getattr(self, 'last_token_action', 0.0)
        
        prompt_state, p_raw_stats, instance_num, rps = self.prompt_collector.get_state_and_stats(
            self.time, self.decision_interval, 
            last_prompt_action=last_prompt_action, 
            last_token_action=last_token_action
        )
        token_state, t_raw_stats, _, _ = self.token_collector.get_state_and_stats(
            self.time, self.decision_interval, 
            last_prompt_action=last_prompt_action, 
            last_token_action=last_token_action
        )
        logging.debug(f"RL Decision Triggered at time {current_time}")

        # ---------------------------------------------------------
        # 2. 基于上一时刻的状态计算两个 Agent 的奖励
        # ---------------------------------------------------------
        if self.last_prompt_state is not None and self.last_token_state is not None:
            # 从 raw_stats 中提取 avg_queue_time 和 avg_nth_token_overhead
            # reward_stats 格式: [prompt_rate, token_rate, sch_p_queue_tokens, sch_d_queue_tokens, 
            #                     n_p, n_t, avg_prompt_size, ttft_rate, tbt_rate, ins_p_queue, 
            #                     ins_d_queue, avg_queue_time, avg_nth_token_overhead]
            
            prompt_reward, prompt_info = self.prompt_reward_calculator.calculate_reward(
                self.cluster,
                self.applications,
                p_raw_stats,
                instance_num,
                action_executed=self.last_prompt_action_executed,
                step=self.decision_step
            )
            token_reward, token_info = self.token_reward_calculator.calculate_reward(
                self.cluster,
                self.applications,
                t_raw_stats,
                instance_num,
                action_executed=self.last_token_action_executed,
                step=self.decision_step
            )

            # 写入各自的 buffer
            self.prompt_agent.buffer.rewards.append(prompt_reward)
            self.prompt_agent.buffer.is_terminals.append(False)
            self.token_agent.buffer.rewards.append(token_reward)
            self.token_agent.buffer.is_terminals.append(False)

            # 记录奖励到 CSV（按 agent 区分）
            if not hasattr(self, 'prompt_reward_recorder'):
                self.prompt_reward_recorder = RewardRecorder("reward_prompt.csv")
            if not hasattr(self, 'token_reward_recorder'):
                self.token_reward_recorder = RewardRecorder("reward_token.csv")

            self.prompt_reward_recorder.record_reward(self.decision_step, prompt_info)
            self.token_reward_recorder.record_reward(self.decision_step, token_info)

            if self.decision_step % 1 == 0:
                logging.info(
                    f"Step: {self.decision_step} | "
                    f"lastAction:{self.last_prompt_action},PromptReward: {prompt_reward:.2f} (machine={prompt_info['cost_score']:.2f},{prompt_info['use_time']},sch_queue={prompt_info['p_queue_len']},ttft_p99={prompt_info['ttft_p99']:.2f},instance_queue={prompt_info['instance_p_queue_len']}) | "
                    f"lastAction:{self.last_token_action},TokenReward: {token_reward:.2f} (machine={token_info['cost_score']:.2f},{token_info['use_time']},sch_queue={token_info['t_queue_len']},tbt_p99={token_info['tbt_p99']:.2f},instance_queue={token_info['instance_t_queue_len']},avg_queue_time={token_info['avg_queue_time']:.3f},avg_nth_token_overhead={token_info['avg_nth_token_overhead']:.3f})"
                )
        if self.finish_training:
            return
        # ---------------------------------------------------------
        # 3. 周期性更新两个 PPO 策略
        # ---------------------------------------------------------
        if self.decision_step % self.update_timestep == 0 and self.decision_step > 0:
            logging.info(f"Updating PPO Policies at step {self.decision_step}...")
            self.prompt_agent.update()
            self.token_agent.update()

        if self.has_continuous_action_space and self.decision_step % self.action_std_decay_freq == 0:
            self.prompt_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)
            self.token_agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)
            logging.info(
                f"Decayed action std to "
                f"prompt={self.prompt_agent.action_std}, token={self.token_agent.action_std}"
            )

        if self.decision_step % self.save_model_freq == 0 and self.decision_step > 0:
            cp_dir = "cp"
            os.makedirs(cp_dir, exist_ok=True)
            import datetime

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_path = f"{cp_dir}/PPO_prompt_{self.decision_step}_{timestamp}.pth"
            token_path = f"{cp_dir}/PPO_token_{self.decision_step}_{timestamp}.pth"
            self.prompt_agent.save(prompt_path)
            self.token_agent.save(token_path)
            logging.info(f"Prompt model saved to {prompt_path}")
            logging.info(f"Token model saved to {token_path}")

        # ---------------------------------------------------------
        # 4. PPO 推理并执行各自动作
        # ---------------------------------------------------------
        prompt_action_arr = self.prompt_agent.select_action(prompt_state)
        token_action_arr = self.token_agent.select_action(token_state)

        prompt_alpha = float(prompt_action_arr[0]) if hasattr(prompt_action_arr, "__len__") else float(prompt_action_arr)
        token_alpha = float(token_action_arr[0]) if hasattr(token_action_arr, "__len__") else float(token_action_arr)

        prompt_executed = self.action_executor.execute_single(prompt_alpha, "prompt")
        token_executed = self.action_executor.execute_single(token_alpha, "token")

        self.last_prompt_action_executed = prompt_executed
        self.last_token_action_executed = token_executed
        self.last_prompt_state = prompt_state
        self.last_token_state = token_state
        # 保存上一次的动作值，用于下一次状态收集
        self.last_prompt_action = prompt_alpha
        self.last_token_action = token_alpha

        self.decision_step += 1

        # ---------------------------------------------------------
        # 5. 清除快照缓存，确保下一个决策周期会重新收集
        # ---------------------------------------------------------
        from RL.state import RLStateCollector
        RLStateCollector.clear_snapshot_cache()

        # ---------------------------------------------------------
        # 6. [关键] 递归调度下一次决策
        # ---------------------------------------------------------
        # 只要还没到结束时间，就安排下一次。rps判断是否模拟结束
        if current_time + self.decision_interval < self.end_time:
            # 让奖励收集回去
            if rps == 0:
                self.finish_training = True
            self.schedule(self.decision_interval, self.run_decision_cycle)

    def save_results(self, detailed=True):
        """
        Save results at the end of the simulation.
        第一次运行时清空文件，后续循环 trace 时追加。
        """
        import os
        from hydra.utils import get_original_cwd
        
        # 获取当前工作目录（Hydra 切换后的目录）
        current_dir = os.getcwd()
        logging.info(f"Saving results to directory: {current_dir}")
        
        self.router.save_results()

        sched_results = {}
        alloc_results = {}
        for application_id, application in self.applications.items():
            allocator_results, scheduler_results = application.get_results()
            alloc_results[application_id] = allocator_results
            sched_results[application_id] = scheduler_results

        # summary sched results
        summary_results = defaultdict(list)
        for application_id, results_dict in sched_results.items():
            summary_results["application_id"].append(application_id)
            for key, values in results_dict.items():
                summary = utils.get_statistics(values)
                # merge summary into summary_results
                for metric, value in summary.items():
                    summary_results[f"{key}_{metric}"].append(value)

        # 判断是否是第一次保存：第一次清空文件，后续追加
        is_first_save = TraceRLSimulator._first_save
        if is_first_save:
            TraceRLSimulator._first_save = False

        # 构建保存路径（使用绝对路径确保正确）
        summary_path = os.path.join(current_dir, "summary.csv")
        utils.save_dict_as_csv(summary_results, summary_path, append=not is_first_save)
        logging.info(f"Summary results saved to: {summary_path}")

        if detailed:
            # 确保 detailed 目录存在
            detailed_dir = os.path.join(current_dir, "detailed")
            os.makedirs(detailed_dir, exist_ok=True)
            
            # create a dataframe of all requests, save as csv
            for application_id, result in sched_results.items():
                sched_path = os.path.join(detailed_dir, f"{application_id}.csv")
                utils.save_dict_as_csv(result, sched_path, append=not is_first_save)
                logging.info(f"Scheduler results for {application_id} saved to: {sched_path}")
                
            for application_id, result in alloc_results.items():
                alloc_path = os.path.join(detailed_dir, f"{application_id}_alloc.csv")
                utils.save_dict_as_csv(result, alloc_path, append=not is_first_save)
                logging.info(f"Allocator results for {application_id} saved to: {alloc_path}")