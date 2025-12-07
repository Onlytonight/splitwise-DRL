import heapq
import logging
from collections import defaultdict
import utils
from RL.state import RLStateCollector
from RL.reward import RLRewardCalculator
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
            self.logger.debug(f"{event.time},{event.action},{event.status}")



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


class TraceRLSimulator(TraceSimulator):
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
        self.decision_interval = 10  # 保存间隔

        rl_config = {
            "w_cost": 0.6,
            "w_slo": 2.0,
            "w_switch": 0.15,
            "w_util": 0.05,
            "action_scale_step": 5,
            "action_mig_step": 3,
            "min_instances_per_pool": 2,
            "max_total_instances": 100
        }

        # 用于保存上一次决策时的统计快照，用于计算区间内的速率（Rate）
        self.rl_collector = RLStateCollector(
            cluster=cluster,
            router=router,
            applications=applications,
            stack_size=4
        )
        # 初始化奖励计算器
        self.reward_calculator = RLRewardCalculator(
            config={},
            max_instances=cluster.MAX_INSTANCES
        )
        self.action_executor = RLActionExecutor(
            cluster=cluster,
            config=rl_config  # 从配置中读取步长等参数
        )

        # 用于存储上一步的 Observation，用于 PPO 存储 (s, a, r, s')
        self.last_observation = None
        self.last_action = None

        # --- PPO Hyperparameters (从你的代码中提取并简化) ---
        self.has_continuous_action_space = True
        self.action_std = 0.6  # 初始动作方差
        self.action_std_decay_rate = 0.05
        self.min_action_std = 0.1
        self.action_std_decay_freq = int(2.5e5)  # 步长，注意仿真步长通常比Gym少，需按需调整

        self.update_timestep = 2000  # 每多少个决策步更新一次网络 (类似 max_ep_len * 4)
        self.K_epochs = 80
        self.eps_clip = 0.2
        self.gamma = 0.99
        self.lr_actor = 0.0003
        self.lr_critic = 0.001

        # --- 初始化 Agent ---
        # 状态维数: 64 (stack_size=4 * feature=16)
        # 动作维数: 3 (alpha_p, alpha_t, alpha_mig)
        state_dim = 64
        action_dim = 3

        self.agent = PPO(state_dim, action_dim, self.lr_actor, self.lr_critic,
                         self.gamma, self.K_epochs, self.eps_clip,
                         self.has_continuous_action_space, self.action_std)

        # --- 训练状态追踪 ---
        self.decision_step = 0  # 相当于 time_step
        self.last_observation = None  # s_t
        self.save_model_freq = 10000  # 保存模型频率

        logging.info("TraceSimulator initialized")
        self.load_trace()


    # ... load_trace 方法保持不变 ...

    def run(self):
        # start simulation by scheduling a cluster run
        self.schedule(0, self.cluster.run)
        self.schedule(0, self.router.run)
        self.schedule(0, self.arbiter.run)

        # [新增] 启动决策心跳循环
        # 如果设置了间隔，且大于0，则调度第一次决策
        if self.decision_interval > 0:
            logging.info(f"Starting decision cycle with interval {self.decision_interval}")
            self.schedule(0, self.run_decision_cycle)

        # run simulation
        super().run()
        # ... 后续代码不变 ...

    # [核心] 定义决策周期函数
    def run_decision_cycle(self):
        """
        这是 RL Agent 的核心介入点。
        """
        current_time = self.time

        # ---------------------------------------------------------
        # 1. 状态收集 (State Collection)
        # ---------------------------------------------------------
        state, raw_stats,instance_num = self.rl_collector.get_state_and_stats(
            self.time, self.decision_interval
        )
        logging.debug(f"RL Decision Triggered at time {current_time}")
        reward = 0
        info = {}
        if self.last_observation is not None:
            reward, info = self.reward_calculator.calculate_reward(
                self.cluster,
                self.applications,
                raw_stats , # 包含 SLO 违约率等
                instance_num
            )

            # [关键] 这里通过 PPO 接口存储 Experience Replay
            # self.agent.store_transition(self.last_observation, self.last_action, reward, state)
            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(False)

            # 日志记录 (非常重要)
            if self.decision_step % 100 == 0:
                logging.info(f"Step: {self.decision_step} | Reward: {reward:.4f} | Cost: {info['raw_cost']:.2f}")

        if self.decision_step % self.update_timestep == 0 and self.decision_step > 0:
            logging.info(f"Updating PPO Policy at step {self.decision_step}...")
            self.agent.update()
        if self.has_continuous_action_space and self.decision_step % self.action_std_decay_freq == 0:
            self.agent.decay_action_std(self.action_std_decay_rate, self.min_action_std)
            logging.info(f"Decayed action std to {self.agent.action_std}")
        if self.decision_step % self.save_model_freq == 0 and self.decision_step > 0:
            save_path = f"PPO_checkpoint_{self.decision_step}.pth"
            self.agent.save(save_path)
            logging.info(f"Model saved to {save_path}")

        # ---------------------------------------------------------
        # 2. PPO 推理 (Agent Inference)
        # ---------------------------------------------------------
        action = self.agent.select_action(state)
        # logging.info(f"Agent chose action: {action}")

        # ---------------------------------------------------------
        # 3. 执行动作 (Action Execution)
        # ---------------------------------------------------------
        self.action_executor.execute(action)
        self.last_observation = state
        self.decision_step += 1
        # self.last_action = action

        # ---------------------------------------------------------
        # 4. [关键] 递归调度下一次决策
        # ---------------------------------------------------------
        # 只要还没到结束时间，就安排下一次
        if current_time + self.decision_interval < self.end_time:
            self.schedule(self.decision_interval, self.run_decision_cycle)