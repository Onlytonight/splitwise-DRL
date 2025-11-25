import logging
import os
import time

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

import utils

from executor import Executor, ExecutorType
from interconnect import DummyLink
from performance_model import get_duration
from simulator import clock, schedule_event, cancel_event, reschedule_event
from task import Task, TaskType, TokenTask, PromptTask
from flow import FlowType


class Scheduler(ABC):
    """
    Scheduler schedules Requests to Instances and spawns Executors to handle them.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 debug=False):
        self.application = application
        self.router = router
        self.overheads = overheads
        self.executor_overheads = executor_overheads
        self.debug = debug

        # instances
        self.instances = []

        # request queues
        self.pending_queue = []
        self.executing_queue = []
        self.completed_queue = []

        # executors
        self.executor_type = ExecutorType.CentralExecutor
        self.executors = {}

        # logger for scheduler actions
        logger_name = f"schedulers/{self.application.application_id}"
        level = logging.DEBUG if self.debug else logging.INFO
        os.makedirs("schedulers", exist_ok=True)
        self.scheduler_logger = utils.file_logger(logger_name, level=level)
        self.scheduler_logger.info("time,action,info")


    @property
    def application(self):
        return self._application

    @application.setter
    def application(self, application):
        self._application = application

    def add_instance(self, instance):
        """
        Track instances at the scheduler level.
        Helps maintain the scheduler-specific view of instances. 
        """
        self.instances.append(instance)

    @abstractmethod
    def schedule(self, request, *args, **kwargs):
        """
        Main scheduler logic to assign request to instances.
        Called when a request is run.
        Creates a plan for the request.
        """
        raise NotImplementedError

    def request_arrival(self, request):
        """
        Handles the arrival of a new Request.
        """
        request.arrive_at_scheduler()
        self.pending_queue.append(request)
        if len(self.pending_queue) == 1:
            self.run_request(request)

    def request_completion(self, request):
        """
        Handles the completion of a Request.
        """
        request.complete_at_scheduler()
        self.executing_queue.remove(request)
        self.completed_queue.append(request)
        self.router.request_completion(request)

    def run_request(self, request):
        """
        Runs the Request by scheduling it and spawning an Executor.
        """
        request.run_on_executor()
        # measure scheduling overhead
        start = time.time()
        self.schedule(request)
        end = time.time()
        self.scheduler_logger.debug('%s,sched_overhead,%s', clock(), end-start)
        self.spawn_executor(ExecutorType.CentralExecutor,
                            request)
        self.pending_queue.remove(request)
        self.executing_queue.append(request)

    def spawn_executor(self, executor_type, request):
        """
        Spawn an Executor for the request.
        Executors can logically execute anywhere.
        We don't model where they run in simulation.
        """
        executor = Executor.create(executor_type,
                                   request,
                                   self,
                                   self.executor_overheads)
        self.executors[request.request_id] = executor
        executor.run()

    def notify_busy_instance(self, instance):
        """
        Notify to the Scheduler that the instance is busy.
        """

    def notify_free_instance(self, instance):
        """
        Notify to the Scheduler that the instance is free.
        """

    def terminate_executor(self, executor):
        """
        Delete the Executor from the Scheduler.
        """
        del self.executors[executor.request.request_id]

    def save_all_request_metrics(self):
        """
        Saves start and end timestamps for all request nodes.
        Helpful for Gantt charts.
        """
        node_metrics = []
        for request in self.completed_queue:
            node_metrics.extend(request.get_all_node_metrics())
        node_metrics_df = pd.DataFrame(node_metrics)
        node_metrics_df.to_csv("request_nodes.csv", index=False)

    def get_results(self):
        """
        Returns results for all completed requests.   
        """
        array_results = {}

        request_ids = [r.request_id for r in self.completed_queue]
        array_results["request_ids"] = np.array(request_ids)

        response_times = [r.metrics.router_response_time for r in self.completed_queue]
        array_results["response_times"] = np.array(response_times)

        queue_times = [r.metrics.queue_time for r in self.completed_queue]
        array_results["queue_times"] = np.array(queue_times)

        ttft_times = [r.metrics.TTFT for r in self.completed_queue]
        array_results["ttft_times"] = np.array(ttft_times)

        tbt_times = [(r.metrics.router_response_time - r.metrics.TTFT) / (r.token_size)
                     for r in self.completed_queue]
        array_results["tbt_times"] = np.array(tbt_times)

        nth_token_overhead = [r.get_nth_token_overhead() for r in self.completed_queue]
        array_results["nth_token_overheads"] = np.array(nth_token_overhead)

        prompt_sizes = [r.prompt_size for r in self.completed_queue]
        array_results["prompt_sizes"] = np.array(prompt_sizes)

        token_sizes = [r.token_size for r in self.completed_queue]
        array_results["token_sizes"] = np.array(token_sizes)

        return array_results


class KVScheduler(Scheduler):
    """
    KVScheduler is a base class for Schedulers that ship KV caches.
    It does not implement the schedule method.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         debug)
        self.prompt_processors = prompt_processors
        self.token_processors = token_processors
        self.prompt_instances = []
        self.token_instances = []

    def add_instance(self, instance):
        """
        Tracks prompt and token instances differently.
        NOTE: assumes instance tags are distinguishers, not h/w itself
        TODO: make this more flexible and robust
        """
        self.instances.append(instance)
        if instance.tag == "prompt":
            self.prompt_instances.append(instance)
        elif instance.tag == "token":
            self.token_instances.append(instance)
        else:
            # alternative way to distinguish instances
            if isinstance(self.prompt_processors, list):
                if instance.name in self.prompt_processors:
                    self.prompt_instances.append(instance)
                elif instance.name in self.token_processors:
                    self.token_instances.append(instance)
                else:
                    raise ValueError(f"Unsupported instance type: \
                                        {instance.processors[0].name}")

    def add_kv_cache_transfer(self, request, src_instance, dest_instance, bandwidth):
        """
        Convert prompt->token request to prompt->kvtransfer->token request
        by adding a flow node to the request graph.
        """
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        # create new tasks and flows
        flow_size = request.estimate_kv_cache_size(
                                        num_tokens=prompt_task.prompt_size,
                                        model=src_instance.model)
        kv_transfer_flow = request.create_flow(FlowType.KVCacheTransfer,
                                               size=flow_size,
                                               src=src_instance,
                                               dest=dest_instance)
        kv_transfer_flow.notify = True

        # update request DAG
        request.flow_node = kv_transfer_flow
        request.dag.remove_edge(prompt_task, token_task)
        request.dag.add_edge(prompt_task, kv_transfer_flow)
        request.dag.add_edge(kv_transfer_flow, token_task)

        # assign tasks and flows to instances and links
        prompt_task.instance = src_instance
        token_task.instance = dest_instance
        # NOTE: simulate delay by adding a link of configurable bandwidth
        kv_transfer_flow.link = DummyLink(name="DummyLink",
                                          bandwidth=bandwidth)


class RandomScheduler(Scheduler):
    """
    RandomScheduler schedules Requests to Instances randomly.
    """
    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request to a random instance
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = np.random.choice(self.instances)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")


class RoundRobinScheduler(Scheduler):
    """
    RoundRobinScheduler schedules Requests in a round-robin fashion
    across all Instances.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         debug)
        self.instance_index = 0

    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request to the next instance
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = self.instances[self.instance_index]
        self.instance_index = (self.instance_index + 1) % len(self.instances)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")


class JSQScheduler(Scheduler):
    """
    JSQScheduler schedules Requests to the Instance with smallest Request queue.
    Currently uses an inefficient O(n) search.
    """
    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request to the least loaded instance
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = min(self.instances,
                       key=lambda instance: len(instance.pending_requests))
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")


class TokenJSQScheduler(Scheduler):
    """
    JSQScheduler schedules Requests to the Instance with smallest pending tokens queue.
    Currently uses an inefficient O(n) search.
    """
    def schedule(self, request, *args, **kwargs):
        """
        Assigns all nodes in request DAG to the instance with smallest queue
        """
        if len(self.instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        # enable run-to-completion by chaining
        prompt_task.chain = [token_task]

        instance = min(self.instances,
                       key=lambda instance: instance.sched_pending_tokens)
        for node in request.dag.nodes:
            if isinstance(node, Task):
                node.instance = instance
            else:
                raise ValueError(f"Unsupported node type: {type(node)}")

        # bookkeeping
        instance.sched_pending_tokens += prompt_task.prompt_size + 1


class KVRoundRobinScheduler(KVScheduler):
    """
    Schedules requests on prompt and token instances using round-robin.
    Prompt and token instances are not interchangeable.
    Always ships KV-caches from the prompt to the token instances.
    Does not overlap the KV-cache shipping with prompt computation.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []
        self.prompt_instance_index = 0
        self.token_instance_index = 0

    def schedule(self, request, *args, **kwargs):
        """
        Assigns the prompt task to the next fast instance, and the token task
        to the next slow instance in a round-robin fashion.
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            raise ValueError("No instances available")

        prompt_instance = self.prompt_instances[self.prompt_instance_index]
        token_instance = self.token_instances[self.token_instance_index]
        self.prompt_instance_index = (self.prompt_instance_index + 1) % \
                                                len(self.prompt_instances)
        self.token_instance_index = (self.token_instance_index + 1) % \
                                                len(self.token_instances)

        self.add_kv_cache_transfer(request,
                                   prompt_instance,
                                   token_instance,
                                   self.transfer_bandwidth)


class KVJSQScheduler(KVScheduler):
    """
    KVJSQScheduler schedules Requests to the Instance with smallest queue.
    Always ships KV-caches from the prompt to the token instances.
    Currently uses an inefficient O(n) search.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []
        self.prompt_instance_index = 0
        self.token_instance_index = 0

    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by queue length)
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        prompt_instance = min(self.prompt_instances,
                              key=lambda instance: len(instance.pending_requests))
        token_instance = min(self.token_instances,
                             key=lambda instance: len(instance.pending_requests))

        # ship KV-cache between instances
        self.add_kv_cache_transfer(request,
                                   prompt_instance,
                                   token_instance,
                                   self.transfer_bandwidth)


class OverlapKVJSQScheduler(KVJSQScheduler):
    """
    Same as KVJSQScheduler, but overlaps the KV-shipping with prompt.
    Always ships KV-caches from the prompt to the token instances.
    Simulates 90% overlap by using 10x the interconnect bandwidth.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         transfer_bandwidth * 10,
                         debug)


class KVTokenJSQScheduler(KVScheduler):
    """
    KVTokenJSQScheduler schedules Requests to the Instance with smallest pending tokens queue.
    Always ships KV-caches from the prompt to the token instances.
    Currently uses an inefficient O(n) search.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []

    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by queue length)
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        prompt_instance = min(self.prompt_instances,
                              key=lambda instance: instance.sched_pending_tokens)
        token_instance = min(self.token_instances,
                             key=lambda instance: instance.sched_pending_tokens)

        # ship KV-cache between instances
        self.add_kv_cache_transfer(request,
                                   prompt_instance,
                                   token_instance,
                                   self.transfer_bandwidth)


class OverlapKVTokenJSQScheduler(KVTokenJSQScheduler):
    """
    Same as KVTokenJSQScheduler, but overlaps the KV-shipping with prompt.
    Simulates 90% overlap by using 10x the interconnect bandwidth.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         transfer_bandwidth * 10,
                         debug)


class MixedPoolScheduler(KVScheduler):
    """
    MixedPoolScheduler schedules Requests to the Instance with smallest pending tokens queue.
    Always ships KV-caches from the prompt to the token instances.
    Currently uses an inefficient O(n) search.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 prompt_max_pending_batch_tokens,
                 token_max_pending_batch_tokens,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.prompt_max_pending_batch_tokens = prompt_max_pending_batch_tokens
        self.token_max_pending_batch_tokens = token_max_pending_batch_tokens
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.mixed_instances = []
        self.token_instances = []

    def is_memory_loaded(self, instance, tasks):
        """
        Check if instance is loaded by task
        """
        request_memory = sum(task.max_memory(instance) for task in tasks)
        if instance.sched_memory + request_memory >= instance.max_memory:
            return True
        return False

    def is_queue_long(self, instance, task):
        """
        Check if prompt queue is long
        """
        if len(instance.pending_queue) > 0 and \
            instance.sched_pending_tokens + task.tokens_per_iteration > \
                self.prompt_max_pending_batch_tokens:
            return True
        return False

    def find_best_prompt_instance(self, instances, prompt_task):
        """
        Check if prompt queue is long
        """
        if len(instances) == 0:
            return None
        prompt_instance = min(instances,
                              key=lambda instance: instance.sched_pending_tokens)
        if self.is_queue_long(prompt_instance, prompt_task):
            return None
        return prompt_instance

    def find_best_token_instance(self, instances, prompt_task, token_task):
        """
        Checks if instance memory is full
        """
        if len(instances) == 0:
            return None
        token_instance = min(instances,
                             key=lambda instance: (instance.sched_memory))
        if self.is_memory_loaded(token_instance, [prompt_task, token_task]):
            return None
        return token_instance

    def notify_free_instance(self, instance):
        """
        Notifies that a mixed instance is free; moves it to appropriate pool
        """
        if instance.sched_tag == "mixed":
            instance.sched_tag = None
            self.mixed_instances.remove(instance)
            if instance.tag == "prompt":
                self.prompt_instances.append(instance)
            elif instance.tag == "token":
                self.token_instances.append(instance)
            else:
                raise ValueError(f"Unsupported instance tag: {instance.tag} on \
                    {instance.name}_{instance.instance_id}")

    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by queue length)
        """
        if (len(self.prompt_instances) == 0 or len(self.token_instances) == 0) \
            and len(self.mixed_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        # 找到待处理最少的实例，满载则不选
        prompt_instance = None
        for instances in [self.prompt_instances, self.mixed_instances]:
            prompt_instance = self.find_best_prompt_instance(instances, prompt_task)
            if prompt_instance is not None:
                #print("Found prompt in prompt+mixed", clock(), request.request_id)
                break

        token_instance = None
        for instances in [self.token_instances, self.mixed_instances]:
            token_instance = self.find_best_token_instance(instances, prompt_task, token_task)
            if token_instance is not None:
                #print("Found token in token+mixed", clock(), request.request_id)
                break

        #找不到时将专用实例转换为mixed实例
        if prompt_instance is None and len(self.token_instances) > 0:
            # take an instance from token instances and add to mixed instances
            prompt_instance = min(self.token_instances,
                                  key=lambda instance: instance.sched_pending_tokens)
            self.token_instances.remove(prompt_instance)
            self.mixed_instances.append(prompt_instance)
            prompt_instance.sched_tag = "mixed"

        if token_instance is None and len(self.prompt_instances) > 0:
            # take an instance from prompt instances and add to mixed instances
            token_instance = min(self.prompt_instances,
                                 key=lambda instance: (instance.sched_memory))
            self.prompt_instances.remove(token_instance)
            self.mixed_instances.append(token_instance)
            token_instance.sched_tag = "mixed"

        # if we didn't find any instance still, devolve to baseline mixed batching
        if prompt_instance is None or token_instance is None:
            all_instances = self.prompt_instances + self.mixed_instances + self.token_instances
            prompt_instance = min(all_instances,
                                  key=lambda instance: instance.sched_pending_tokens)
            token_instance = prompt_instance

        if prompt_instance != token_instance:
            # ship KV-cache between instances
            self.add_kv_cache_transfer(request,
                                       prompt_instance,
                                       token_instance,
                                       self.transfer_bandwidth)
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance)
            token_instance.sched_memory += prompt_task.max_memory(token_instance) + \
                                           token_task.max_memory(token_instance)
        else:
            # run on same instance
            prompt_task.instance = prompt_instance
            token_task.instance = token_instance
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance) + \
                                            token_task.max_memory(prompt_instance)
            prompt_task.chain = [token_task]

        # bookkeeping
        prompt_instance.sched_pending_tokens += prompt_task.prompt_size
        token_instance.sched_pending_tokens += 1
        print("prompt instance num is", len(self.prompt_instances), ",token instance num is",
              len(self.token_instances), "mixed instance num is", len(self.mixed_instances))


class AdaptiveMixedPoolScheduler(KVScheduler):
    """
    AdaptiveMixedPoolScheduler dynamically converts instances based on queue length ratios.
    When prompt queue length is more than 4x the token pool, or token queue length is less 
    than 1/4 of the prompt pool, it converts the most idle instance from the opposite pool.
    """
    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 prompt_max_pending_batch_tokens,
                 token_max_pending_batch_tokens,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.prompt_max_pending_batch_tokens = prompt_max_pending_batch_tokens
        self.token_max_pending_batch_tokens = token_max_pending_batch_tokens
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # convert to B/s
        self.prompt_instances = []
        self.token_instances = []
        self.load_balance_fac = 2
        self.interval=0
        self.adjust_interval = 1
        print("AdaptiveMixedPoolScheduler initialized,adjust interval is", self.adjust_interval,
              "prompt max pending batch tokens is", self.prompt_max_pending_batch_tokens)
        self.last_completed_count = 0  # 跟踪上次检查时已完成的请求数量
        self.interval_ttft_stats = []  # 存储两次schedule调用间的TTFT统计
        from notebooks.perf_model import PerfModel
        # self.perf_model = PerfModel("D:\homework\网络\论文\LLMshedule\pd分离\splitwise-DRL\data\perf_model.csv", init=True)
        self.perf_model = PerfModel("/home/xfusion/conda/splitwise-DRL/data/perf_model.csv", init=True)


    def is_memory_loaded(self, instance, tasks):
        """
        Check if instance is loaded by task
        """
        request_memory = sum(task.max_memory(instance) for task in tasks)
        if instance.sched_memory + request_memory >= instance.max_memory:
            return True
        return False

    def is_queue_long(self, instance, task):
        """
        Check if prompt queue is long
        """
        if len(instance.pending_queue) > 0 and \
            instance.sched_pending_tokens + task.tokens_per_iteration > \
                self.prompt_max_pending_batch_tokens:
            return True
        return False

    def count_instance_types(self):
        """
        Count instances with different task combinations:
        - Instances with both prompt and token tasks
        - Instances with only prompt tasks
        - Instances with only token tasks
        """
        mixed_instances_count = 0  # Both prompt and token tasks
        prompt_only_instances_count = 0  # Only prompt tasks
        token_only_instances_count = 0   # Only token tasks
        
        # Check all instances (both prompt and token instances)
        all_instances = self.prompt_instances + self.token_instances
        
        for instance in all_instances:
            has_prompt = False
            has_token = False
            
            # Check pending queue
            for task in instance.pending_queue:
                if isinstance(task, PromptTask):
                    has_prompt = True
                elif isinstance(task, TokenTask):
                    has_token = True
            
            # Check executing batch
            for task in instance.batch:
                if isinstance(task, PromptTask):
                    has_prompt = True
                elif isinstance(task, TokenTask):
                    has_token = True
            
            # Categorize instance based on task types
            if has_prompt and has_token:
                mixed_instances_count += 1
            elif has_prompt and not has_token:
                prompt_only_instances_count += 1
            elif has_token and not has_prompt:
                token_only_instances_count += 1
            # Note: We don't count instances with no tasks
        
        return {
            "mixed_instances": mixed_instances_count,           # Both P and T tasks
            "prompt_only_instances": prompt_only_instances_count,  # Only P tasks
            "token_only_instances": token_only_instances_count     # Only T tasks
        }

    def find_best_prompt_instance(self, instances, prompt_task):
        """
        Check if prompt queue is long
        """
        if len(instances) == 0:
            raise ValueError("No prompt instances")
        prompt_instance = min(instances,
                              key=lambda instance: instance.sched_pending_tokens)
        if self.is_queue_long(prompt_instance, prompt_task):
            return None
        return prompt_instance

    def find_best_token_instance(self, instances, prompt_task, token_task):
        """
        Checks if instance memory is full
        """
        if len(instances) == 0:
            raise ValueError("No token instances")
        token_instance = min(instances,
                             key=lambda instance: (instance.sched_memory))
        if self.is_memory_loaded(token_instance, [prompt_task, token_task]):
            return None
        return token_instance

    def transfer_token_to_prompt(self,idlest_token_instance):
        if len(self.token_instances) <= 1:
            # print("can not transfer best token to prompt ")
            return None
        self.token_instances.remove(idlest_token_instance)
        self.prompt_instances.append(idlest_token_instance)
        return idlest_token_instance

    def transfer_best_token_to_prompt(self):
        idlest_token_instance = min(self.token_instances,
                                    key=lambda instance: instance.sched_memory)
        return self.transfer_token_to_prompt(idlest_token_instance)

    def transfer_prompt_to_token(self, idlest_prompt_instance):
        if len(self.prompt_instances) <= 1:
            # print("can not transfer best prompt to token ")
            return None
        self.prompt_instances.remove(idlest_prompt_instance)
        self.token_instances.append(idlest_prompt_instance)
        return idlest_prompt_instance

    def transfer_best_prompt_to_token(self):
        idlest_prompt_instance = min(self.prompt_instances,
                                     key=lambda instance: instance.sched_pending_tokens)
        # 将该prompt实例从prompt池移到token池
        return self.transfer_prompt_to_token(idlest_prompt_instance)

    def notify_free_instance(self, instance):
        if instance.sched_tag == "mixed":
            instance.sched_tag = None
            self.mixed_instances.remove(instance)
            if instance.tag == "prompt":
                self.prompt_instances.append(instance)
            elif instance.tag == "token":
                self.token_instances.append(instance)
            else:
                raise ValueError(f"Unsupported instance tag: {instance.tag} on \
                    {instance.name}_{instance.instance_id}")

    def adjust_instances_by_queue_ratios(self):
        """
        Adjust instances based on queue length ratios between prompt and token instances.
        - If prompt queue is much larger than token queue, convert token instance to prompt
        - If token queue is much larger than prompt queue, convert prompt instance to token
        """
        # Calculate queue ratios for adaptive conversion
        total_prompt_queue = sum(instance.sched_pending_tokens for instance in self.prompt_instances)
        total_token_queue = sum(instance.sched_pending_tokens for instance in self.token_instances)

        # Check if we need to convert instances based on queue ratios
        # Convert token instance to prompt if prompt queue is more than load_balance_fac * token pool
        if total_prompt_queue > self.load_balance_fac * total_token_queue:
            self.transfer_best_token_to_prompt()
        elif total_token_queue > total_prompt_queue * self.load_balance_fac:
            self.transfer_best_prompt_to_token()

    def adjust_instances_dynamically(self):
        # 计算过载的prompt实例数
        overloaded_prompt_count = 0
        for instance in self.prompt_instances:
            if instance.sched_pending_tokens > self.prompt_max_pending_batch_tokens * 0.8:
                overloaded_prompt_count += 1

        # 计算过载的token实例数
        overloaded_token_count = 0
        for instance in self.token_instances:
            if instance.sched_memory > instance.max_memory * 0.8:
                overloaded_token_count += 1

        # 计算过载比例
        prompt_overload_ratio = overloaded_prompt_count / len(self.prompt_instances) if self.prompt_instances else 0
        token_overload_ratio = overloaded_token_count / len(self.token_instances) if self.token_instances else 0

        # 根据相对负载情况调整实例数
        # 如果prompt过载严重而token负载较轻，增加prompt实例
        if prompt_overload_ratio > 0.8 and token_overload_ratio < 0.2 and len(self.token_instances) > 1:
            self.transfer_best_token_to_prompt()

        # 如果token过载严重而prompt负载较轻，增加token实例
        elif token_overload_ratio > 0.8 and prompt_overload_ratio < 0.2 and len(self.prompt_instances) > 1:
            self.transfer_best_prompt_to_token()

        # 如果两者都负载较轻，可以适当减少实例数
        elif prompt_overload_ratio < 0.2 and token_overload_ratio < 0.2:
            # 根据具体情况决定减少哪种实例
            if len(self.prompt_instances) > len(self.token_instances):
                if len(self.prompt_instances) > 1:
                    self.transfer_best_prompt_to_token()
            else:
                if len(self.token_instances) > 1:
                    self.transfer_best_token_to_prompt()

        # 新增基于负载率的动态实例调整函数

    def adjust_instances_by_load_ratio(self):
        """
        Dynamically adjust instances based on load ratio of pending tokens to max batch tokens.
        For prompt instances, load ratio = sched_pending_tokens / prompt_max_pending_batch_tokens.
        For token instances, load ratio = sched_memory / max_memory.
        Converts instances based on average relative load ratios.
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            return

        # Calculate load ratios for prompt instances
        prompt_load_ratios = []
        for instance in self.prompt_instances:
            load_ratio = instance.sched_pending_tokens / self.prompt_max_pending_batch_tokens
            prompt_load_ratios.append(load_ratio)

        # Calculate load ratios for token instances
        token_load_ratios = []
        for instance in self.token_instances:
            load_ratio = instance.sched_memory / instance.max_memory
            token_load_ratios.append(load_ratio)

        # Calculate average load ratios
        avg_prompt_load = sum(prompt_load_ratios) / len(prompt_load_ratios) if prompt_load_ratios else 0
        avg_token_load = sum(token_load_ratios) / len(token_load_ratios) if token_load_ratios else 0

        # If one type is significantly more loaded than the other, convert instances
        if avg_prompt_load - avg_token_load > 0.1:
            # Convert the least loaded token instance to prompt instance
            self.transfer_best_token_to_prompt()
        elif avg_token_load - avg_prompt_load > 0.1:
            # Convert the least loaded prompt instance to token instance
            self.transfer_best_prompt_to_token()

    def get_period_result_(self):
        new_completed_count = len(self.completed_queue)
        if new_completed_count > self.last_completed_count:
            # 有新完成的请求
            newly_completed_requests = self.completed_queue[self.last_completed_count:]
            newly_completed_ttfts = [req.metrics.TTFT for req in newly_completed_requests]

            if newly_completed_ttfts:
                # 计算统计信息
                avg_ttft = sum(newly_completed_ttfts) / len(newly_completed_ttfts)
                self.interval_ttft_stats.append({
                    "timestamp": clock(),
                    "count": len(newly_completed_ttfts),
                    "avg_ttft": avg_ttft,
                    "min_ttft": min(newly_completed_ttfts),
                    "max_ttft": max(newly_completed_ttfts)
                })
                # 可以在这里打印或记录统计信息
                print(f"Between schedules: {len(newly_completed_ttfts)} requests completed, avg TTFT: {avg_ttft:.2f}")

            self.last_completed_count = new_completed_count

    def schedule(self, request, *args, **kwargs):
        """
        Assigns each to the least loaded instance (by queue length)
        Implements adaptive conversion logic based on queue ratios
        """
        if len(self.prompt_instances) == 0 or len(self.token_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        # Find best instances for prompt and token tasks
        prompt_instance = self.find_best_prompt_instance(self.prompt_instances, prompt_task)
        token_instance = self.find_best_token_instance(self.token_instances, prompt_task, token_task)
        # 如果找不到合适的token实例，返回负载最低的prompt实例
        # if prompt_instance is None:
        #     prompt_instance = min(self.token_instances,
        #                           key=lambda instance: instance.sched_pending_tokens)
        #     self.transfer_token_to_prompt(prompt_instance)
        # if token_instance is None:
        #     token_instance = min(self.prompt_instances,
        #                          key=lambda instance: (instance.sched_memory))
        #     self.transfer_prompt_to_token(token_instance)
        # 仍然找不到则返回负载最低实例
        if prompt_instance is None or token_instance is None:
            all_instances = self.prompt_instances + self.token_instances
            prompt_instance = min(all_instances,
                                  key=lambda instance: instance.sched_pending_tokens)
            token_instance = prompt_instance

        self.interval+=1
        if self.interval % self.adjust_interval == 0:
            # self.adjust_instances_dynamically()
            # self.adjust_instances_by_load_ratio()
            res = self.adjust_instances_by_ttft_tbt_ratio()
            # 极端情况，最后一个token实例仍然空闲则允许一个混合实例
            if res=='TTFT':
                token_instance = prompt_instance
            elif res=='TBT':
                prompt_instance = token_instance

            # 统计并输出实例任务类型信息
            instance_stats = self.count_instance_types()
            print(f"实例统计 - 混合任务实例(PT): {instance_stats['mixed_instances']}, "
                  f"纯Prompt实例(P): {instance_stats['prompt_only_instances']}, "
                  f"纯Token实例(T): {instance_stats['token_only_instances']}")

        if prompt_instance is None or token_instance is None:
            raise ValueError("No instances available, load is too high",prompt_instance,token_instance)

        if prompt_instance != token_instance:
            # ship KV-cache between instances
            self.add_kv_cache_transfer(request,
                                       prompt_instance,
                                       token_instance,
                                       self.transfer_bandwidth)
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance)
            token_instance.sched_memory += prompt_task.max_memory(token_instance) + \
                                           token_task.max_memory(token_instance)
        else:
            # run on same instance
            prompt_task.instance = prompt_instance
            token_task.instance = token_instance
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance) + \
                                            token_task.max_memory(prompt_instance)
            prompt_task.chain = [token_task]

        # bookkeeping
        prompt_instance.sched_pending_tokens += prompt_task.prompt_size
        token_instance.sched_pending_tokens += 1


    def adjust_instances_by_ttft_tbt_ratio(self):
        """
        根据p50归一化后的TTFT和TBT值的倍数关系，决定是否进行实例类型转换

        参数:
            p50_normalized_ttft: 归一化后的TTFT p50分位数
            p50_normalized_tbt: 归一化后的TBT p50分位数

        逻辑:
            - 如果TTFT比TBT显著高(倍数超过阈值)，则增加prompt实例(转换token实例)
            - 如果TBT比TTFT显著高(倍数超过阈值)，则增加token实例(转换prompt实例)
        """
        # 设置调整阈值，可根据实际情况调整
        adjust_threshold = 5  # 当一个指标是另一个的1.5倍以上时进行调整
        p50_normalized_ttft,_, p50_normalized_tbt,_ = self.get_period_result()
        # 确保两个指标都有效(不为0)
        if p50_normalized_ttft > 0 and p50_normalized_tbt > 0:
            # 计算TTFT与TBT的比值
            ttft_to_tbt_ratio = p50_normalized_ttft / p50_normalized_tbt
            tbt_to_ttft_ratio = p50_normalized_tbt / p50_normalized_ttft

            # 如果TTFT显著高于TBT，增加prompt实例
            if ttft_to_tbt_ratio > adjust_threshold:
                if len(self.token_instances) > 1:
                    print(
                        f"TTFT ({p50_normalized_ttft:.2f}) 比 TBT ({p50_normalized_tbt:.2f}) 高 {ttft_to_tbt_ratio:.2f}倍，转换token实例到prompt")
                    self.transfer_best_token_to_prompt()
                    return None
                return 'TTFT'
            # 如果TBT显著高于TTFT，增加token实例
            if tbt_to_ttft_ratio > adjust_threshold:
                if len(self.prompt_instances) > 1:
                    print(
                        f"TBT ({p50_normalized_tbt:.2f}) 比 TTFT ({p50_normalized_ttft:.2f}) 高 {tbt_to_ttft_ratio:.2f}倍，转换prompt实例到token")
                    self.transfer_best_prompt_to_token()
                    return None
                return 'TBT'
            # 否则不进行转换
            print(f"TTFT ({p50_normalized_ttft:.2f}) 和 TBT ({p50_normalized_tbt:.2f}) 平衡，无需转换实例")
            return None

    def get_period_result(self):
        new_completed_count = len(self.completed_queue)
        if new_completed_count > self.last_completed_count:
            # 有新完成的请求
            newly_completed_requests = self.completed_queue[self.last_completed_count:]

            # 准备数据用于归一化
            request_data = []
            ttfts = []
            tbts = []

            for req in newly_completed_requests:
                # 获取TTFT
                ttft = req.metrics.TTFT
                ttfts.append(ttft)

                # 计算TBT = (总响应时间 - TTFT) / token_size
                # 假设token_size可以从请求中获取
                token_size = getattr(req, 'token_size', 1)  # 提供默认值以防属性不存在
                tbt = (req.metrics.router_response_time - ttft) / token_size if token_size > 0 else 0
                tbts.append(tbt)

                # 收集数据用于归一化
                request_data.append({
                    'prompt_sizes': getattr(req, 'prompt_size', 0),  # 假设请求有prompt_size属性
                    'ttft': ttft,
                    'tbt': tbt
                })

            if request_data:
                # 创建DataFrame并使用PerfModel进行归一化
                request_df = pd.DataFrame(request_data)
                normalized_df = self.perf_model.add_baseline_perf(request_df)

                # 计算归一化后的TTFT和TBT
                normalized_df['normalized_ttft'] = normalized_df['ttft'] / normalized_df['baseline_ttft']
                normalized_df['normalized_tbt'] = normalized_df['tbt'] / normalized_df['baseline_tbt']

                # 计算归一化后的p50和p99分位数
                p50_normalized_ttft = normalized_df['normalized_ttft'].quantile(0.5)
                p99_normalized_ttft = normalized_df['normalized_ttft'].quantile(0.99)
                p50_normalized_tbt = normalized_df['normalized_tbt'].quantile(0.5)
                p99_normalized_tbt = normalized_df['normalized_tbt'].quantile(0.99)

                # 打印统计信息
                # print(f"Between schedules: {len(ttfts)} requests completed")
                # print(f"Normalized TTFT - P50: {p50_normalized_ttft:.2f}, P99: {p99_normalized_ttft:.2f}")
                # print(f"Normalized TBT - P50: {p50_normalized_tbt:.2f}, P99: {p99_normalized_tbt:.2f}")

                # 返回归一化后的p50和p99分位数
                self.last_completed_count = new_completed_count
                return p50_normalized_ttft, p99_normalized_ttft, p50_normalized_tbt, p99_normalized_tbt

            self.last_completed_count = new_completed_count

        # 如果没有新完成的请求，返回0
        return 0, 0, 0, 0

class LyapunovScheduler(MixedPoolScheduler):
    """
    Lyapunov-based Scheduler that minimizes Drift-plus-Penalty.

    It dynamically trades off between:
    1. Stability (balancing memory pressure and queues to avoid deadlocks)
    2. Performance (minimizing latency)

    Control Parameter: V (v_parameter)
    """

    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 prompt_max_pending_batch_tokens,
                 token_max_pending_batch_tokens,
                 transfer_bandwidth,
                 v_parameter=1.0,  # Weight for Latency (Penalty)
                 beta_mem=10.0,  # Weight for Memory Pressure (Drift)
                 gamma_queue=0.5,  # Weight for Queue backlog (Drift)
                 theta_preempt=5.0,  # Penalty for preemption risk in Mixed
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         prompt_max_pending_batch_tokens,
                         token_max_pending_batch_tokens,
                         transfer_bandwidth,
                         debug)

        self.V = v_parameter
        self.beta_mem = beta_mem
        self.gamma_queue = gamma_queue
        self.theta_preempt = theta_preempt

        # Simple tracker for active transfers to estimate link congestion
        self.active_transfers = 0

    def get_memory_pressure(self, instance):
        """
        Calculates normalized memory pressure (0 to 1+).
        This is the 'Virtual Queue' for VRAM.
        """
        if instance.max_memory == 0: return 1.0
        usage_ratio = instance.sched_memory / instance.max_memory
        # Non-linear penalty when approaching limit to simulate "barrier function"
        # If usage > 90%, pressure spikes to push back traffic
        if usage_ratio > 0.9:
            return usage_ratio * 5
        return usage_ratio

    def estimate_latency(self, prompt_task, token_task,
                         p_instance, t_instance, is_mixed_batch):
        """
        Estimates Expected Latency (The Penalty Term).
        """
        # 1. Compute Latency
        # Simplified estimation: proportional to token sizes
        # In real sim, you might use a regression model or historical average
        compute_speed = 1.0
        if is_mixed_batch:
            compute_speed = 1.0 / 1.10  # 10% overhead for Mixed Batch

        compute_time = (prompt_task.prompt_size + token_task.token_size) / compute_speed

        # 2. Transfer Latency
        transfer_time = 0
        if p_instance != t_instance:
            kv_size = prompt_task.prompt_size * 2 * 2  # simplified KV size formula
            transfer_time = kv_size / self.transfer_bandwidth

            # Add congestion delay estimation
            # latency += active_transfers * small_delay_factor

        # 3. Queueing Delay (Wait time)
        # Estimate based on pending tokens
        wait_time_p = p_instance.sched_pending_tokens * 0.01  # dummy coefficient
        wait_time_t = t_instance.sched_pending_tokens * 0.001

        return compute_time + transfer_time + wait_time_p + wait_time_t

    def calculate_cost(self, prompt_task, token_task, p_instance, t_instance):
        """
        Calculates Drift-plus-Penalty cost for a candidate pair.
        """
        is_same_node = (p_instance == t_instance)
        is_mixed_pool = (getattr(p_instance, 'tag', '') == 'mixed' or \
                         getattr(p_instance, 'sched_tag', '') == 'mixed')

        # --- 1. Drift Terms (Stability) ---

        # Memory Pressure (Critical for avoiding deadlock)
        # We care mostly about the TARGET node's memory
        mem_pressure = self.get_memory_pressure(t_instance)

        # Queue Pressure (Load Balancing)
        queue_pressure = len(p_instance.pending_requests) + \
                         len(t_instance.pending_requests)

        # Preemption Risk (Specific to Mixed Nodes)
        # If a mixed node is running T tasks, inserting a P task causes preemption
        preemption_risk = 0
        if is_mixed_pool:
            # If mixed node has many pending tokens, P task will hurt them
            preemption_risk = t_instance.sched_pending_tokens

        drift = (self.beta_mem * mem_pressure) + \
                (self.gamma_queue * queue_pressure) + \
                (self.theta_preempt * preemption_risk)

        # --- 2. Penalty Term (Performance) ---
        expected_latency = self.estimate_latency(prompt_task, token_task,
                                                 p_instance, t_instance,
                                                 is_mixed_batch=is_mixed_pool)

        penalty = self.V * expected_latency

        return drift + penalty

    def schedule(self, request, *args, **kwargs):
        """
        Lyapunov scheduling logic:
        Evaluate all valid (P, T) pairs and pick the one with min(Drift + Penalty).
        """
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        best_cost = float('inf')
        best_pair = (None, None)

        # Candidates Construction
        # 1. Splitwise Candidates (P_pool -> T_pool)
        # We don't scan all pairs (O(N*M) too slow), we sample best from each pool
        # Or for simulation scale, we can iterate all if N is small (<50)

        # Helper to filter valid candidates (Hard Constraints)
        def is_valid(p, t):
            # Hard memory constraint check
            req_mem = 0
            if p == t:
                req_mem = prompt_task.max_memory(t) + token_task.max_memory(t)
            else:
                # If split, we only check T's memory for the token phase
                # (Simulating P-lock release is complex, simplified here to Check T)
                req_mem = token_task.max_memory(t)

            # Using sched_memory which tracks reservations
            return (t.sched_memory + req_mem) <= t.max_memory

        # Strategy: Evaluate Top-K candidates from each category to save time
        candidates = []

        # Path A: Standard Splitwise (P_pool -> T_pool)
        # Heuristic: Pick top 3 least loaded P and top 3 least loaded T
        top_p = sorted(self.prompt_instances, key=lambda x: len(x.pending_requests))[:3]
        top_t = sorted(self.token_instances, key=lambda x: x.sched_memory)[:3]

        for p in top_p:
            for t in top_t:
                if is_valid(p, t):
                    candidates.append((p, t))

        # Path B: Mixed/Local Execution (Mixed_pool or Borrowed)
        # Includes current Mixed instances AND potential borrows
        potential_mixed = self.mixed_instances + \
                          self.prompt_instances + \
                          self.token_instances

        # Pick top candidates for local execution
        top_mixed = sorted(potential_mixed, key=lambda x: x.sched_memory)[:5]

        for m in top_mixed:
            if is_valid(m, m):
                candidates.append((m, m))

        # --- Optimization Step ---
        for (p, t) in candidates:
            cost = self.calculate_cost(prompt_task, token_task, p, t)
            if cost < best_cost:
                best_cost = cost
                best_pair = (p, t)

        # Dispatch
        p_instance, t_instance = best_pair

        if p_instance is None:
            # Fallback: If no valid assignment found (Cluster Full),
            # queue locally or handle rejection.
            # Here we revert to simple RoundRobin or wait (raise Error for now)
            # In robust system: return and retry later
            print(f"WARNING: Cluster Saturated. Req {request.request_id}")
            # Attempt strict fallback to min-memory node
            all_nodes = self.prompt_instances + self.token_instances + self.mixed_instances
            t_instance = min(all_nodes, key=lambda x: x.sched_memory)
            p_instance = t_instance

        # Handle Mixed Instance Logic (Tagging)
        if p_instance == t_instance:
            # If we chose a node that isn't 'mixed' yet, convert/tag it temporarily
            if p_instance not in self.mixed_instances and \
                    getattr(p_instance, 'sched_tag', None) != 'mixed':

                # Remove from original lists if necessary or just tag
                if p_instance in self.prompt_instances:
                    self.prompt_instances.remove(p_instance)
                elif p_instance in self.token_instances:
                    self.token_instances.remove(p_instance)

                self.mixed_instances.append(p_instance)
                p_instance.sched_tag = "mixed"

        # Apply Schedule
        if p_instance != t_instance:
            self.add_kv_cache_transfer(request, p_instance, t_instance, self.transfer_bandwidth)
            self.active_transfers += 1  # Increment transfer counter

            # Update sched_memory
            p_instance.sched_memory += prompt_task.max_memory(p_instance)
            # T reserves memory NOW to avoid future deadlock
            t_instance.sched_memory += prompt_task.max_memory(t_instance) + \
                                       token_task.max_memory(t_instance)
        else:
            prompt_task.instance = p_instance
            token_task.instance = t_instance
            prompt_task.chain = [token_task]
            p_instance.sched_memory += prompt_task.max_memory(p_instance) + \
                                       token_task.max_memory(p_instance)

        # Bookkeeping
        p_instance.sched_pending_tokens += prompt_task.prompt_size
        t_instance.sched_pending_tokens += 1


class SeparateMixedScheduler(KVScheduler):
    """
    SeparateMixedScheduler 将所有实例设置为混合模式，但为prompt和token任务分别查找最佳实例。
    这样既能充分利用所有实例的混合处理能力，又能通过分别优化任务分配来提高整体性能。
    """

    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 prompt_max_pending_batch_tokens,
                 token_max_pending_batch_tokens,
                 transfer_bandwidth,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        self.prompt_max_pending_batch_tokens = prompt_max_pending_batch_tokens
        self.token_max_pending_batch_tokens = token_max_pending_batch_tokens
        self.transfer_bandwidth = transfer_bandwidth * 1024 ** 3  # convert to B/s
        self.mixed_instances = []  # 所有实例都在混合实例池中
        # 保留这些列表以保持兼容性，但不会主动使用它们
        self.prompt_instances = []
        self.token_instances = []
        print("SeparateMixedScheduler initialized - All instances will be used in mixed mode")

    def add_instance(self, instance):
        """添加实例到混合实例池"""
        super().add_instance(instance)
        # 确保所有实例都被添加到混合实例列表
        if instance not in self.mixed_instances:
            self.mixed_instances.append(instance)
            instance.sched_tag = "mixed"  # 标记为混合实例
        # 从专用池中移除（如果存在）
        if instance in self.prompt_instances:
            self.prompt_instances.remove(instance)
        if instance in self.token_instances:
            self.token_instances.remove(instance)

    def is_memory_loaded(self, instance, tasks):
        """检查实例内存是否已满"""
        request_memory = sum(task.max_memory(instance) for task in tasks)
        if instance.sched_memory + request_memory >= instance.max_memory:
            return True
        return False

    def is_queue_long(self, instance, task):
        """检查队列是否过长"""
        if len(instance.pending_queue) > 0 and \
                instance.sched_pending_tokens + task.tokens_per_iteration > \
                self.prompt_max_pending_batch_tokens:
            return True
        return False

    def find_best_prompt_instance(self, instances, prompt_task):
        """为prompt任务查找最佳混合实例"""
        if len(instances) == 0:
            return None

        # 根据队列长度选择最佳实例
        prompt_instance = min(instances,
                              key=lambda instance: instance.sched_pending_tokens)

        # 检查队列是否过长
        if self.is_queue_long(prompt_instance, prompt_task):
            return None

        return prompt_instance

    def find_best_token_instance(self, instances, prompt_task, token_task):
        """为token任务查找最佳混合实例"""
        if len(instances) == 0:
            return None

        # 根据内存使用选择最佳实例
        token_instance = min(instances,
                             key=lambda instance: instance.sched_memory)

        # 检查内存是否足够
        if self.is_memory_loaded(token_instance, [prompt_task, token_task]):
            return None

        return token_instance

    def notify_free_instance(self, instance):
        """通知实例空闲，但保持其混合模式"""
        # 即使空闲也保持为混合实例
        if instance.sched_tag != "mixed" and instance in self.mixed_instances:
            instance.sched_tag = "mixed"
        # 不从混合池中移除实例，确保所有实例始终可用作混合实例

    def schedule(self, request, *args, **kwargs):
        """
        调度请求：为prompt和token任务分别在混合实例池中查找最佳实例
        """
        if len(self.mixed_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        # 分别为prompt和token任务在混合实例池中查找最佳实例
        prompt_instance = self.find_best_prompt_instance(self.mixed_instances, prompt_task)
        token_instance = self.find_best_token_instance(self.mixed_instances, prompt_task, token_task)

        # 如果找不到合适的实例，回退到负载最低的实例
        if prompt_instance is None or token_instance is None:
            prompt_instance = min(self.mixed_instances,
                                  key=lambda instance: instance.sched_pending_tokens)
            token_instance=prompt_instance

        if prompt_instance != token_instance:
            # 如果实例不同，需要传输KV缓存
            self.add_kv_cache_transfer(request,
                                       prompt_instance,
                                       token_instance,
                                       self.transfer_bandwidth)
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance)
            token_instance.sched_memory += prompt_task.max_memory(token_instance) + \
                                           token_task.max_memory(token_instance)
        else:
            # 在同一实例上运行
            prompt_task.instance = prompt_instance
            token_task.instance = token_instance
            prompt_instance.sched_memory += prompt_task.max_memory(prompt_instance) + \
                                            token_task.max_memory(prompt_instance)
            prompt_task.chain = [token_task]

        # 更新待处理令牌数
        prompt_instance.sched_pending_tokens += prompt_task.prompt_size
        token_instance.sched_pending_tokens += 1
        
class UnifiedFluidScheduler(KVScheduler):
    """
    Fixed Version: Includes memory release logic to prevent deadlock/crashes.
    """

    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 interference_penalty=0.3, # 30% 混合惩罚
                 enable_pipelining=True,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        
        self.unified_instances = [] 
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 
        self.interference_penalty = interference_penalty
        self.enable_pipelining = enable_pipelining

    def add_instance(self, instance):
        super().add_instance(instance)
        if instance not in self.unified_instances:
            self.unified_instances.append(instance)
            # [Fix 1] 确保属性初始化，防止 AttributeError 或 脏数据
            if not hasattr(instance, 'sched_memory'):
                instance.sched_memory = 0
            if not hasattr(instance, 'sched_pending_tokens'):
                instance.sched_pending_tokens = 0

    def estimate_compute_time(self, task, instance):
        # 简单估算模型，如果有更复杂的 Profiler 可以替换
        if task.task_type == "prompt":
            return task.prompt_size * 0.0005 
        else:
            return 1 * 0.001 

    def calculate_cost(self, p_node, t_node, request, prompt_task, token_task):
        # 1. Load Balancing Cost
        wait_cost_p = len(p_node.pending_requests) * 0.01 
        wait_cost_t = len(t_node.pending_requests) * 0.01

        # 2. Transfer Cost
        transfer_cost = 0
        kv_size = request.estimate_kv_cache_size(prompt_task.prompt_size, p_node.model)
        
        if p_node != t_node:
            raw_transfer_time = kv_size / self.transfer_bandwidth
            if self.enable_pipelining:
                transfer_cost = raw_transfer_time * 0.1 
            else:
                transfer_cost = raw_transfer_time

        # 3. Interference Cost
        interference_cost = 0
        if p_node == t_node:
            # 如果节点已经在做 Token 生成 (Load > 10)，施加混合惩罚
            if p_node.sched_pending_tokens > 10: 
                estimated_p_time = self.estimate_compute_time(prompt_task, p_node)
                interference_cost = estimated_p_time * self.interference_penalty

        return wait_cost_p + wait_cost_t + transfer_cost + interference_cost

    def check_memory_feasibility(self, instance, new_memory_demand):
        available = instance.max_memory - instance.sched_memory
        # 加一个小 buffer (例如 1MB) 防止浮点数误差
        return available >= new_memory_demand - 1e6

    # --- 调度核心入口 ---
    def run_request(self, request):
        request.run_on_executor()
        success = self.schedule(request)
        
        if not success:
            # Backpressure: 留在 pending_queue 中等待
            return

        self.spawn_executor(self.executor_type, request)
        self.pending_queue.remove(request)
        self.executing_queue.append(request)

    # --- [Fix 2] 核心修复：资源释放逻辑 ---
    def request_completion(self, request):
        """
        当请求完成时调用，用于释放显存占用 (sched_memory) 并触发排队任务
        """
        # 1. 找到该请求当时占用的节点
        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))
        
        p_node = prompt_task.instance
        t_node = token_task.instance

        # 2. 计算需要释放的显存 (逻辑是 schedule 的逆过程)
        
        # P 节点任务完成 (临时显存释放)
        p_mem_freed = prompt_task.max_memory(p_node)
        p_node.sched_memory -= p_mem_freed
        if p_node.sched_memory < 0: p_node.sched_memory = 0
        p_node.sched_pending_tokens -= prompt_task.prompt_size

        # T 节点任务完成 (KV Cache 释放)
        # 注意：在 schedule 里，如果是远程传输，t_node 增加了 (p_max + t_max)
        # 如果是本地，也增加了 (p_max + t_max)
        # 所以逻辑是通用的：
        
        # 只有当 p 和 t 不是同一个节点时，我们需要分别减。
        # 如果是同一个节点，我们需要减一次总和，避免双重扣减的错觉
        
        if p_node != t_node:
             t_mem_freed = prompt_task.max_memory(t_node) + token_task.max_memory(t_node)
             t_node.sched_memory -= t_mem_freed
             if t_node.sched_memory < 0: t_node.sched_memory = 0
             t_node.sched_pending_tokens -= 1
        else:
             # 如果是同一节点，上面已经减去了 p_mem_freed
             # 这里还需要减去 t_task 部分以及 KV 占用
             # 本地调度的内存是：p_max + t_max
             # 上面减了 p_max，这里还需要减 t_max
             t_node.sched_memory -= token_task.max_memory(t_node)
             if t_node.sched_memory < 0: t_node.sched_memory = 0
             t_node.sched_pending_tokens -= 1

        # 3. 调用父类处理（通知 Router 等）
        super().request_completion(request)

        # 4. [重要] 有资源释放了，尝试运行积压的任务
        self.try_schedule_pending()

    def notify_free_instance(self, instance):
        # 硬件通知空闲，也尝试调度
        self.try_schedule_pending()

    def try_schedule_pending(self):
        """
        Head-of-Line Blocking Retry: 尝试调度队列头的任务
        """
        if len(self.pending_queue) > 0:
            next_req = self.pending_queue[0]
            # 再次调用 run_request (它内部会调用 schedule)
            self.run_request(next_req)

    def schedule(self, request, *args, **kwargs):
        if len(self.unified_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        best_cost = float('inf')
        best_p_instance = None
        best_t_instance = None
        
        candidates = np.random.permutation(self.unified_instances)

        for p_cand in candidates:
            # 显存预检查：Prompt 阶段
            p_mem_req = prompt_task.max_memory(p_cand)
            if not self.check_memory_feasibility(p_cand, p_mem_req):
                continue 

            for t_cand in candidates:
                # 显存预检查：Token 阶段 (KV Storage)
                # Admission Control 防死锁的关键
                if p_cand == t_cand:
                    total_mem_req = p_mem_req + token_task.max_memory(t_cand)
                    if not self.check_memory_feasibility(t_cand, total_mem_req):
                        continue
                else:
                    t_mem_req = p_mem_req + token_task.max_memory(t_cand) 
                    if not self.check_memory_feasibility(t_cand, t_mem_req):
                        continue

                cost = self.calculate_cost(p_cand, t_cand, request, prompt_task, token_task)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p_instance = p_cand
                    best_t_instance = t_cand

        # 依然没找到位置：返回 False
        if best_p_instance is None or best_t_instance is None:
            return False

        # --- 资源锁定与 DAG 修改 ---
        if best_p_instance != best_t_instance:
            self.add_kv_cache_transfer(request,
                                       best_p_instance,
                                       best_t_instance,
                                       self.transfer_bandwidth)
            
            best_p_instance.sched_memory += prompt_task.max_memory(best_p_instance)
            best_t_instance.sched_memory += (prompt_task.max_memory(best_t_instance) + 
                                             token_task.max_memory(best_t_instance))
        else:
            prompt_task.instance = best_p_instance
            token_task.instance = best_t_instance
            prompt_task.chain = [token_task]
            
            total_req = prompt_task.max_memory(best_p_instance) + token_task.max_memory(best_t_instance)
            best_p_instance.sched_memory += total_req

        best_p_instance.sched_pending_tokens += prompt_task.prompt_size
        best_t_instance.sched_pending_tokens += 1
        
        return True
    """
    Fixed Version: Handles Backpressure correctly by overriding run_request.
    """

    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 interference_penalty=0.3,
                 enable_pipelining=True,
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        
        # 统一资源池逻辑
        self.unified_instances = [] 
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 
        self.interference_penalty = interference_penalty
        self.enable_pipelining = enable_pipelining

    def add_instance(self, instance):
        super().add_instance(instance)
        if instance not in self.unified_instances:
            self.unified_instances.append(instance)

    def estimate_compute_time(self, task, instance):
        if task.task_type == "prompt":
            return task.prompt_size * 0.0005 
        else:
            return 1 * 0.001 

    def calculate_cost(self, p_node, t_node, request, prompt_task, token_task):
        # 1. Load Balancing Cost
        wait_cost_p = len(p_node.pending_requests) * 0.01 
        wait_cost_t = len(t_node.pending_requests) * 0.01

        # 2. Transfer Cost
        transfer_cost = 0
        kv_size = request.estimate_kv_cache_size(prompt_task.prompt_size, p_node.model)
        
        if p_node != t_node:
            raw_transfer_time = kv_size / self.transfer_bandwidth
            if self.enable_pipelining:
                transfer_cost = raw_transfer_time * 0.1 
            else:
                transfer_cost = raw_transfer_time

        # 3. Interference Cost
        interference_cost = 0
        if p_node == t_node:
            if p_node.sched_pending_tokens > 10: 
                estimated_p_time = self.estimate_compute_time(prompt_task, p_node)
                interference_cost = estimated_p_time * self.interference_penalty

        return wait_cost_p + wait_cost_t + transfer_cost + interference_cost

    def check_memory_feasibility(self, instance, new_memory_demand):
        available = instance.max_memory - instance.sched_memory
        return available >= new_memory_demand

    # --- 修复核心：覆写 run_request ---
    def run_request(self, request):
        """
        Override standard run_request to support Backpressure (waiting).
        """
        request.run_on_executor()
        
        # 尝试调度
        success = self.schedule(request)

        if not success:
            # [关键修复] 如果调度失败（反压），直接返回，不生成Executor，不从 pending_queue 移除
            if self.debug:
                print(f"[FluidSched] Backpressure activated for Req {request.request_id}. Waiting in queue.")
            return

        # 如果成功，继续执行基类的标准流程
        self.spawn_executor(self.executor_type, request)
        self.pending_queue.remove(request)
        self.executing_queue.append(request)

    # --- 修复核心：实现重试机制 ---
    def notify_free_instance(self, instance):
        """
        Trigger retry when an instance frees up memory/slots.
        """
        # 必须先更新资源的 sched_memory 等状态（如果有相应逻辑）
        # 这里简化处理：只要有节点空闲，就尝试看看 pending_queue 里的任务能不能做了
        if len(self.pending_queue) > 0:
            # 尝试重新运行队头的任务
            next_request = self.pending_queue[0]
            
            # 注意：run_request 会再次调用 schedule。
            # 如果这次成功了，它会把自己从 pending_queue 移走。
            # 这是一个简单的 Head-of-Line 重试逻辑。
            self.run_request(next_request)

    def schedule(self, request, *args, **kwargs):
        """
        Returns True if scheduled successfully, False otherwise.
        """
        if len(self.unified_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        best_cost = float('inf')
        best_p_instance = None
        best_t_instance = None
        
        candidates = np.random.permutation(self.unified_instances)

        for p_cand in candidates:
            # 检查显存（Prompt任务临时显存）
            p_mem_req = prompt_task.max_memory(p_cand)
            if not self.check_memory_feasibility(p_cand, p_mem_req):
                continue 

            for t_cand in candidates:
                # 检查显存（预约制，防止死锁）
                if p_cand == t_cand:
                    total_mem_req = p_mem_req + token_task.max_memory(t_cand)
                    if not self.check_memory_feasibility(t_cand, total_mem_req):
                        continue
                else:
                    t_mem_req = p_mem_req + token_task.max_memory(t_cand) 
                    if not self.check_memory_feasibility(t_cand, t_mem_req):
                        continue

                cost = self.calculate_cost(p_cand, t_cand, request, prompt_task, token_task)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p_instance = p_cand
                    best_t_instance = t_cand

        # [关键修复] 如果没找到路径，返回 False 通知 run_request 等待
        if best_p_instance is None or best_t_instance is None:
            return False

        # --- Execution Assignment ---
        if best_p_instance != best_t_instance:
            self.add_kv_cache_transfer(request,
                                       best_p_instance,
                                       best_t_instance,
                                       self.transfer_bandwidth)
            best_p_instance.sched_memory += prompt_task.max_memory(best_p_instance)
            best_t_instance.sched_memory += (prompt_task.max_memory(best_t_instance) + 
                                             token_task.max_memory(best_t_instance))
        else:
            prompt_task.instance = best_p_instance
            token_task.instance = best_t_instance
            prompt_task.chain = [token_task]
            
            total_req = prompt_task.max_memory(best_p_instance) + token_task.max_memory(best_t_instance)
            best_p_instance.sched_memory += total_req

        best_p_instance.sched_pending_tokens += prompt_task.prompt_size
        best_t_instance.sched_pending_tokens += 1
        
        # 返回 True 表示调度成功
        return True
    """
    Fluid Scheduler:
    1. Removes strict distinctions between Prompt and Token pools. 
       All instances are treated as Unified Workers.
    2. Decisions are made per-request based on a Cost Model:
       Cost = ExecutionTime + QueueWait + TransferLatency + InterferencePenalty
    3. Prevents Deadlock via Strict Memory Reservation (Admission Control).
    """

    def __init__(self,
                 application,
                 router,
                 overheads,
                 executor_overheads,
                 prompt_processors,
                 token_processors,
                 transfer_bandwidth,
                 interference_penalty=0.2, # 20% slowdown if mixing P and T
                 enable_pipelining=True,   # If True, assumes computation overlaps transfer
                 debug=False):
        super().__init__(application,
                         router,
                         overheads,
                         executor_overheads,
                         prompt_processors,
                         token_processors,
                         debug)
        
        # Unified pool: We don't care about tags anymore.
        # But we merge them to track all available resources.
        self.unified_instances = [] 
        self.transfer_bandwidth = transfer_bandwidth * 1024**3 # GB/s -> B/s
        
        # Model Parameters
        self.interference_penalty = interference_penalty
        self.enable_pipelining = enable_pipelining

    def add_instance(self, instance):
        # Override parent method to simply add to the unified list
        super().add_instance(instance)
        # We assume all hardware is capable of doing both, 
        # or at least the Scheduler treats them as potential candidates.
        if instance not in self.unified_instances:
            self.unified_instances.append(instance)

    def estimate_compute_time(self, task, instance):
        """
        Estimate P-phase or T-phase execution time.
        In a real scenario, this uses a Profiler. 
        Here we simplify using the model's iteration logic.
        """
        # This is a heuristic proxy.
        # Ideally, use instance.get_duration(task) if accessible without running.
        # Here we assume homogenous compute roughly, scaling by prompt size.
        if task.task_type == "prompt":
            return task.prompt_size * 0.0005 # Hypothetical coefficient
        else:
            return 1 * 0.001 # Decoding one token takes longer relative to prefill/token

    def calculate_cost(self, p_node, t_node, request, prompt_task, token_task):
        """
        Core logic: Calculate the 'Virtual Cost' of a path.
        """
        
        # 1. Base Wait Cost (Load Balancing)
        # Use queue length as a proxy for wait time
        wait_cost_p = len(p_node.pending_requests) * 0.01 
        wait_cost_t = len(t_node.pending_requests) * 0.01

        # 2. Transfer Cost (Communication)
        transfer_cost = 0
        kv_size = request.estimate_kv_cache_size(prompt_task.prompt_size, p_node.model)
        
        if p_node != t_node:
            raw_transfer_time = kv_size / self.transfer_bandwidth
            
            if self.enable_pipelining:
                # If pipelined, transfer is hidden behind compute (or vice-versa)
                # Cost is max(compute, transfer) - compute [overhead only]
                # For simplicity in cost function, we penalize only the non-overlapped part.
                # Here we conservatively add a fraction of transfer time as latency overhead.
                transfer_cost = raw_transfer_time * 0.1 # Optimistic overlap
            else:
                transfer_cost = raw_transfer_time

        # 3. Interference Cost (Mixing Penalty)
        # If the P-Node is already serving T-tasks (acting as a T-node), 
        # adding a P-task introduces contention (Cache trashing, bandwidth contention).
        interference_cost = 0
        if p_node == t_node:
            # Check if this node is "heavy" with decoding
            if p_node.sched_pending_tokens > 10: # Threshold for "Active T Node"
                # The P task will slow down existing T tasks, and run slower itself.
                estimated_p_time = self.estimate_compute_time(prompt_task, p_node)
                interference_cost = estimated_p_time * self.interference_penalty

        # Total Cost Function
        total_cost = wait_cost_p + wait_cost_t + transfer_cost + interference_cost
        return total_cost

    def check_memory_feasibility(self, instance, new_memory_demand):
        """
        Check if instance has enough space. 
        Critical for preventing Deadlock.
        """
        # sched_memory tracks the reserved memory for scheduled (but maybe not running) tasks
        available = instance.max_memory - instance.sched_memory
        return available >= new_memory_demand

    def schedule(self, request, *args, **kwargs):
        """
        Schedule logic:
        Iterate O(N^2) pairs of (P_Node, T_Node) to find the minimum Cost path.
        N is typically small (number of GPU nodes), so this is fast.
        """
        if len(self.unified_instances) == 0:
            raise ValueError("No instances available")

        prompt_task = request.root_node
        token_task = next(request.successors(prompt_task))

        best_cost = float('inf')
        best_p_instance = None
        best_t_instance = None
        
        # Filter: Only candidates that have memory for at least P-computation (if strictly separate)
        # But usually P-memory is temporary if transferred.
        # We simplify: we need P-memory on P-node (temp) and T-memory on T-node (long-term).
        
        # Heuristic: Shuffle to avoid fixed-order bias
        candidates = np.random.permutation(self.unified_instances)

        # --- Phase 1: Search for Best Path ---
        for p_cand in candidates:
            # 1. Check P-Node Admission (Does it have temp memory for P?)
            # Prompt memory: Model weights (fixed) + KV (temporary generated)
            p_mem_req = prompt_task.max_memory(p_cand)
            if not self.check_memory_feasibility(p_cand, p_mem_req):
                continue 

            for t_cand in candidates:
                # 2. Check T-Node Reservation (Does it have long-term memory for KV?)
                # This is the "Admission Control" preventing OOM/Deadlock.
                # If Local (p=t), check combined requirement.
                if p_cand == t_cand:
                    total_mem_req = p_mem_req + token_task.max_memory(t_cand)
                    if not self.check_memory_feasibility(t_cand, total_mem_req):
                        continue
                else:
                    # Remote: Only need T requirement on T node
                    t_mem_req = p_mem_req + token_task.max_memory(t_cand) 
                    # Note: usually T needs the KV size. max_memory() usually includes KV estimation.
                    if not self.check_memory_feasibility(t_cand, t_mem_req):
                        continue

                # 3. Calculate Score
                cost = self.calculate_cost(p_cand, t_cand, request, prompt_task, token_task)
                
                if cost < best_cost:
                    best_cost = cost
                    best_p_instance = p_cand
                    best_t_instance = t_cand

        # --- Phase 2: Action or Wait ---
        if best_p_instance is None or best_t_instance is None:
            # NO feasible path found (all full).
            # DO NOT schedule. Leave in pending_queue. 
            # This is "Backpressure". 
            # By not scheduling, we avoid creating tasks that stall or crash.
            return

        # --- Phase 3: Execute Assignment ---
        
        # 1. If Remote, Inject Transfer Task
        if best_p_instance != best_t_instance:
            self.add_kv_cache_transfer(request,
                                       best_p_instance,
                                       best_t_instance,
                                       self.transfer_bandwidth)
            # Reserve Memory
            # P-Node reserves P-memory (Temporary, released after transfer theoretically, 
            # but simulator logic usually holds it until task done)
            best_p_instance.sched_memory += prompt_task.max_memory(best_p_instance)
            # T-Node reserves Memory NOW (Reservation)
            best_t_instance.sched_memory += (prompt_task.max_memory(best_t_instance) + 
                                             token_task.max_memory(best_t_instance))
        else:
            # Local Execution
            prompt_task.instance = best_p_instance
            token_task.instance = best_t_instance
            # Enable chaining to minimize overhead
            prompt_task.chain = [token_task]
            
            # Reserve Memory (Combined)
            total_req = prompt_task.max_memory(best_p_instance) + token_task.max_memory(best_t_instance)
            best_p_instance.sched_memory += total_req

        # Bookkeeping for load balancing (metrics)
        best_p_instance.sched_pending_tokens += prompt_task.prompt_size
        best_t_instance.sched_pending_tokens += 1
        
        # Debug Log
        if self.debug:
            action_type = "Local" if best_p_instance == best_t_instance else "Remote"
            print(f"[FluidSched] Req {request.request_id}: {action_type} | P->{best_p_instance.instance_id} T->{best_t_instance.instance_id} Cost:{best_cost:.4f}")