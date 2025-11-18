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
        # if self.is_queue_long(prompt_instance, prompt_task):
        #     return None
        return prompt_instance

    def find_best_token_instance(self, instances, prompt_task, token_task):
        """
        Checks if instance memory is full
        """
        if len(instances) == 0:
            raise ValueError("No token instances")
        token_instance = min(instances,
                             key=lambda instance: (instance.sched_memory))
        # if self.is_memory_loaded(token_instance, [prompt_task, token_task]):
        #     return None
        return token_instance

    def transfer_best_token_to_prompt(self):
        if len(self.token_instances) <= 1:
            # print("can not transfer best token to prompt ")
            return None
        idlest_token_instance = min(self.token_instances,
                                    key=lambda instance: instance.sched_memory)
        self.token_instances.remove(idlest_token_instance)
        self.prompt_instances.append(idlest_token_instance)
        return idlest_token_instance

    def transfer_best_prompt_to_token(self):
        if len(self.prompt_instances) <= 1:
            # print("can not transfer best prompt to token ")
            return None

        idlest_prompt_instance = min(self.prompt_instances,
                                     key=lambda instance: instance.sched_pending_tokens)
        # 将该prompt实例从prompt池移到token池
        self.prompt_instances.remove(idlest_prompt_instance)
        self.token_instances.append(idlest_prompt_instance)
        return idlest_prompt_instance

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

        if prompt_instance is not None and token_instance is not None:
            # 极端情况，最后一个token实例仍然空闲则允许一个混合实例
            if (len(self.prompt_instances)==1 and
                    prompt_instance.sched_pending_tokens < self.prompt_max_pending_batch_tokens * 0.2):
                token_instance = prompt_instance
            if (len(self.token_instances)==1 and
                    token_instance.sched_memory < token_instance.max_memory * 0.2):
                prompt_instance = token_instance
        # else:
        #     if prompt_instance is None:
        #         # 如果找不到合适的prompt实例，返回负载最低的token实例
        #         prompt_instance=self.find_best_token_instance(self.token_instances, prompt_task, token_task)
        #     if token_instance is None:
        #         # 如果找不到合适的token实例，返回负载最低的prompt实例
        #         token_instance=self.find_best_prompt_instance(self.prompt_instances, prompt_task)
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

        self.interval+=1
        if self.interval % self.adjust_interval == 0:
            # self.adjust_instances_dynamically()
            # self.adjust_instances_by_load_ratio()
            self.adjust_instances_by_ttft_tbt_ratio()

            # 统计并输出实例任务类型信息
            instance_stats = self.count_instance_types()
            print(f"实例统计 - 混合任务实例(PT): {instance_stats['mixed_instances']}, "
                  f"纯Prompt实例(P): {instance_stats['prompt_only_instances']}, "
                  f"纯Token实例(T): {instance_stats['token_only_instances']}")

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
        adjust_threshold = 1.5  # 当一个指标是另一个的1.5倍以上时进行调整
        p50_normalized_ttft,_, p50_normalized_tbt,_ = self.get_period_result()
        # 确保两个指标都有效(不为0)
        if p50_normalized_ttft > 0 and p50_normalized_tbt > 0:
            # 计算TTFT与TBT的比值
            ttft_to_tbt_ratio = p50_normalized_ttft / p50_normalized_tbt
            tbt_to_ttft_ratio = p50_normalized_tbt / p50_normalized_ttft

            # 如果TTFT显著高于TBT，增加prompt实例
            if ttft_to_tbt_ratio > adjust_threshold and len(self.token_instances) > 1:
                print(
                    f"TTFT ({p50_normalized_ttft:.2f}) 比 TBT ({p50_normalized_tbt:.2f}) 高 {ttft_to_tbt_ratio:.2f}倍，转换token实例到prompt")
                self.transfer_best_token_to_prompt()
            # 如果TBT显著高于TTFT，增加token实例
            elif tbt_to_ttft_ratio > adjust_threshold and len(self.prompt_instances) > 1:
                print(
                    f"TBT ({p50_normalized_tbt:.2f}) 比 TTFT ({p50_normalized_ttft:.2f}) 高 {tbt_to_ttft_ratio:.2f}倍，转换prompt实例到token")
                self.transfer_best_prompt_to_token()
            # 否则不进行转换
            else:
                print(f"TTFT ({p50_normalized_ttft:.2f}) 和 TBT ({p50_normalized_tbt:.2f}) 平衡，无需转换实例")


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
                print(f"Between schedules: {len(ttfts)} requests completed")
                print(f"Normalized TTFT - P50: {p50_normalized_ttft:.2f}, P99: {p99_normalized_ttft:.2f}")
                print(f"Normalized TBT - P50: {p50_normalized_tbt:.2f}, P99: {p99_normalized_tbt:.2f}")

                # 返回归一化后的p50和p99分位数
                self.last_completed_count = new_completed_count
                return p50_normalized_ttft, p99_normalized_ttft, p50_normalized_tbt, p99_normalized_tbt

            self.last_completed_count = new_completed_count

        # 如果没有新完成的请求，返回0
        return 0, 0, 0, 0