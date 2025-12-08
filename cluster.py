import logging

from collections import defaultdict
from itertools import count

from hydra.utils import instantiate

import hardware_repo

from simulator import clock, schedule_event, cancel_event, reschedule_event
from server import Server


class Cluster:
    """
    Cluster is a collection of Servers and interconnected Links.
    """
    def __init__(self,
                 servers,
                 interconnects,
                 power_budget,
                 cluster_cfg=None):
        self.servers = servers
        self.interconnects = interconnects
        self.power_budget = power_budget
        self.cluster_cfg = cluster_cfg  # 保存集群配置
        self.total_power = 0
        for sku_name in self.servers:
            for server in self.servers[sku_name]:
                server.cluster = self
                self.total_power += server.power
        self.inflight_commands = []

        # logger for simulated power usage (NOTE: currently unsupported)
        #self.power_logger = utils.file_logger("power")
        #self.power_logger.info("time,server,power")

    def __str__(self):
        return "Cluster:" + str(self.servers)

    def add_server(self, server):
        self.servers.append(server)

    def remove_server(self, server):
        self.servers.remove(server)

    def models(self):
        models = []
        for server in self.servers:
            models.extend(server.models)
        return models
    
    def get_cluster_config(self):
        """
        获取集群配置
        
        Returns:
            cluster_cfg: 集群配置对象
        """
        return self.cluster_cfg
    
    def get_server_config(self, sku_name):
        """
        获取指定 SKU 的服务器配置
        
        Args:
            sku_name: 服务器 SKU 名称
            
        Returns:
            server_cfg: 服务器配置对象（如果存在）
        """
        if self.cluster_cfg is None:
            return None
        
        for server_cfg in self.cluster_cfg.servers:
            if server_cfg.sku == sku_name:
                return server_cfg
        return None
    
    def add_server_to_cluster(self, server, sku_name):
        """
        向集群添加服务器
        
        Args:
            server: 要添加的服务器
            sku_name: 服务器 SKU 名称
        """
        server.cluster = self
        if sku_name not in self.servers:
            self.servers[sku_name] = []
        self.servers[sku_name].append(server)
        self.update_power(server.power)
    
    def remove_server_from_cluster(self, server):
        """
        从集群移除服务器
        
        Args:
            server: 要移除的服务器
        """
        for sku_name in self.servers:
            if server in self.servers[sku_name]:
                self.servers[sku_name].remove(server)
                self.update_power(-server.power)
                server.cluster = None
                return True
        return False

    @property
    def power(self, cached=True, servers=None):
        """
        Returns the total power usage of the cluster.
        Can return the cached value for efficiency.
        TODO: unsupported
        """
        if cached and servers is None:
            return self.total_power
        if servers is None:
            servers = self.servers
        return sum(server.power() for server in servers)

    def update_power(self, power_diff):
        """
        Updates the total power usage of the cluster.
        TODO: unsupported
        """
        self.total_power += power_diff

    def power_telemetry(self, power):
        """
        Logs the power usage of the cluster.
        TODO: currently unsupported; make configurable

        Args:
            power (float): The power usage.
        """
        time_interval = 60
        schedule_event(time_interval,
                       lambda self=self, power=self.total_power: \
                           self.power_telemetry(0))

    def run(self):
        """
        Runs servers in the cluster.
        """
        # NOTE: power usage updates not supported
        power = 0
        for sku in self.servers:
            for server in self.servers[sku]:
                server.run()
                power += server.power

    @classmethod
    def from_config(cls, *args, **kwargs):
        # args processing
        cluster_cfg = args[0]
        servers_cfg = cluster_cfg.servers
        interconnects_cfg = cluster_cfg.interconnects

        # instantiate servers
        server_id = count()
        servers = defaultdict(list)
        for server_cfg in servers_cfg:
            for n in range(server_cfg.count):
                sku_cfg = hardware_repo.get_sku_config(server_cfg.sku)
                server = Server.from_config(sku_cfg, server_id=next(server_id))
                servers[server_cfg.sku].append(server)

        # instantiate interconnects
        # TODO: add better network topology / configuration support
        interconnects = []
        for interconnect_cfg in interconnects_cfg:
            if interconnect_cfg.topology == "p2p":
                continue
            interconnect = instantiate(interconnect_cfg)
            interconnects.append(interconnect)

        return cls(servers=servers,
                   interconnects=interconnects,
                   power_budget=cluster_cfg.power_budget,
                   cluster_cfg=cluster_cfg)


if __name__ == "__main__":
    pass
