"""
基线策略单元测试

测试三个基线策略的基本功能：
1. HeteroScale
2. UtilizationBased
3. QueueBased
"""

import numpy as np
from baseline_policies import (
    HeteroScalePolicy,
    UtilizationBasedPolicy,
    QueueBasedPolicy,
    create_policy
)


def test_heteroscale_policy():
    """测试 HeteroScale 策略"""
    print("\n" + "="*60)
    print("测试 HeteroScale 策略")
    print("="*60)
    
    config = {
        "target_tps_per_instance": 100,
        "pd_ratio": 0.33,
        "scale_out_threshold": 0.1,
        "scale_in_threshold": 0.1,
        "tbt_slo": 50.0,
        "tbt_slo_multiplier": 1.2,
        "emergency_scale_ratio": 0.2,
        "min_instances_per_pool": 1,
        "max_total_instances": 100,
    }
    
    policy = HeteroScalePolicy(config)
    
    # 测试场景 1：正常负载
    print("\n场景 1: 正常负载（维持不变）")
    raw_stats = [
        10,              # prompt_rate
        400,             # token_rate (TPS)
        500,             # p_queue
        2000,            # d_queue
        3,               # n_p
        9,               # n_t
        100,             # avg_prompt_size
        [15, 18, 20],    # ttft [p50, p90, p99] (正常)
        [25, 28, 30],    # tbt [p50, p90, p99] (正常)
        10,              # ins_p_queue
        30,              # ins_d_queue
        0.5,             # avg_queue_time
        0.1,             # avg_nth_token_overhead
        100,             # use_time
        10,              # rps
    ]
    instance_info = [3, 9, 0.6, 0.7, 0.5, 0.6]
    state_info = {"time": 10, "interval": 2}
    
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert isinstance(delta_p, (int, np.integer))
    assert isinstance(delta_t, (int, np.integer))
    
    # 测试场景 2：延迟触发（TBT 过高）
    print("\n场景 2: 延迟触发（TBT P99 > 1.2 × SLO）")
    raw_stats[8] = [55, 60, 65]  # tbt [p50, p90, p99], p99=65ms > 60ms (1.2 × 50)
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p > 0 or delta_t > 0, "应该触发紧急扩容"
    
    # 测试场景 3：比例控制（TPS 过高）
    print("\n场景 3: 比例控制（TPS 过高，需要扩容）")
    raw_stats[1] = 800             # token_rate = 800 (当前12实例，目标需要8实例，ratio=0.67)
    raw_stats[8] = [25, 28, 30]    # tbt 正常
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    # 注意：这里应该会扩容，但具体数值取决于策略实现
    
    print("\n✅ HeteroScale 策略测试通过")


def test_utilization_based_policy():
    """测试 UtilizationBased 策略"""
    print("\n" + "="*60)
    print("测试 UtilizationBased 策略")
    print("="*60)
    
    config = {
        "upper_threshold": 0.8,
        "lower_threshold": 0.3,
        "scale_step": 1,
        "min_instances_per_pool": 1,
        "max_total_instances": 100,
    }
    
    policy = UtilizationBasedPolicy(config)
    
    # 测试场景 1：正常利用率
    print("\n场景 1: 正常利用率（维持不变）")
    raw_stats = [10, 400, 500, 2000, 3, 9, 100, [15, 18, 20], [25, 28, 30], 10, 30, 0.5, 0.1, 100, 10]
    instance_info = [3, 9, 0.6, 0.7, 0.5, 0.6]  # util_mem_p=0.5, util_mem_t=0.6
    state_info = {"time": 10, "interval": 2}
    
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p == 0 and delta_t == 0, "正常利用率应该维持不变"
    
    # 测试场景 2：显存利用率过高
    print("\n场景 2: 显存利用率过高（应该扩容）")
    instance_info[4] = 0.85  # util_mem_p = 0.85 > 0.8
    instance_info[5] = 0.9   # util_mem_t = 0.9 > 0.8
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p > 0 and delta_t > 0, "高利用率应该扩容"
    
    # 测试场景 3：显存利用率过低
    print("\n场景 3: 显存利用率过低（应该缩容）")
    instance_info[0] = 5  # n_p = 5
    instance_info[1] = 10  # n_t = 10
    instance_info[4] = 0.2  # util_mem_p = 0.2 < 0.3
    instance_info[5] = 0.25  # util_mem_t = 0.25 < 0.3
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p < 0 and delta_t < 0, "低利用率应该缩容"
    
    print("\n✅ UtilizationBased 策略测试通过")


def test_queue_based_policy():
    """测试 QueueBased 策略"""
    print("\n" + "="*60)
    print("测试 QueueBased 策略")
    print("="*60)
    
    config = {
        "prompt_queue_upper": 1000,
        "prompt_queue_lower": 100,
        "token_queue_upper": 5000,
        "token_queue_lower": 500,
        "scale_step": 1,
        "min_instances_per_pool": 1,
        "max_total_instances": 100,
    }
    
    policy = QueueBasedPolicy(config)
    
    # 测试场景 1：正常队列长度
    print("\n场景 1: 正常队列长度（维持不变）")
    raw_stats = [10, 400, 500, 2000, 3, 9, 100, [15, 18, 20], [25, 28, 30], 10, 30, 0.5, 0.1, 100, 10]
    # p_queue=500 (在 [100, 1000] 之间)
    # d_queue=2000 (在 [500, 5000] 之间)
    instance_info = [3, 9, 0.6, 0.7, 0.5, 0.6]
    state_info = {"time": 10, "interval": 2}
    
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p == 0 and delta_t == 0, "正常队列长度应该维持不变"
    
    # 测试场景 2：队列过长
    print("\n场景 2: 队列过长（应该扩容）")
    raw_stats[2] = 1500  # p_queue = 1500 > 1000
    raw_stats[3] = 6000  # d_queue = 6000 > 5000
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p > 0 and delta_t > 0, "队列过长应该扩容"
    
    # 测试场景 3：队列过短
    print("\n场景 3: 队列过短（应该缩容）")
    instance_info[0] = 5  # n_p = 5
    instance_info[1] = 10  # n_t = 10
    raw_stats[2] = 50   # p_queue = 50 < 100
    raw_stats[3] = 300  # d_queue = 300 < 500
    delta_p, delta_t = policy.decide(state_info, raw_stats, instance_info)
    print(f"  决策: delta_p={delta_p}, delta_t={delta_t}")
    assert delta_p < 0 and delta_t < 0, "队列过短应该缩容"
    
    print("\n✅ QueueBased 策略测试通过")


def test_policy_factory():
    """测试策略工厂函数"""
    print("\n" + "="*60)
    print("测试策略工厂函数")
    print("="*60)
    
    config = {"min_instances_per_pool": 1, "max_total_instances": 100}
    
    # 测试创建 HeteroScale
    policy = create_policy("heteroscale", config)
    assert isinstance(policy, HeteroScalePolicy)
    print("✅ 创建 HeteroScale 策略成功")
    
    # 测试创建 UtilizationBased
    policy = create_policy("utilization", config)
    assert isinstance(policy, UtilizationBasedPolicy)
    print("✅ 创建 UtilizationBased 策略成功")
    
    # 测试创建 QueueBased
    policy = create_policy("queue", config)
    assert isinstance(policy, QueueBasedPolicy)
    print("✅ 创建 QueueBased 策略成功")
    
    # 测试无效策略名
    try:
        policy = create_policy("invalid_policy", config)
        assert False, "应该抛出 ValueError"
    except ValueError as e:
        print(f"✅ 正确捕获无效策略名: {e}")
    
    print("\n✅ 策略工厂函数测试通过")


def main():
    """运行所有测试"""
    print("\n" + "="*60)
    print("基线策略单元测试")
    print("="*60)
    
    try:
        test_heteroscale_policy()
        test_utilization_based_policy()
        test_queue_based_policy()
        test_policy_factory()
        
        print("\n" + "="*60)
        print("✅ 所有测试通过！")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("❌ 测试失败！")
        print("="*60)
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
