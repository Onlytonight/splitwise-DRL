# 强化学习仿真器配置说明

本目录包含强化学习仿真器的配置文件。

## 配置文件

- `rl_sac.yaml`: SAC (Soft Actor-Critic) 算法配置
- `rl_ppo.yaml`: PPO (Proximal Policy Optimization) 算法配置

## 如何切换算法

在 `configs/config.yaml` 中修改 defaults 部分：

```yaml
defaults:
  - simulator: rl_sac  # 使用 SAC 算法
  # 或
  - simulator: rl_ppo  # 使用 PPO 算法
```

## 配置参数说明

### 通用参数

- `algorithm`: 算法类型 (sac/ppo)
- `decision_interval`: 决策间隔（秒）
- `stack_size`: 状态堆叠的时间窗大小
- `debug_features`: 是否启用调试特征

### 状态特征配置

`enabled_features`: 启用的状态特征列表，可选项包括：
- `queue`: 队列长度
- `none_count`: 空闲实例数
- `instance_count`: 实例总数
- `timestamp`: 时间戳
- `rps`: 每秒请求数
- `rps_delta`: RPS 变化率
- `length`: 请求长度
- `rate`: 处理速率
- `util_mem`: 内存利用率
- `draining`: 排空状态
- `p_ins_pending_token`: prompt 实例待处理 token 数
- `queue_delta`: 队列变化率

### 奖励权重配置

`reward_weights`: 奖励函数各项权重
- `w_cost`: 成本惩罚权重
- `w_queue`: 队列惩罚权重
- `w_util`: 利用率权重
- `w_slo`: SLO 违反惩罚权重 (PPO)
- `w_switch`: 动作切换惩罚 (PPO)

### 动作执行参数

- `action_scale_step`: 缩放动作步长
- `action_mig_step`: 迁移动作步长
- `min_instances_per_pool`: 每个池的最小实例数
- `max_total_instances`: 最大总实例数

### 算法特定参数

#### SAC 参数

**网络参数** (`network`):
- `layer_size`: 隐藏层大小

**训练参数** (`training`):
- `replay_buffer_size`: 经验回放缓冲区大小
- `batch_size`: 批量大小
- `train_freq`: 训练频率
- `min_steps_before_training`: 开始训练前的最小步数
- `discount`: 折扣因子
- `soft_target_tau`: 软目标更新系数
- `policy_lr`: 策略网络学习率
- `qf_lr`: Q 网络学习率
- `use_automatic_entropy_tuning`: 是否使用自动熵调节

**优先经验回放** (`prioritized_replay`):
- `alpha`: 优先级指数
- `beta`: 重要性采样初始值
- `beta_increment_per_sampling`: beta 增量
- `eps`: 防止除零的小值

#### PPO 参数

**网络参数** (`network`):
- `hidden_dim`: 隐藏层维度

**训练参数** (`training`):
- `has_continuous_action_space`: 是否使用连续动作空间
- `action_std`: 初始动作方差
- `action_std_decay_rate`: 方差衰减率
- `min_action_std`: 最小动作方差
- `action_std_decay_freq`: 方差衰减频率
- `update_timestep`: 网络更新间隔
- `K_epochs`: 每次更新的 epoch 数
- `eps_clip`: PPO 裁剪参数
- `gamma`: 折扣因子
- `lr_actor`: Actor 学习率
- `lr_critic`: Critic 学习率

### 模型保存

- `save_model_freq`: 模型保存频率（每多少步）
- `checkpoint_dir`: checkpoint 保存目录

## 示例：自定义配置

创建新的配置文件 `configs/simulator/my_config.yaml`：

```yaml
# 继承 SAC 配置并修改部分参数
defaults:
  - rl_sac

# 覆盖特定参数
decision_interval: 5  # 改为 5 秒决策一次

reward_weights:
  w_cost: 0.2  # 增加成本权重
  w_queue: 0.8  # 降低队列权重

training:
  batch_size: 512  # 增大批量大小
  train_freq: 10   # 更频繁地训练
```

然后在 `configs/config.yaml` 中使用：

```yaml
defaults:
  - simulator: my_config
```

## 从命令行覆盖参数

使用 Hydra 的命令行覆盖功能：

```bash
python run.py simulator.decision_interval=5 simulator.reward_weights.w_cost=0.2
```

或者切换算法：

```bash
python run.py simulator=rl_ppo
```

