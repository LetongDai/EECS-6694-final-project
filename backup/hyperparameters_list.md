# MADDPG 训练超参数列表

## 📋 完整超参数清单

### 1. 环境超参数 (Environment Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_envs` | 4 | 并行环境数量 |
| `max_steps` | 24 | 每个episode的最大步数（24小时） |
| `use_policy` | True | 是否使用政策激励（绿色能源补贴） |
| `use_time_of_export_pricing` | True | 是否使用动态出口定价 |
| `normalize_rewards` | True | 是否归一化奖励 |

#### 微电网组件容量
| 参数 | 值 | 说明 |
|------|-----|------|
| `wind_capacity` | 100 kW | 风力发电机容量 |
| `solar_capacity` | 50 kW | 光伏系统容量 |
| `diesel_capacity` | 75 kW | 柴油发电机容量 |
| `battery_capacity` | 200 kWh | 电池容量 |
| `battery_max_charge_rate` | 50 kW | 电池最大充电功率 |
| `battery_max_discharge_rate` | 50 kW | 电池最大放电功率 |
| `battery_initial_soc` | 0.5 | 电池初始SOC（50%） |

#### 价格范围
| 参数 | 值 | 说明 |
|------|-----|------|
| `price_min` | 5.0 cents/kWh | 最低电价 |
| `price_max` | 45.0 cents/kWh | 最高电价 |

#### 需求响应约束
| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_CURTAILMENT_RATIO` | 0.30 | 最大削减比例（30%） |
| `K` (不适成本系数) | 10.0 cents/kWh | 客户削减需求的不适成本 |

#### 政策激励参数
| 参数 | 值 | 说明 |
|------|-----|------|
| `RENEWABLE_SUBSIDY` | 3.0 cents/kWh | 可再生能源补贴 |
| `CARBON_TAX` | 2.5 cents/kWh | 碳排放税/污染税 |
| `GREEN_DISCOUNT` | 0.5 | 绿色电力折扣系数 |

---

### 2. 训练超参数 (Training Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `total_episodes` | 200 | 总训练episode数 |
| `batch_size` | 64 | 批次大小 |
| `warmup_episodes` | 10 | 预热episode数（不训练，只收集数据） |
| `log_interval` | 10 | 日志记录间隔（每N个episode） |

---

### 3. 学习率超参数 (Learning Rate Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `actor_lr` | 1e-4 (0.0001) | Actor网络学习率 |
| `critic_lr` | 1e-3 (0.001) | Critic网络学习率 |

---

### 4. RL算法超参数 (RL Algorithm Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `gamma` | 0.95 | 折扣因子（未来奖励的重要性） |
| `tau` | 0.01 | 软更新系数（target network更新速度） |

---

### 5. 经验回放缓冲区超参数 (Replay Buffer Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `buffer_capacity` | 10000 | 经验回放缓冲区容量 |

---

### 6. 探索噪声超参数 (Exploration Noise Hyperparameters)

| 参数 | 值 | 说明 |
|------|-----|------|
| `noise_scale` | 0.1 | 初始噪声缩放因子 |
| `noise_decay` | 0.999 | 噪声衰减率（每个episode） |
| `noise_min` | 0.01 | 最小噪声值 |

---

### 7. 神经网络架构超参数 (Neural Network Architecture)

#### Actor网络
| 参数 | 值 | 说明 |
|------|-----|------|
| `input_size` | obs_size | 输入维度（观察空间大小） |
| `hidden_layer_1` | 128 | 第一隐藏层神经元数 |
| `hidden_layer_2` | 256 | 第二隐藏层神经元数 |
| `output_size` | act_size | 输出维度（动作空间大小） |
| `activation` | ReLU | 隐藏层激活函数 |
| `output_activation` | Tanh | 输出层激活函数（动作范围[-1, 1]） |

#### Critic网络
| 参数 | 值 | 说明 |
|------|-----|------|
| `input_size` | obs_size * num_agents + act_size * num_agents | 输入维度（所有agent的观察+动作） |
| `hidden_layer_1` | 128 | 第一隐藏层神经元数 |
| `hidden_layer_2` | 256 | 第二隐藏层神经元数 |
| `output_size` | 1 | 输出维度（Q值） |
| `activation` | ReLU | 激活函数 |

#### 优化器
| 参数 | 值 | 说明 |
|------|-----|------|
| `optimizer` | Adam | 优化器类型 |
| `actor_optimizer` | Adam(actor_lr) | Actor优化器 |
| `critic_optimizer` | Adam(critic_lr) | Critic优化器 |

---

### 8. Agent配置 (Agent Configuration)

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_agents` | 5 | Agent数量 |
| `agent_names` | ['wind', 'solar', 'diesel', 'battery', 'customer'] | Agent名称列表 |
| `max_act_size` | 2 | 最大动作维度（用于padding） |

#### 各Agent观察空间大小
| Agent | obs_size | 说明 |
|--------|----------|------|
| wind | 4 | 风速、功率输出、时间、电网价格 |
| solar | 4 | 辐照度、功率输出、时间、电网价格 |
| diesel | 5 | 燃料水平、功率输出、时间、电网价格、负荷 |
| battery | 5 | SOC、功率输出、时间、电网价格、负荷 |
| customer | 4 | 总需求、消费、时间、电网价格 |

#### 各Agent动作空间大小
| Agent | act_size | 说明 |
|--------|----------|------|
| wind | 1 | 出价 |
| solar | 1 | 出价 |
| diesel | 2 | 功率比例、出价 |
| battery | 2 | 充放电功率、出价 |
| customer | 1 | 削减比例 |

---

### 9. 路径配置 (Path Configuration)

| 参数 | 值 | 说明 |
|------|-----|------|
| `save_dir` | "debug_checkpoints" | 模型保存目录 |
| `log_dir` | "debug_logs" | 日志保存目录 |
| `exp_name` | "auction_logged_{timestamp}" | 实验名称（自动生成时间戳） |

---

## 📊 超参数总结

### 关键超参数快速参考

```python
# 环境
num_envs = 4
max_steps = 24

# 训练
total_episodes = 200
batch_size = 64
warmup_episodes = 10

# 学习率
actor_lr = 1e-4
critic_lr = 1e-3

# RL参数
gamma = 0.95
tau = 0.01

# 缓冲区
buffer_capacity = 10000

# 噪声
noise_scale = 0.1
noise_decay = 0.999
noise_min = 0.01

# 环境约束
MAX_CURTAILMENT_RATIO = 0.30
K = 10.0  # 不适成本系数

# 政策激励
RENEWABLE_SUBSIDY = 3.0
CARBON_TAX = 2.5
GREEN_DISCOUNT = 0.5
```

---

## 🔧 超参数调优建议

### 学习率
- **Actor LR**: 1e-4 通常较稳定，可尝试 5e-5 到 2e-4
- **Critic LR**: 1e-3 是标准值，可尝试 5e-4 到 2e-3

### 折扣因子
- **Gamma**: 0.95 适合长期任务，短期任务可用 0.9

### 软更新
- **Tau**: 0.01 较保守，可尝试 0.005 到 0.05

### 批次大小
- **Batch Size**: 64 是平衡值，内存充足可增加到 128

### 噪声
- **Noise Scale**: 0.1 是起始值，可根据探索需求调整
- **Noise Decay**: 0.999 意味着每episode衰减0.1%

---

## 📝 注意事项

1. **削减比例约束**: `MAX_CURTAILMENT_RATIO = 0.30` 是硬约束，确保不超过30%
2. **政策激励**: 仅在 `use_policy=True` 时生效
3. **动态定价**: 仅在 `use_time_of_export_pricing=True` 时生效
4. **奖励归一化**: `normalize_rewards=True` 时会对奖励进行归一化处理

