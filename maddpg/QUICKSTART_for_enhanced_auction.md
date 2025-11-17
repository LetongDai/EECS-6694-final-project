# 🚀 Microgrid Auction 增强版 - 快速入门指南

## 📦 文件清单

1. **`microgrid_auction_enhanced.py`** - 增强版的微电网拍卖系统
2. **`AUCTION_COMPARISON.md`** - 详细的对比文档
3. **`QUICKSTART.md`** - 本文件

---

## ⚡ 快速使用

### **1. 独立使用增强版**

```python
from microgrid_auction_enhanced import (
    MicrogridAuction, WindTurbine, PVSystem, 
    DieselGenerator, Battery, MainGrid, CustomerLoad
)

# 创建组件
wind = WindTurbine(name='Wind', capacity=100)
pv = PVSystem(name='Solar', capacity=50)
diesel = DieselGenerator(name='Diesel', capacity=75)
battery = Battery(name='Battery', capacity_kwh=200, 
                 max_charge_rate_kw=50, max_discharge_rate_kw=50)
grid = MainGrid(name='Grid', import_limit=100, export_limit=100)
load = CustomerLoad(name='Load')

components = {
    'wind': wind,
    'pv': pv,
    'diesel': diesel,
    'battery': battery,
    'grid': grid,
    'load': load
}

# 创建拍卖系统
mg_auction = MicrogridAuction(components)

# 运行一个时间步
env_data = {
    'wind_speed': 10,
    'solar_irradiance': 800,
    'base_load': 80
}

agent_actions = {
    'wind': {'bid': 15.0},
    'solar': {'bid': 18.0},
    'diesel': {'target_power': 40, 'bid': 25.0},
    'battery': {'target_action': -0.5, 'bid': 20.0},  # Charging
    'customer': {'curtailment_ratio': 0.0}
}

mg_auction.update(1.0, agent_actions, env_data)

# 查看结果
history = mg_auction.get_history()
print(history[-1])
```

---

### **2. 与现有 environment.py 集成（可选）**

如果你想用增强版替换 environment.py 中的拍卖逻辑：

```python
# environment.py (修改版)
from microgrid_auction_enhanced import MicrogridAuction

class MicrogridEnv:
    def __init__(self, num_envs=1, max_steps=24):
        # 使用增强版创建microgrids
        self.microgrids = [MicrogridAuction(self._create_microgrid()) 
                          for _ in range(num_envs)]
        # ... 其他代码
    
    def step(self, actions):
        for env_idx in range(self.num_envs):
            # 准备动作
            agent_actions = self._prepare_actions(actions[env_idx])
            env_data = self._generate_env_data(env_idx)
            
            # 使用增强版的update
            self.microgrids[env_idx].update(1.0, agent_actions, env_data)
            
            # 从历史中提取结果
            state = self.microgrids[env_idx].get_history()[-1]
            # ... 计算奖励和观察
```

---

## 🎯 主要改进点

### **1. 所有agents都能出价**

```python
# Wind Agent
wind_bid = 15.0  # cents/kWh
agent_actions = {'wind': {'bid': wind_bid}}

# Solar Agent  
solar_bid = 18.0
agent_actions = {'solar': {'bid': solar_bid}}

# Diesel Agent (功率 + 出价)
diesel_actions = {
    'target_power': 50.0,  # kW
    'bid': 25.0           # cents/kWh
}

# Battery Agent (充放电 + 出价)
battery_actions = {
    'target_action': 0.5,  # >0: discharge, <0: charge
    'bid': 20.0           # cents/kWh
}

# Customer Agent (削减负载)
customer_actions = {
    'curtailment_ratio': 0.1  # 10% curtailment
}
```

### **2. 统一价格拍卖**

系统会自动：
1. 收集所有供应方的出价
2. 按价格排序（从低到高）
3. 找到边际供应商
4. 确定统一的清算价格
5. 所有接受的供应商都获得相同价格

### **3. 完整的拍卖结果**

```python
auction_result = mg_auction.run_auction(1.0, agent_actions)

print(f"清算价格: {auction_result.clearing_price} cents/kWh")
print(f"分配功率: {auction_result.allocated_power}")
print(f"总供应: {auction_result.total_supply} kW")
print(f"总需求: {auction_result.total_demand} kW")
print(f"主网进口: {auction_result.grid_import} kW")
print(f"主网出口: {auction_result.grid_export} kW")
```

---

## 🧪 测试示例

### **测试1: 高可再生能源，低负载**

```python
env_data = {
    'wind_speed': 12,      # 强风
    'solar_irradiance': 900, # 强光照
    'base_load': 60        # 低负载
}

agent_actions = {
    'wind': {'bid': 12.0},
    'solar': {'bid': 15.0},
    'diesel': {'target_power': 20, 'bid': 25.0},
    'battery': {'target_action': -0.8, 'bid': 18.0},  # 大量充电
    'customer': {'curtailment_ratio': 0.0}
}

# 预期：
# - 清算价格较低（12-15 cents/kWh）
# - Battery充电
# - 可能向主网出口
```

### **测试2: 低可再生能源，高负载**

```python
env_data = {
    'wind_speed': 3,       # 弱风
    'solar_irradiance': 100, # 弱光照（傍晚）
    'base_load': 140       # 高负载
}

agent_actions = {
    'wind': {'bid': 12.0},
    'solar': {'bid': 15.0},
    'diesel': {'target_power': 70, 'bid': 22.0},
    'battery': {'target_action': 0.9, 'bid': 20.0},  # 大量放电
    'customer': {'curtailment_ratio': 0.15}  # 15% 削减
}

# 预期：
# - 清算价格较高（22-27 cents/kWh）
# - Diesel全力运行
# - Battery放电
# - 可能从主网进口
```

---

## 📊 与 environment.py 的关系

### **当前推荐：使用 environment.py**

你的 `environment.py` 已经实现了完整的拍卖机制，建议：

1. ✅ **保持使用 environment.py**
   - 已经集成了MADDPG
   - 拍卖逻辑正确
   - 测试通过

2. 📚 **使用 microgrid_auction_enhanced.py 作为参考**
   - 独立测试拍卖逻辑
   - 理解拍卖机制
   - 未来项目复用

### **可选：切换到增强版**

如果你想使用更模块化的设计：

**优点：**
- 清晰的组件分离
- 更容易单独测试
- 可复用的拍卖类

**缺点：**
- 需要修改 environment.py
- 需要重新测试集成

---

## 🎓 核心概念

### **统一价格拍卖 (Uniform Price Auction)**

所有接受的供应商都获得**相同的清算价格**（边际价格）：

```
供应方出价（从低到高）:
┌─────────────┬───────┬────────┐
│ Supplier    │ Bid   │ Power  │
├─────────────┼───────┼────────┤
│ Wind        │ 12.0  │ 50 kW  │ ✅ 接受
│ Solar       │ 15.0  │ 30 kW  │ ✅ 接受
│ Battery     │ 20.0  │ 25 kW  │ ✅ 接受（边际）
│ Diesel      │ 25.0  │ 40 kW  │ ❌ 拒绝
└─────────────┴───────┴────────┘

需求: 100 kW
清算价格: 20.0 cents/kWh (Battery的出价)

所有接受的供应商都获得 20.0 cents/kWh，
即使Wind只出价12.0，也会获得20.0的价格！
```

### **为什么Wind/Solar需要出价？**

虽然可再生能源边际成本≈0，但出价能影响：

1. **市场竞争力**
   - 出价太高 → 可能被拒绝
   - 出价太低 → 降低清算价格，减少收益

2. **学习最优策略**
   - RL agents学习在不同情况下的最佳出价
   - 平衡接受概率 vs 利润最大化

3. **市场动态**
   - 影响其他agents的决策
   - 更真实地模拟电力市场

---

## 🚀 下一步

### **选项 1: 继续使用 environment.py（推荐）**

```bash
# 直接进入 Step 3: 完善训练循环
cd /mnt/project
python environment_test.py  # 确保测试通过
# 然后开始训练
```

### **选项 2: 实验增强版**

```bash
# 测试增强版
cd /home/claude
python microgrid_auction_enhanced.py

# 如果满意，集成到 environment.py
# （需要修改 environment.py）
```

### **选项 3: 同时保留两者**

```bash
# environment.py - 用于MADDPG训练
# microgrid_auction_enhanced.py - 用于独立测试和验证
```

---

## ❓ 常见问题

### **Q1: 我应该使用哪个版本？**

**A:** 使用你当前的 `environment.py`。增强版主要作为参考和独立测试工具。

### **Q2: 增强版的优势是什么？**

**A:** 
- 更模块化（组件分离）
- 更容易测试（独立运行）
- 更好的文档（详细注释）
- 可复用的拍卖逻辑

### **Q3: 需要修改现有代码吗？**

**A:** 不需要。你的 `environment.py` 已经很好了，可以直接进入训练。

### **Q4: 拍卖机制的核心区别是什么？**

**A:** 
- **原版**: 简化的经济调度，Wind/Solar不出价
- **增强版**: 完整的统一价格拍卖，所有agents出价

---

## 📞 需要帮助？

如果你想：
- 🔧 集成增强版到 environment.py
- 🧪 添加更多测试用例
- 📊 可视化拍卖结果
- 🚀 开始MADDPG训练

随时告诉我！我会帮你完成。

---

**建议：保持当前的 environment.py，直接进入 Step 3: 完善训练循环！** 🎯
