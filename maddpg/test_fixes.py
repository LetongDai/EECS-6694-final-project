#!/usr/bin/env python3
"""
测试脚本：验证 agent.py 和 main.py 的修复
"""

import torch
import numpy as np

print("=" * 60)
print("🧪 测试 Step 1 的修复")
print("=" * 60)

# Test 1: 导入检查
print("\n✅ Test 1: 检查导入")
try:
    from models import Actor, Critic
    from agent import Agent
    from trainer import Trainer
    from replaybuffer import ReplayBuffer
    print("   ✓ 所有模块导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    exit(1)

# Test 2: 创建 Agent 实例
print("\n✅ Test 2: 创建 Agent 实例")
try:
    obs_size = 10
    act_size = 3
    num_agents = 3
    
    agent = Agent(obs_size, act_size, num_agents)
    print(f"   ✓ Agent 创建成功")
    print(f"   - obs_size: {obs_size}")
    print(f"   - act_size: {act_size}")
    print(f"   - num_agents: {num_agents}")
except Exception as e:
    print(f"   ✗ Agent 创建失败: {e}")
    exit(1)

# Test 3: 检查 polyak_avg 方法存在
print("\n✅ Test 3: 检查 polyak_avg 方法")
try:
    assert hasattr(agent, 'polyak_avg'), "polyak_avg 方法不存在"
    agent.polyak_avg()  # 测试调用
    print("   ✓ polyak_avg 方法存在且可调用")
except Exception as e:
    print(f"   ✗ polyak_avg 测试失败: {e}")
    exit(1)

# Test 4: 检查 predict 方法
print("\n✅ Test 4: 测试 predict 方法")
try:
    dummy_obs = torch.randn(5, obs_size)  # batch_size=5
    action = agent.predict(dummy_obs)
    
    assert isinstance(action, np.ndarray), "predict 应该返回 numpy array"
    assert action.shape == (5, act_size), f"动作形状错误: {action.shape}"
    
    print(f"   ✓ predict 方法工作正常")
    print(f"   - 输入形状: {dummy_obs.shape}")
    print(f"   - 输出形状: {action.shape}")
except Exception as e:
    print(f"   ✗ predict 测试失败: {e}")
    exit(1)

# Test 5: 测试 ReplayBuffer
print("\n✅ Test 5: 测试 ReplayBuffer")
try:
    buffer = ReplayBuffer(capacity=1000)
    
    # 添加一些样本
    for i in range(10):
        obs = np.random.randn(num_agents, obs_size)
        acts = np.random.randn(num_agents, act_size)
        rewards = np.random.randn(num_agents)
        next_obs = np.random.randn(num_agents, obs_size)
        dones = np.zeros(num_agents)
        
        buffer.add((obs, acts, rewards, next_obs, dones))
    
    assert len(buffer) == 10, f"Buffer 长度错误: {len(buffer)}"
    
    # 测试采样
    batch = buffer.sample(5)
    assert len(batch) == 5, "采样返回的 tuple 长度应该是 5"
    
    print(f"   ✓ ReplayBuffer 工作正常")
    print(f"   - 容量: 1000")
    print(f"   - 当前大小: {len(buffer)}")
    print(f"   - 采样大小: 5")
except Exception as e:
    print(f"   ✗ ReplayBuffer 测试失败: {e}")
    exit(1)

print("\n" + "=" * 60)
print("🎉 所有测试通过！Step 1 修复成功！")
print("=" * 60)

print("\n📋 修复总结:")
print("   ✓ agent.py: 添加了 Actor, Critic 导入")
print("   ✓ agent.py: 修正了 update_targets() -> polyak_avg()")
print("   ✓ main.py: 移除了多余的 'self' 参数")
print("\n🚀 准备进入 Step 2: 创建环境包装器")