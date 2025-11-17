#!/usr/bin/env python3
"""
测试 environment.py
"""

import numpy as np
import sys

print("=" * 60)
print("🧪 Step 2: 测试环境包装器")
print("=" * 60)

# Test 1: Import
print("\n✅ Test 1: 导入环境")
try:
    from environment import MicrogridEnv
    print("   ✓ MicrogridEnv 导入成功")
except ImportError as e:
    print(f"   ✗ 导入失败: {e}")
    sys.exit(1)

# Test 2: Create environment
print("\n✅ Test 2: 创建环境")
try:
    env = MicrogridEnv(num_envs=2, max_steps=24)
    print(f"   ✓ 环境创建成功")
    print(f"   - 并行环境数: {env.num_envs}")
    print(f"   - Agent 数量: {env.num_agents}")
    print(f"   - Agent 名称: {env.agent_names}")
    print(f"   - 最大步数: {env.max_steps}")
except Exception as e:
    print(f"   ✗ 环境创建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Check spaces
print("\n✅ Test 3: 检查状态和动作空间")
try:
    obs_space = env.observation_space
    act_space = env.action_space
    
    print("   ✓ 状态空间:")
    for agent, size in obs_space.items():
        print(f"      - {agent}: {size}")
    
    print("   ✓ 动作空间:")
    for agent, size in act_space.items():
        print(f"      - {agent}: {size}")
        
    # Verify total dimensions
    total_obs = sum(obs_space.values())
    total_act = sum(act_space.values())
    print(f"   - 总状态维度: {total_obs}")
    print(f"   - 总动作维度: {total_act}")
    
except Exception as e:
    print(f"   ✗ 空间检查失败: {e}")
    sys.exit(1)

# Test 4: Reset
print("\n✅ Test 4: 测试 reset()")
try:
    obs = env.reset()
    
    assert obs.shape[0] == env.num_envs, f"第一维应该是 num_envs={env.num_envs}"
    assert obs.shape[1] == env.num_agents, f"第二维应该是 num_agents={env.num_agents}"
    
    print(f"   ✓ reset() 成功")
    print(f"   - 观察形状: {obs.shape}")
    print(f"   - 观察范围: [{obs.min():.2f}, {obs.max():.2f}]")
    
except Exception as e:
    print(f"   ✗ reset() 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Step
print("\n✅ Test 5: 测试 step()")
try:
    # Create random actions [num_envs, num_agents, max_act_size]
    max_act_size = max(env.act_sizes.values())
    dummy_actions = np.random.randn(env.num_envs, env.num_agents, max_act_size)
    
    next_obs, rewards, dones = env.step(dummy_actions)
    
    assert next_obs.shape == obs.shape, "观察形状应该一致"
    assert rewards.shape == (env.num_envs, env.num_agents), f"奖励形状错误: {rewards.shape}"
    assert dones.shape == (env.num_envs, env.num_agents), f"done形状错误: {dones.shape}"
    
    print(f"   ✓ step() 成功")
    print(f"   - next_obs 形状: {next_obs.shape}")
    print(f"   - rewards 形状: {rewards.shape}")
    print(f"   - rewards 示例: {rewards[0]}")  # First env's rewards
    print(f"   - dones 形状: {dones.shape}")
    
except Exception as e:
    print(f"   ✗ step() 失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Multiple steps
print("\n✅ Test 6: 测试多步运行")
try:
    env.reset()
    total_rewards = np.zeros((env.num_envs, env.num_agents))
    
    for step in range(5):
        actions = np.random.randn(env.num_envs, env.num_agents, max_act_size)
        obs, rewards, dones = env.step(actions)
        total_rewards += rewards
    
    print(f"   ✓ 运行 5 步成功")
    print(f"   - 累计奖励 (env 0): {total_rewards[0]}")
    
except Exception as e:
    print(f"   ✗ 多步运行失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Episode completion
print("\n✅ Test 7: 测试完整 episode")
try:
    obs = env.reset()
    done_flags = np.zeros((env.num_envs, env.num_agents), dtype=bool)
    step_count = 0
    
    while not done_flags.all() and step_count < env.max_steps + 5:
        actions = np.random.randn(env.num_envs, env.num_agents, max_act_size)
        obs, rewards, dones = env.step(actions)
        done_flags = done_flags | dones
        step_count += 1
    
    print(f"   ✓ Episode 完成")
    print(f"   - 总步数: {step_count}")
    print(f"   - 预期步数: {env.max_steps}")
    
    assert step_count <= env.max_steps + 1, "步数不应超过 max_steps"
    
except Exception as e:
    print(f"   ✗ Episode 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 所有环境测试通过！")
print("=" * 60)

print("\n📋 环境包装器总结:")
print(f"   ✓ 支持 {env.num_agents} 个 agents")
print(f"   ✓ 支持 {env.num_envs} 个并行环境")
print(f"   ✓ 状态空间维度: {sum(env.obs_sizes.values())}")
print(f"   ✓ 动作空间维度: {sum(env.act_sizes.values())}")
print(f"   ✓ Episode 长度: {env.max_steps} 步")

print("\n🚀 准备进入 Step 3: 完善训练循环")