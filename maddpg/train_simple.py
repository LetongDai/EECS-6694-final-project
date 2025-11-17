#!/usr/bin/env python3
"""
Simple MADDPG Training Script
简化版训练脚本 - 可以直接运行
"""

import torch
import numpy as np
import os
from pathlib import Path

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer

# ============================================================================
# 配置
# ============================================================================

# 环境配置
NUM_ENVS = 2              # 并行环境数（从2开始，稳定后可增加）
MAX_STEPS = 24            # 每个episode的最大步数（24小时）

# 训练配置
TOTAL_EPISODES = 100      # 总训练episodes（先跑100个测试）
ROLLOUT_STEPS = 24        # 每次rollout的步数（一个完整episode）
BATCH_SIZE = 64           # 训练batch大小
TRAIN_FREQ = 1            # 每rollout多少次训练一次

# 学习率
ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

# RL参数
GAMMA = 0.95
TAU = 0.01

# Buffer
BUFFER_CAPACITY = 10000

# 日志
LOG_INTERVAL = 5          # 每N个episodes打印一次
SAVE_INTERVAL = 20        # 每N个episodes保存一次

# 路径
SAVE_DIR = "checkpoints"
Path(SAVE_DIR).mkdir(exist_ok=True)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 70)
    print("🚀 MADDPG Microgrid Training - Simple Version")
    print("=" * 70)
    
    # 1. 创建环境
    print(f"\n🌍 Step 1: 创建环境")
    env = MicrogridEnv(num_envs=NUM_ENVS, max_steps=MAX_STEPS)
    print(f"   ✓ 环境创建成功")
    print(f"   - 并行环境数: {NUM_ENVS}")
    print(f"   - Agents: {env.agent_names}")
    print(f"   - 观察空间: {env.observation_space}")
    print(f"   - 动作空间: {env.action_space}")
    
    # 2. 创建Agents
    agents = []
    max_act_size = 2  # diesel和battery的最大动作维度

    for i, agent_name in enumerate(env.agent_names):
        obs_size = env.obs_sizes[agent_name]
        act_size = env.act_sizes[agent_name]
        
        actual_obs_size = 5
        
        agent = Agent(
            obs_size=actual_obs_size,
            act_size=act_size,
            num_agents=env.num_agents,
            max_act_size=max_act_size,  # 添加这个参数
            lr=ACTOR_LR,
            critic_lr=CRITIC_LR,
            gamma=GAMMA,
            tau=TAU
        )
        agents.append(agent)
    
    # 3. 创建训练器
    print(f"\n🎯 Step 3: 创建训练器")
    trainer = Trainer(
        envs=env,
        agents=agents,
        obs_size=None,  # 不使用
        act_size=None,  # 不使用
        buffer_capacity=BUFFER_CAPACITY,
        lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
        gamma=GAMMA,
        tau=TAU
    )
    print(f"   ✓ 训练器创建成功")
    print(f"   - Buffer容量: {int(BUFFER_CAPACITY)}")
    
    # 4. 训练循环
    print(f"\n🏋️  Step 4: 开始训练")
    print("=" * 70)
    
    episode_rewards_history = []
    
    for episode in range(TOTAL_EPISODES):
        # Rollout
        trainer.rollout(ROLLOUT_STEPS)
        
        # 训练（如果buffer足够）
        if len(trainer.replay_buffer) >= BATCH_SIZE:
            trainer.train_agents(rollout_steps=0, batch_size=BATCH_SIZE)
        
        # 评估当前episode的表现（简化版）
        # 从buffer中获取最近的数据
        if len(trainer.replay_buffer) > 0:
            # 简单统计：计算最近的平均奖励
            recent_samples = min(ROLLOUT_STEPS * NUM_ENVS, len(trainer.replay_buffer))
            
            # 这里我们做一个简化：每个episode结束后重置环境并测试一次
            obs = env.reset()
            episode_reward = np.zeros(env.num_agents)
            step = 0
            
            for _ in range(MAX_STEPS):
                # 收集动作
                acts_list = []
                for i, agent in enumerate(agents):
                    # obs shape: [num_envs, num_agents, obs_size]
                    # 我们取第一个环境
                    act = agent.predict(torch.FloatTensor(obs[0, i:i+1]))
                    acts_list.append(act)
                
                # 构造动作 [num_envs, num_agents, max_act_size]
                max_act_size = 2
                acts = np.zeros((NUM_ENVS, env.num_agents, max_act_size))
                for i, act in enumerate(acts_list):
                    act_size = env.act_sizes[env.agent_names[i]]
                    acts[0, i, :act_size] = act[0, :act_size]
                
                # Step
                next_obs, rewards, dones = env.step(acts)
                episode_reward += rewards[0]  # 第一个环境的奖励
                obs = next_obs
                step += 1
                
                if dones[0].all():
                    break
            
            episode_rewards_history.append(episode_reward)
        
        # 日志
        if (episode + 1) % LOG_INTERVAL == 0:
            if len(episode_rewards_history) > 0:
                recent_rewards = episode_rewards_history[-LOG_INTERVAL:]
                avg_total_reward = np.mean([r.sum() for r in recent_rewards])
                avg_agent_rewards = np.mean(recent_rewards, axis=0)
                
                print(f"\nEpisode {episode+1}/{TOTAL_EPISODES}")
                print(f"  平均总奖励: {avg_total_reward:8.2f}")
                print(f"  各Agent平均奖励:")
                for i, name in enumerate(env.agent_names):
                    print(f"    {name:10s}: {avg_agent_rewards[i]:7.3f}")
                print(f"  Buffer大小: {len(trainer.replay_buffer)}")
        
        # 保存
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_path = f"{SAVE_DIR}/checkpoint_ep{episode+1}.pt"
            torch.save({
                'episode': episode + 1,
                'agents': [{'actor': agent.actor.state_dict(),
                           'critic': agent.critic.state_dict()} 
                          for agent in agents],
            }, save_path)
            print(f"  💾 保存检查点: {save_path}")
    
    # 5. 训练完成
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    
    # 保存最终模型
    final_path = f"{SAVE_DIR}/final_model.pt"
    torch.save({
        'episode': TOTAL_EPISODES,
        'agents': [{'actor': agent.actor.state_dict(),
                   'critic': agent.critic.state_dict()} 
                  for agent in agents],
        'rewards_history': [r.tolist() for r in episode_rewards_history],
    }, final_path)
    print(f"\n💾 最终模型已保存: {final_path}")
    
    # 打印统计
    if len(episode_rewards_history) > 0:
        print(f"\n📊 训练统计:")
        print(f"   - 总Episodes: {TOTAL_EPISODES}")
        print(f"   - 最终平均奖励: {np.mean([r.sum() for r in episode_rewards_history[-10:]]):8.2f}")
        print(f"   - 各Agent最终表现:")
        final_avg = np.mean(episode_rewards_history[-10:], axis=0)
        for i, name in enumerate(env.agent_names):
            print(f"     {name:10s}: {final_avg[i]:7.3f}")

if __name__ == "__main__":
    main()
