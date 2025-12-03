#!/usr/bin/env python3
"""
Enhanced MADDPG Training Script
完整版训练脚本 - 包含评估、噪声、模型保存等功能
"""

import torch
import numpy as np
import os
import json
from pathlib import Path
from datetime import datetime

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer


# ============================================================================
# 配置类
# ============================================================================

class Config:
    # 实验名称
    exp_name = f"maddpg_microgrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 环境配置
    num_envs = 4              # 并行环境数
    max_steps = 24            # 每个episode的最大步数
    
    # 训练配置
    total_episodes = 1000     # 总训练episodes
    batch_size = 128          # 训练batch大小
    warmup_episodes = 10      # 预热episodes（只收集数据）
    
    # 学习率
    actor_lr = 1e-4
    critic_lr = 1e-3
    
    # RL参数
    gamma = 0.95
    tau = 0.01
    
    # Buffer
    buffer_capacity = 50000
    
    # 探索噪声
    noise_scale = 0.1         # 探索噪声标准差
    noise_decay = 0.9995      # 噪声衰减率
    noise_min = 0.01          # 最小噪声
    
    # 日志和保存
    log_interval = 10         # 每N个episodes打印日志
    eval_interval = 50        # 每N个episodes评估
    save_interval = 100       # 每N个episodes保存模型
    
    # 路径
    save_dir = "checkpoints"
    log_dir = "logs"
    
    def __init__(self):
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)


# ============================================================================
# 增强版训练器
# ============================================================================

class EnhancedTrainer:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config
        
        # 创建基础trainer
        self.trainer = Trainer(
            envs=env,
            agents=agents,
            obs_size=None,
            act_size=None,
            buffer_capacity=config.buffer_capacity,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        
        # 记录
        self.episode_rewards = []
        self.episode_steps = []
        self.eval_rewards = []
        
        # 当前噪声水平
        self.current_noise = config.noise_scale
        
    def rollout_episode(self, add_noise=True):
        """执行一个完整的episode"""
        obs = self.env.reset()
        episode_reward = np.zeros(self.env.num_agents)
        step = 0
        
        for _ in range(self.config.max_steps):
            # 收集动作（带探索噪声）
            acts_list = []
            for i, agent in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)  # [num_envs, act_size]
                
                # 添加探索噪声
                if add_noise:
                    noise = np.random.randn(*act.shape) * self.current_noise
                    act = np.clip(act + noise, 0, 1)
                
                acts_list.append(act)
            
            # 构造动作数组
            max_act_size = max(act.shape[1] for act in acts_list)
            acts = np.zeros((self.config.num_envs, self.env.num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            # Step环境
            next_obs, rewards, dones = self.env.step(acts)
            
            # 存储到buffer
            self.trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            # 累积奖励（取第一个环境）
            episode_reward += rewards[0]
            obs = next_obs
            step += 1
            
            if dones[0].all():
                break
        
        return episode_reward, step
    
    def train_step(self, batch_size):
        """执行一次训练步骤"""
        if len(self.trainer.replay_buffer) < batch_size:
            return None
        
        # 不执行rollout，只训练
        self.trainer.train_agents(rollout_steps=0, batch_size=batch_size)
        
        # 衰减噪声
        self.current_noise = max(
            self.config.noise_min,
            self.current_noise * self.config.noise_decay
        )
    
    @torch.no_grad()
    def evaluate(self, num_episodes=5):
        """评估当前策略（不添加噪声）"""
        eval_rewards = []
        eval_steps = []
        
        for _ in range(num_episodes):
            episode_reward, step = self.rollout_episode(add_noise=False)
            eval_rewards.append(episode_reward)
            eval_steps.append(step)
        
        return np.mean(eval_rewards, axis=0), np.mean(eval_steps)


# ============================================================================
# 主函数
# ============================================================================

def main():
    config = Config()
    
    print("=" * 70)
    print("🚀 MADDPG Microgrid Training - Enhanced Version")
    print("=" * 70)
    print(f"📝 实验名称: {config.exp_name}")
    print(f"📅 开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 创建环境
    print(f"\n🌍 Step 1: 创建环境")
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    print(f"   ✓ 环境创建成功")
    print(f"   - 并行环境数: {config.num_envs}")
    print(f"   - Agents: {env.agent_names}")
    
    # 2. 创建Agents
    print(f"\n🤖 Step 2: 创建Agents")
    agents = []
    max_act_size = 2  # diesel和battery的最大动作维度
    
    for i, agent_name in enumerate(env.agent_names):
        obs_size = env.obs_sizes[agent_name]
        act_size = env.act_sizes[agent_name]
        actual_obs_size = 5  # 统一观察维度
        
        agent = Agent(
            obs_size=actual_obs_size,
            act_size=act_size,
            num_agents=env.num_agents,
            max_act_size=max_act_size,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        agents.append(agent)
        print(f"   ✓ {agent_name:10s}: obs={actual_obs_size}, act={act_size}")
    
    # 3. 创建增强版训练器
    print(f"\n🎯 Step 3: 创建增强版训练器")
    trainer = EnhancedTrainer(env, agents, config)
    print(f"   ✓ 训练器创建成功")
    print(f"   - Buffer容量: {config.buffer_capacity}")
    print(f"   - 初始噪声: {config.noise_scale}")
    print(f"   - 噪声衰减: {config.noise_decay}")
    
    # 4. 预热阶段
    print(f"\n🔥 Step 4: 预热阶段 (收集初始数据)")
    for episode in range(config.warmup_episodes):
        trainer.rollout_episode(add_noise=True)
        if (episode + 1) % 5 == 0:
            print(f"   预热 {episode+1}/{config.warmup_episodes} - Buffer大小: {len(trainer.trainer.replay_buffer)}")
    
    # 5. 训练循环
    print(f"\n🏋️  Step 5: 开始训练")
    print("=" * 70)
    
    best_reward = -float('inf')
    
    for episode in range(config.total_episodes):
        # Rollout一个episode
        episode_rewards, episode_steps = trainer.rollout_episode(add_noise=True)
        
        # 训练多个steps
        for _ in range(episode_steps):
            trainer.train_step(config.batch_size)
        
        # 记录
        trainer.episode_rewards.append(episode_rewards)
        trainer.episode_steps.append(episode_steps)
        
        # 日志
        if (episode + 1) % config.log_interval == 0:
            recent_rewards = trainer.episode_rewards[-config.log_interval:]
            avg_total = np.mean([r.sum() for r in recent_rewards])
            avg_steps = np.mean(trainer.episode_steps[-config.log_interval:])
            avg_per_agent = np.mean(recent_rewards, axis=0)
            
            print(f"\n📊 Episode {episode+1}/{config.total_episodes}")
            print(f"  平均总奖励: {avg_total:8.2f} | 步数: {avg_steps:5.1f} | 噪声: {trainer.current_noise:.4f}")
            print(f"  各Agent平均奖励:")
            for i, name in enumerate(env.agent_names):
                print(f"    {name:10s}: {avg_per_agent[i]:8.3f}")
            print(f"  Buffer大小: {len(trainer.trainer.replay_buffer)}")
        
        # 评估
        if (episode + 1) % config.eval_interval == 0:
            print(f"\n🎯 评估中...")
            eval_rewards, eval_steps = trainer.evaluate(num_episodes=10)
            eval_total = eval_rewards.sum()
            trainer.eval_rewards.append((episode+1, eval_total, eval_rewards.copy()))
            
            print(f"  ✓ 评估完成 (10 episodes)")
            print(f"  总奖励: {eval_total:8.2f} | 步数: {eval_steps:5.1f}")
            print(f"  各Agent奖励:")
            for i, name in enumerate(env.agent_names):
                print(f"    {name:10s}: {eval_rewards[i]:8.3f}")
            
            # 保存最佳模型
            if eval_total > best_reward:
                best_reward = eval_total
                save_path = f"{config.save_dir}/{config.exp_name}_best.pt"
                torch.save({
                    'episode': episode + 1,
                    'eval_reward': eval_total,
                    'eval_rewards_per_agent': eval_rewards,
                    'agents': [agent.actor.state_dict() for agent in agents],
                    'config': vars(config),
                }, save_path)
                print(f"  ⭐ 保存最佳模型 (奖励: {eval_total:.2f})")
        
        # 定期保存检查点
        if (episode + 1) % config.save_interval == 0:
            save_path = f"{config.save_dir}/{config.exp_name}_ep{episode+1}.pt"
            torch.save({
                'episode': episode + 1,
                'agents': [agent.actor.state_dict() for agent in agents],
                'config': vars(config),
            }, save_path)
            print(f"  💾 保存检查点: ep{episode+1}")
    
    # 6. 最终评估
    print("\n" + "=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    
    print(f"\n📊 最终评估 (20 episodes)...")
    final_rewards, final_steps = trainer.evaluate(num_episodes=20)
    final_total = final_rewards.sum()
    
    print(f"  总奖励: {final_total:8.2f}")
    print(f"  平均步数: {final_steps:5.1f}")
    print(f"  各Agent奖励:")
    for i, name in enumerate(env.agent_names):
        print(f"    {name:10s}: {final_rewards[i]:8.3f}")
    
    # 保存最终模型
    final_path = f"{config.save_dir}/{config.exp_name}_final.pt"
    torch.save({
        'episode': config.total_episodes,
        'final_reward': final_total,
        'final_rewards_per_agent': final_rewards,
        'agents': [agent.actor.state_dict() for agent in agents],
        'config': vars(config),
    }, final_path)
    print(f"\n💾 保存最终模型: {final_path}")
    
    # 保存训练统计
    stats = {
        'exp_name': config.exp_name,
        'config': vars(config),
        'episode_rewards': [r.tolist() for r in trainer.episode_rewards],
        'episode_steps': trainer.episode_steps,
        'eval_rewards': [(ep, r, rpa.tolist()) for ep, r, rpa in trainer.eval_rewards],
        'final_reward': final_total,
        'final_rewards_per_agent': final_rewards.tolist(),
        'best_reward': best_reward,
    }
    
    stats_path = f"{config.log_dir}/{config.exp_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 保存训练统计: {stats_path}")
    
    print(f"\n✅ 所有文件已保存:")
    print(f"   - 模型: {config.save_dir}/")
    print(f"   - 日志: {config.log_dir}/")
    print(f"   - 最佳奖励: {best_reward:.2f}")
    print(f"   - 最终奖励: {final_total:.2f}")
    
    print(f"\n⏰ 结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
