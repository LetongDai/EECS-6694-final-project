#!/usr/bin/env python3
"""
MADDPG Training Script for Microgrid Energy Management
完整的训练脚本，包含日志、保存模型、可视化等功能
"""

import torch
import numpy as np
import os
import json
from datetime import datetime
from pathlib import Path

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer

# ============================================================================
# 配置
# ============================================================================

class Config:
    """训练配置"""
    # 环境配置
    num_envs = 4              # 并行环境数
    max_steps = 24            # 每个episode的最大步数（24小时）
    
    # Agent配置
    num_agents = 5            # agents数量
    
    # 训练配置
    total_episodes = 1000     # 总训练episodes
    rollout_steps = 100       # 每次rollout的步数
    batch_size = 128          # 训练batch大小
    
    # 学习率
    actor_lr = 1e-4           # Actor学习率
    critic_lr = 1e-3          # Critic学习率
    
    # RL参数
    gamma = 0.95              # 折扣因子
    tau = 0.01                # Soft update参数
    
    # Buffer配置
    buffer_capacity = 1e5     # Replay buffer容量
    
    # 日志和保存
    log_interval = 10         # 每N个episodes记录一次
    save_interval = 50        # 每N个episodes保存一次模型
    eval_interval = 20        # 每N个episodes评估一次
    
    # 路径
    save_dir = "checkpoints"
    log_dir = "logs"
    
    def __init__(self):
        # 创建目录
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        
        # 保存配置
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"maddpg_{self.timestamp}"

# ============================================================================
# 训练器增强版
# ============================================================================

class EnhancedTrainer(Trainer):
    """增强版训练器，添加日志和评估功能"""
    
    def __init__(self, envs, agents, config):
        super().__init__(
            envs=envs,
            agents=agents,
            obs_size=None,  # 不再需要
            act_size=None,  # 不再需要
            buffer_capacity=config.buffer_capacity,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        self.config = config
        
        # 统计信息
        self.episode_rewards = []
        self.episode_steps = []
        
    def rollout_episode(self):
        """
        执行一个完整的episode
        
        Returns:
            episode_reward: 每个agent的累计奖励
            episode_steps: episode长度
        """
        obs = self.envs.reset()
        episode_reward = np.zeros((self.envs.num_envs, self.envs.num_agents))
        step = 0
        done = False
        
        while not done and step < self.config.max_steps:
            # 获取动作
            # acts shape: [num_agents, num_envs, act_size]
            acts_list = []
            for i, agent in enumerate(self.agents):
                # obs[:, i] shape: [num_envs, obs_size]
                act = agent.predict(torch.FloatTensor(obs[:, i]))
                acts_list.append(act)
            
            # 转换为 [num_envs, num_agents, act_size]
            acts = np.stack(acts_list, axis=1)
            
            # 环境step
            next_obs, rewards, dones = self.envs.step(acts)
            
            # 存储经验
            self.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            # 累计奖励
            episode_reward += rewards
            
            # 更新状态
            obs = next_obs
            step += 1
            
            # 检查是否结束
            done = dones.all()
        
        return episode_reward.mean(axis=0), step
    
    def train_step(self, batch_size):
        """
        执行一次训练步骤
        
        Returns:
            losses: 每个agent的loss
        """
        if len(self.replay_buffer) < batch_size:
            return None
        
        # 采样batch
        obs, acts, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
        
        # 计算下一个动作
        next_acts = torch.stack([
            agent.target_actor(next_obs[:, i]).detach() 
            for i, agent in enumerate(self.agents)
        ], dim=1)
        
        # 准备sample
        sample = (obs, acts, rewards, next_obs, dones, next_acts)
        
        # 训练每个agent
        losses = []
        for i, agent in enumerate(self.agents):
            loss = agent.train_on(sample, i)
            losses.append(loss)
        
        return losses
    
    @torch.no_grad()
    def evaluate(self, num_episodes=5):
        """
        评估当前策略
        
        Returns:
            avg_rewards: 平均奖励
            avg_steps: 平均步数
        """
        eval_rewards = []
        eval_steps = []
        
        for _ in range(num_episodes):
            obs = self.envs.reset()
            episode_reward = np.zeros((self.envs.num_envs, self.envs.num_agents))
            step = 0
            done = False
            
            while not done and step < self.config.max_steps:
                # 使用确定性策略（无探索噪声）
                acts_list = []
                for i, agent in enumerate(self.agents):
                    act = agent.predict(torch.FloatTensor(obs[:, i]))
                    acts_list.append(act)
                
                acts = np.stack(acts_list, axis=1)
                next_obs, rewards, dones = self.envs.step(acts)
                
                episode_reward += rewards
                obs = next_obs
                step += 1
                done = dones.all()
            
            eval_rewards.append(episode_reward.mean(axis=0))
            eval_steps.append(step)
        
        avg_rewards = np.mean(eval_rewards, axis=0)
        avg_steps = np.mean(eval_steps)
        
        return avg_rewards, avg_steps

# ============================================================================
# 主训练函数
# ============================================================================

def main():
    """主训练函数"""
    
    print("=" * 70)
    print("🚀 MADDPG Microgrid Training")
    print("=" * 70)
    
    # 1. 创建配置
    config = Config()
    print(f"\n📋 配置:")
    print(f"   - 实验名称: {config.exp_name}")
    print(f"   - 并行环境数: {config.num_envs}")
    print(f"   - Agents数量: {config.num_agents}")
    print(f"   - 总Episodes: {config.total_episodes}")
    print(f"   - Batch大小: {config.batch_size}")
    print(f"   - Actor LR: {config.actor_lr}")
    print(f"   - Critic LR: {config.critic_lr}")
    
    # 2. 创建环境
    print(f"\n🌍 创建环境...")
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    print(f"   ✓ 环境创建成功")
    print(f"   - Agent名称: {env.agent_names}")
    print(f"   - 观察空间: {env.observation_space}")
    print(f"   - 动作空间: {env.action_space}")
    
    # 3. 创建Agents
    print(f"\n🤖 创建Agents...")
    🤖 创建Agents...
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
            max_act_size=max_act_size,  
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        agents.append(agent)
        print(f"   ✓ {agent_name:10s}: obs={actual_obs_size}, act={act_size}")
    
    # 4. 创建训练器
    print(f"\n🎯 创建训练器...")
    trainer = EnhancedTrainer(env, agents, config)
    print(f"   ✓ 训练器创建成功")
    print(f"   - Buffer容量: {int(config.buffer_capacity)}")
    
    # 5. 训练循环
    print(f"\n🏋️ 开始训练...")
    print("=" * 70)
    
    best_reward = -float('inf')
    
    for episode in range(config.total_episodes):
        # Rollout一个episode
        episode_rewards, episode_steps = trainer.rollout_episode()
        
        # 训练agents
        for _ in range(episode_steps):
            losses = trainer.train_step(config.batch_size)
        
        # 记录
        trainer.episode_rewards.append(episode_rewards)
        trainer.episode_steps.append(episode_steps)
        
        # 日志
        if (episode + 1) % config.log_interval == 0:
            avg_reward = np.mean([r.sum() for r in trainer.episode_rewards[-config.log_interval:]])
            avg_steps = np.mean(trainer.episode_steps[-config.log_interval:])
            
            print(f"Episode {episode+1}/{config.total_episodes}")
            print(f"  总奖励: {avg_reward:8.2f} | 步数: {avg_steps:5.1f}")
            print(f"  各Agent奖励: {episode_rewards}")
        
        # 评估
        if (episode + 1) % config.eval_interval == 0:
            eval_rewards, eval_steps = trainer.evaluate(num_episodes=5)
            eval_total = eval_rewards.sum()
            
            print(f"\n📊 评估结果 (Episode {episode+1}):")
            print(f"  总奖励: {eval_total:8.2f} | 步数: {eval_steps:5.1f}")
            print(f"  各Agent: {eval_rewards}")
            
            # 保存最佳模型
            if eval_total > best_reward:
                best_reward = eval_total
                save_path = f"{config.save_dir}/{config.exp_name}_best.pt"
                torch.save({
                    'episode': episode,
                    'agents': [agent.actor.state_dict() for agent in agents],
                    'rewards': eval_rewards,
                }, save_path)
                print(f"  ✓ 保存最佳模型: {save_path}")
            print()
        
        # 定期保存
        if (episode + 1) % config.save_interval == 0:
            save_path = f"{config.save_dir}/{config.exp_name}_ep{episode+1}.pt"
            torch.save({
                'episode': episode,
                'agents': [agent.actor.state_dict() for agent in agents],
                'config': vars(config),
            }, save_path)
            print(f"  💾 保存检查点: {save_path}\n")
    
    # 6. 训练完成
    print("=" * 70)
    print("🎉 训练完成！")
    print("=" * 70)
    
    # 最终评估
    print(f"\n📊 最终评估...")
    final_rewards, final_steps = trainer.evaluate(num_episodes=10)
    print(f"  总奖励: {final_rewards.sum():8.2f}")
    print(f"  各Agent奖励: {final_rewards}")
    print(f"  平均步数: {final_steps:5.1f}")
    
    # 保存最终模型
    final_path = f"{config.save_dir}/{config.exp_name}_final.pt"
    torch.save({
        'episode': config.total_episodes,
        'agents': [agent.actor.state_dict() for agent in agents],
        'rewards': final_rewards,
        'config': vars(config),
    }, final_path)
    print(f"\n💾 保存最终模型: {final_path}")
    
    # 保存训练统计
    stats = {
        'episode_rewards': [r.tolist() for r in trainer.episode_rewards],
        'episode_steps': trainer.episode_steps,
        'final_rewards': final_rewards.tolist(),
        'best_reward': best_reward,
    }
    
    stats_path = f"{config.log_dir}/{config.exp_name}_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"💾 保存训练统计: {stats_path}")
    
    print(f"\n✅ 所有文件已保存到:")
    print(f"   - 模型: {config.save_dir}/")
    print(f"   - 日志: {config.log_dir}/")

if __name__ == "__main__":
    main()
