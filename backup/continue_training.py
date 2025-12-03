#!/usr/bin/env python3
"""
从保存的checkpoint继续训练 - 完全修复版
Continue Training from Saved Checkpoint - Fixed Version
"""

import torch
import numpy as np
from datetime import datetime
from pathlib import Path

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer


class ContinueTrainingConfig:
    """继续训练的配置"""
    
    def __init__(self, checkpoint_path):
        # 加载原始配置
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        original_config = checkpoint.get('config', {})
        
        # 从checkpoint继承配置
        for key, value in original_config.items():
            setattr(self, key, value)
        
        # 新的训练设置
        self.checkpoint_path = checkpoint_path
        self.original_episodes = checkpoint.get('episode', 0)
        
        # 可以调整的参数
        self.additional_episodes = 300  # 额外训练的episodes
        self.total_episodes = self.additional_episodes
        
        # Fine-tuning学习率（降低学习率）
        self.actor_lr = getattr(self, 'actor_lr', 1e-4) * 0.2  # 降低50%
        self.critic_lr = getattr(self, 'critic_lr', 1e-3) * 0.2
        
        # 降低探索噪声
        self.noise_scale = getattr(self, 'noise_scale', 0.1) * 0.5
        self.noise_decay = getattr(self, 'noise_decay', 0.9995)
        self.noise_min = getattr(self, 'noise_min', 0.01)
        
        # 其他训练参数（保持不变）
        self.batch_size = getattr(self, 'batch_size', 128)
        self.buffer_capacity = getattr(self, 'buffer_capacity', 50000)
        self.gamma = getattr(self, 'gamma', 0.95)
        self.tau = getattr(self, 'tau', 0.01)
        self.warmup_episodes = 0  # 不需要预热
        
        # 环境配置
        self.num_envs = getattr(self, 'num_envs', 4)
        self.max_steps = getattr(self, 'max_steps', 24)
        
        # 日志和保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_name = f"continue_{timestamp}"
        self.log_interval = 10
        self.eval_interval = 50
        self.save_interval = 100
        
        # 路径
        self.save_dir = "continue_checkpoints"
        self.log_dir = "continue_logs"
        
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)


class ContinueTrainer:
    """继续训练的Trainer"""
    
    def __init__(self, checkpoint_path, config=None):
        self.checkpoint_path = checkpoint_path
        
        # 加载checkpoint
        print("📥 加载Checkpoint...")
        self.checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ Checkpoint加载成功 (Episode {self.checkpoint.get('episode', 'Unknown')})")
        
        # 配置
        if config is None:
            config = ContinueTrainingConfig(checkpoint_path)
        self.config = config
        
        print(f"\n⚙️  训练配置:")
        print(f"  原始训练: {config.original_episodes} episodes")
        print(f"  继续训练: {config.additional_episodes} episodes")
        print(f"  新学习率: actor={config.actor_lr:.6f}, critic={config.critic_lr:.6f}")
        print(f"  新噪声: {config.noise_scale:.4f}")
        
        # 创建环境
        print(f"\n🌍 创建环境...")
        self.env = MicrogridEnv(
            num_envs=config.num_envs,
            max_steps=config.max_steps
        )
        print(f"✅ 环境创建成功")
        
        # 创建agents并加载权重
        print(f"\n🤖 创建Agents并加载权重...")
        self.agents = []
        for i, agent_name in enumerate(self.env.agent_names):
            agent = Agent(
                obs_size=5,
                act_size=self.env.act_sizes[agent_name],
                num_agents=self.env.num_agents,
                max_act_size=2,
                lr=config.actor_lr,
                critic_lr=config.critic_lr,
                gamma=config.gamma,
                tau=config.tau
            )
            
            # 🔥 只加载actor权重（checkpoint只保存了actor）
            if i < len(self.checkpoint['agents']):
                try:
                    agent.actor.load_state_dict(self.checkpoint['agents'][i])
                    agent.target_actor.load_state_dict(self.checkpoint['agents'][i])
                    print(f"  ✅ {agent_name} (加载了预训练actor)")
                except Exception as e:
                    print(f"  ⚠️  {agent_name} (加载失败，使用新权重): {e}")
            else:
                print(f"  ⚠️  {agent_name} (checkpoint中没有权重，使用新初始化)")
            
            # Critic使用新初始化的权重（因为checkpoint没保存critic）
            # 这样可以让模型适应新的学习率
            
            # 设置为训练模式
            agent.actor.train()
            agent.critic.train()
            
            self.agents.append(agent)
        
        # 创建trainer
        self.trainer = Trainer(
            envs=self.env,
            agents=self.agents,
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
        self.eval_rewards = []
        self.current_noise = config.noise_scale
        
        print(f"\n✅ 所有组件准备完成")
    
    def rollout_episode(self, add_noise=True):
        """执行一个episode"""
        obs = self.env.reset()
        episode_reward = np.zeros(self.env.num_agents)
        step = 0
        
        for _ in range(self.config.max_steps):
            # 收集动作
            acts_list = []
            for i, agent in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)
                
                # 添加探索噪声
                if add_noise:
                    noise = np.random.randn(*act.shape) * self.current_noise
                    act = np.clip(act + noise, 0, 1)
                
                acts_list.append(act)
            
            # 构造动作数组
            max_act_size = 2
            acts = np.zeros((self.config.num_envs, self.env.num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            # Step
            next_obs, rewards, dones = self.env.step(acts)
            self.trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            episode_reward += rewards[0]
            obs = next_obs
            step += 1
            
            if dones[0].all():
                break
        
        return episode_reward, step
    
    def train_step(self, batch_size):
        """训练一步"""
        if len(self.trainer.replay_buffer) < batch_size:
            return
        
        self.trainer.train_agents(rollout_steps=0, batch_size=batch_size)
        
        # 衰减噪声
        self.current_noise = max(
            self.config.noise_min,
            self.current_noise * self.config.noise_decay
        )
    
    @torch.no_grad()
    def evaluate(self, num_episodes=10):
        """评估当前策略"""
        eval_rewards = []
        
        for _ in range(num_episodes):
            episode_reward, _ = self.rollout_episode(add_noise=False)
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards, axis=0)
    
    def save_checkpoint(self, episode, eval_reward=None, is_best=False):
        """保存checkpoint"""
        
        suffix = 'best' if is_best else f'ep{episode:04d}'
        save_path = f"{self.config.save_dir}/{self.config.exp_name}_{suffix}.pt"
        
        checkpoint = {
            'episode': self.config.original_episodes + episode,
            'original_episode': self.config.original_episodes,
            'continue_episode': episode,
            'agents': [agent.actor.state_dict() for agent in self.agents],
            'config': vars(self.config),
            'original_checkpoint': self.checkpoint_path,
        }
        
        if eval_reward is not None:
            checkpoint['eval_reward'] = float(eval_reward)
        
        torch.save(checkpoint, save_path)
        print(f"💾 保存checkpoint: {save_path}")
    
    def train(self):
        """主训练循环"""
        
        print("\n" + "=" * 70)
        print("🚀 开始继续训练")
        print("=" * 70)
        
        best_reward = -float('inf')
        
        # 预热buffer（运行几个episodes不训练）
        if len(self.trainer.replay_buffer) < self.config.batch_size:
            print(f"\n🔥 预热Buffer...")
            while len(self.trainer.replay_buffer) < self.config.batch_size * 2:
                self.rollout_episode(add_noise=True)
                print(f"  Buffer: {len(self.trainer.replay_buffer)}/{self.config.batch_size * 2}")
        
        for episode in range(1, self.config.total_episodes + 1):
            # Rollout
            episode_rewards, episode_steps = self.rollout_episode(add_noise=True)
            
            # 训练
            for _ in range(episode_steps):
                self.train_step(self.config.batch_size)
            
            # 记录
            self.episode_rewards.append(episode_rewards)
            total_reward = episode_rewards.sum()
            
            # 日志
            if episode % self.config.log_interval == 0:
                recent_rewards = self.episode_rewards[-self.config.log_interval:]
                avg_total = np.mean([r.sum() for r in recent_rewards])
                avg_per_agent = np.mean(recent_rewards, axis=0)
                
                print(f"\n{'='*70}")
                print(f"Episode {episode}/{self.config.total_episodes} "
                      f"(Total: {self.config.original_episodes + episode})")
                print(f"{'='*70}")
                print(f"平均总奖励: {avg_total:8.2f} | 当前奖励: {total_reward:8.2f} | "
                      f"噪声: {self.current_noise:.4f}")
                print(f"各Agent平均奖励:")
                for i, name in enumerate(self.env.agent_names):
                    print(f"  {name:10s}: {avg_per_agent[i]:8.3f}")
                print(f"Buffer大小: {len(self.trainer.replay_buffer)}")
            
            # 评估
            if episode % self.config.eval_interval == 0:
                print(f"\n🎯 评估中...")
                eval_rewards = self.evaluate(num_episodes=10)
                eval_total = eval_rewards.sum()
                self.eval_rewards.append((episode, eval_total, eval_rewards.copy()))
                
                print(f"  评估奖励: {eval_total:8.2f}")
                for i, name in enumerate(self.env.agent_names):
                    print(f"    {name:10s}: {eval_rewards[i]:8.3f}")
                
                # 保存最佳模型
                if eval_total > best_reward:
                    best_reward = eval_total
                    self.save_checkpoint(episode, eval_total, is_best=True)
                    print(f"  ⭐ 新的最佳模型！奖励: {eval_total:.2f}")
            
            # 定期保存
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        # 最终评估
        print("\n" + "=" * 70)
        print("🎉 训练完成！")
        print("=" * 70)
        
        print(f"\n📊 最终评估 (20 episodes)...")
        final_rewards = self.evaluate(num_episodes=20)
        final_total = final_rewards.sum()
        
        print(f"  最终奖励: {final_total:8.2f}")
        for i, name in enumerate(self.env.agent_names):
            print(f"    {name:10s}: {final_rewards[i]:8.3f}")
        
        # 保存最终模型
        self.save_checkpoint(
            self.config.total_episodes,
            final_total
        )
        
        # 保存训练统计
        self.save_training_stats(final_total, final_rewards)
        
        print(f"\n✅ 所有文件已保存到:")
        print(f"   - Checkpoints: {self.config.save_dir}/")
        print(f"   - Logs: {self.config.log_dir}/")
        print(f"   - 最佳奖励: {best_reward:.2f}")
        print(f"   - 最终奖励: {final_total:.2f}")
    
    def save_training_stats(self, final_total, final_rewards):
        """保存训练统计"""
        import json
        
        stats = {
            'exp_name': self.config.exp_name,
            'original_checkpoint': self.checkpoint_path,
            'original_episodes': self.config.original_episodes,
            'additional_episodes': self.config.total_episodes,
            'config': {k: v for k, v in vars(self.config).items() 
                      if not k.startswith('_') and isinstance(v, (int, float, str, bool))},
            'episode_rewards': [r.tolist() for r in self.episode_rewards],
            'eval_rewards': [(ep, float(r), rpa.tolist()) for ep, r, rpa in self.eval_rewards],
            'final_total': float(final_total),
            'final_rewards_per_agent': final_rewards.tolist(),
        }
        
        stats_path = f"{self.config.log_dir}/{self.config.exp_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"💾 统计数据已保存: {stats_path}")


def main():
    """主函数"""
    
    import sys
    
    print("=" * 70)
    print("🔄 继续训练MADDPG模型")
    print("=" * 70)
    
    # 从命令行获取checkpoint路径
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
    else:
        # 默认使用最新的evaluated checkpoint
        checkpoint_path = "debug_checkpoints/debug_maddpg_20251124_045324_final_evaluated.pt"
        print(f"\n💡 使用默认checkpoint: {checkpoint_path}")
    
    # 检查文件是否存在
    if not Path(checkpoint_path).exists():
        print(f"\n❌ Checkpoint文件不存在: {checkpoint_path}")
        print(f"\n请指定正确的checkpoint路径:")
        print(f"   python continue_training_fixed.py <checkpoint_path>")
        return
    
    # 创建配置（可以在这里修改参数）
    config = ContinueTrainingConfig(checkpoint_path)
    
    # 🔧 可选：手动调整参数
    # config.additional_episodes = 500  # 训练更多episodes
    # config.actor_lr = 5e-5            # 更低的学习率
    # config.noise_scale = 0.02         # 更小的噪声
    
    print(f"\n📋 训练计划:")
    print(f"  原始训练: {config.original_episodes} episodes")
    print(f"  继续训练: {config.additional_episodes} episodes")
    print(f"  学习率: actor={config.actor_lr:.6f}, critic={config.critic_lr:.6f}")
    print(f"  探索噪声: {config.noise_scale:.4f}")
    
    # 确认
    response = input(f"\n是否继续? (y/n): ")
    if response.lower() != 'y':
        print("取消训练")
        return
    
    # 创建trainer并开始训练
    trainer = ContinueTrainer(checkpoint_path, config)
    trainer.train()


if __name__ == "__main__":
    main()