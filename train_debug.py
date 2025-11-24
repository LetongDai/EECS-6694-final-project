#!/usr/bin/env python3
"""
Debug Training Script with Extensive Logging and Visualization
带有详细日志和可视化的调试版训练脚本
"""

import torch
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
from pathlib import Path
from datetime import datetime
from collections import deque

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer


# ============================================================================
# 配置
# ============================================================================

class DebugConfig:
    # 实验名称
    exp_name = f"debug_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 环境配置
    num_envs = 2              # 先用少量环境调试
    max_steps = 24
    
    # 训练配置
    total_episodes = 200      # 先训练少量 episodes
    batch_size = 64
    warmup_episodes = 5
    
    # 学习率
    actor_lr = 1e-4
    critic_lr = 1e-3
    
    # RL参数
    gamma = 0.95
    tau = 0.01
    
    # Buffer
    buffer_capacity = 10000
    
    # 探索噪声
    noise_scale = 0.1
    noise_decay = 0.999
    noise_min = 0.01
    
    # 日志频率
    log_interval = 1          # 每个episode都打印
    plot_interval = 10        # 每10个episodes画图
    save_interval = 50
    
    # 路径
    save_dir = "debug_checkpoints"
    log_dir = "debug_logs"
    plot_dir = "debug_plots"
    
    def __init__(self):
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.plot_dir).mkdir(exist_ok=True)


# ============================================================================
# 日志记录器
# ============================================================================
class DetailedLogger:
    """详细的训练日志记录器"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
        
        # 训练指标
        self.episode_rewards = []          # 每个episode的总奖励
        self.episode_rewards_per_agent = []  # 每个agent的奖励
        self.episode_steps = []
        self.episode_losses = []           # 损失
        self.buffer_sizes = []
        self.noise_levels = []
        
        # 详细统计
        self.clearing_prices = []          # 拍卖清算价格
        self.allocated_power = []          # 分配的电力
        self.action_distributions = {name: [] for name in agent_names}
        self.reward_distributions = {name: [] for name in agent_names}
        
        # 滑动窗口统计
        self.recent_rewards = deque(maxlen=10)
        
        # 文件日志
        self.log_file = open(f"{config.log_dir}/{config.exp_name}.log", 'w')
        
    def log(self, message, print_console=True):
        """记录日志"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        self.log_file.write(log_msg + '\n')
        self.log_file.flush()
        if print_console:
            print(log_msg)
    
    def log_episode(self, episode, episode_data):
        """记录episode数据"""
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_rewards_per_agent.append(episode_data['rewards_per_agent'])
        self.episode_steps.append(episode_data['steps'])
        self.buffer_sizes.append(episode_data['buffer_size'])
        self.noise_levels.append(episode_data['noise_level'])
        
        if 'losses' in episode_data:
            self.episode_losses.append(episode_data['losses'])
        
        # 滑动窗口
        self.recent_rewards.append(episode_data['total_reward'])
        
        # 详细日志
        self.log(f"\n{'='*70}")
        self.log(f"Episode {episode}/{self.config.total_episodes}")
        self.log(f"{'='*70}")
        self.log(f"总奖励: {episode_data['total_reward']:8.2f} | 步数: {episode_data['steps']:2d}")
        self.log(f"最近10个episodes平均: {np.mean(self.recent_rewards):8.2f}")
        self.log(f"各Agent奖励:")
        for i, name in enumerate(self.agent_names):
            reward = episode_data['rewards_per_agent'][i]
            self.log(f"  {name:10s}: {reward:8.3f}")
        self.log(f"Buffer大小: {episode_data['buffer_size']:5d}")
        self.log(f"噪声水平: {episode_data['noise_level']:.4f}")
        
        # 动作统计
        if 'actions' in episode_data:
            self.log(f"\n动作统计:")
            for i, name in enumerate(self.agent_names):
                actions = episode_data['actions'][i]
                self.log(f"  {name:10s}: mean={np.mean(actions):6.3f}, "
                        f"std={np.std(actions):6.3f}, "
                        f"min={np.min(actions):6.3f}, "
                        f"max={np.max(actions):6.3f}")
        
        # 损失统计
        if 'losses' in episode_data and episode_data['losses']:
            self.log(f"\n损失统计:")
            for i, name in enumerate(self.agent_names):
                if i < len(episode_data['losses']):
                    loss = episode_data['losses'][i]
                    self.log(f"  {name:10s}: {loss:.6f}")
    
    def save_stats(self):
        """保存统计数据"""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_rewards_per_agent': [r.tolist() for r in self.episode_rewards_per_agent],
            'episode_steps': self.episode_steps,
            'buffer_sizes': self.buffer_sizes,
            'noise_levels': self.noise_levels,
            'episode_losses': self.episode_losses,
            'agent_names': self.agent_names,
        }
        
        stats_path = f"{self.config.log_dir}/{self.config.exp_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.log(f"\n💾 统计数据已保存: {stats_path}")
    
    def close(self):
        self.log_file.close()


# ============================================================================
# 可视化工具
# ============================================================================

class Plotter:
    """训练过程可视化"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    def plot_training_curves(self, logger, episode):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        
        # 1. 总奖励曲线
        ax = axes[0, 0]
        ax.plot(episodes, logger.episode_rewards, 'b-', alpha=0.3, label='Raw')
        if len(logger.episode_rewards) > 10:
            smoothed = self._smooth(logger.episode_rewards, window=10)
            ax.plot(episodes, smoothed, 'b-', linewidth=2, label='Smoothed (10)')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各Agent奖励曲线
        ax = axes[0, 1]
        rewards_per_agent = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            agent_rewards = rewards_per_agent[:, i]
            ax.plot(episodes, agent_rewards, alpha=0.6, 
                   color=self.colors[i], label=name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Rewards per Agent')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Episode步数
        ax = axes[0, 2]
        ax.plot(episodes, logger.episode_steps, 'g-', alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # 4. Buffer大小
        ax = axes[1, 0]
        ax.plot(episodes, logger.buffer_sizes, 'r-', alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Buffer Size')
        ax.set_title('Replay Buffer Size')
        ax.grid(True, alpha=0.3)
        
        # 5. 噪声水平
        ax = axes[1, 1]
        ax.plot(episodes, logger.noise_levels, 'purple', alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Noise Level')
        ax.set_title('Exploration Noise')
        ax.grid(True, alpha=0.3)
        
        # 6. 最近奖励分布
        ax = axes[1, 2]
        recent_episodes = min(50, len(logger.episode_rewards))
        if recent_episodes > 0:
            recent_rewards = logger.episode_rewards[-recent_episodes:]
            ax.hist(recent_rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.axvline(np.mean(recent_rewards), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(recent_rewards):.2f}')
            ax.set_xlabel('Total Reward')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Reward Distribution (Last {recent_episodes} Episodes)')
            ax.legend()
        
        plt.tight_layout()
        plot_path = f"{self.config.plot_dir}/training_ep{episode:04d}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def plot_agent_analysis(self, logger, episode):
        """绘制各Agent的详细分析"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Agent Analysis - Episode {episode}', fontsize=16)
        
        rewards_per_agent = np.array(logger.episode_rewards_per_agent)
        episodes = list(range(1, len(rewards_per_agent) + 1))
        
        # 1. 各Agent奖励趋势（平滑）
        ax = axes[0, 0]
        for i, name in enumerate(self.agent_names):
            agent_rewards = rewards_per_agent[:, i]
            if len(agent_rewards) > 10:
                smoothed = self._smooth(agent_rewards, window=10)
                ax.plot(episodes, smoothed, linewidth=2, 
                       color=self.colors[i], label=name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Smoothed Reward')
        ax.set_title('Agent Rewards (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 最终奖励对比
        ax = axes[0, 1]
        if len(rewards_per_agent) > 0:
            final_rewards = rewards_per_agent[-1, :]
            bars = ax.bar(self.agent_names, final_rewards, color=self.colors, alpha=0.7)
            ax.set_ylabel('Reward')
            ax.set_title('Latest Episode Rewards')
            ax.grid(True, alpha=0.3, axis='y')
            # 添加数值标签
            for bar, reward in zip(bars, final_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{reward:.2f}', ha='center', va='bottom' if reward > 0 else 'top')
        
        # 3. 平均奖励对比
        ax = axes[1, 0]
        if len(rewards_per_agent) > 0:
            avg_rewards = np.mean(rewards_per_agent, axis=0)
            bars = ax.bar(self.agent_names, avg_rewards, color=self.colors, alpha=0.7)
            ax.set_ylabel('Average Reward')
            ax.set_title('Average Rewards over All Episodes')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, reward in zip(bars, avg_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{reward:.2f}', ha='center', va='bottom' if reward > 0 else 'top')
        
        # 4. 奖励波动性（标准差）
        ax = axes[1, 1]
        if len(rewards_per_agent) > 1:
            std_rewards = np.std(rewards_per_agent, axis=0)
            bars = ax.bar(self.agent_names, std_rewards, color=self.colors, alpha=0.7)
            ax.set_ylabel('Standard Deviation')
            ax.set_title('Reward Variability (Std Dev)')
            ax.grid(True, alpha=0.3, axis='y')
            for bar, std in zip(bars, std_rewards):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{std:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plot_path = f"{self.config.plot_dir}/agents_ep{episode:04d}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _smooth(self, data, window=10):
        """平滑曲线"""
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed
    
    def create_summary_plot(self, logger):
        """创建最终总结图"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        rewards_per_agent = np.array(logger.episode_rewards_per_agent)
        
        # 大图：总奖励
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(episodes, logger.episode_rewards, 'b-', alpha=0.3, linewidth=1)
        if len(logger.episode_rewards) > 20:
            smoothed = self._smooth(logger.episode_rewards, window=20)
            ax_main.plot(episodes, smoothed, 'b-', linewidth=3, label='Smoothed (20)')
        ax_main.set_xlabel('Episode', fontsize=12)
        ax_main.set_ylabel('Total Reward', fontsize=12)
        ax_main.set_title('Training Progress: Total Reward', fontsize=14, fontweight='bold')
        ax_main.legend(fontsize=10)
        ax_main.grid(True, alpha=0.3)
        
        # 各Agent趋势
        ax1 = fig.add_subplot(gs[1, 0])
        for i, name in enumerate(self.agent_names):
            agent_rewards = rewards_per_agent[:, i]
            if len(agent_rewards) > 10:
                smoothed = self._smooth(agent_rewards, window=10)
                ax1.plot(episodes, smoothed, linewidth=2, color=self.colors[i], label=name)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('Agent Rewards (Smoothed)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 最终性能
        ax2 = fig.add_subplot(gs[1, 1])
        if len(rewards_per_agent) >= 10:
            final_avg = np.mean(rewards_per_agent[-10:, :], axis=0)
            bars = ax2.barh(self.agent_names, final_avg, color=self.colors, alpha=0.7)
            ax2.set_xlabel('Average Reward (Last 10 Eps)')
            ax2.set_title('Final Performance')
            ax2.grid(True, alpha=0.3, axis='x')
            for bar, reward in zip(bars, final_avg):
                width = bar.get_width()
                ax2.text(width, bar.get_y() + bar.get_height()/2.,
                        f'{reward:.2f}', ha='left' if reward > 0 else 'right', va='center')
        
        # 学习进度
        ax3 = fig.add_subplot(gs[1, 2])
        if len(episodes) > 20:
            window = len(episodes) // 5
            early = np.mean(logger.episode_rewards[:window])
            late = np.mean(logger.episode_rewards[-window:])
            improvement = ((late - early) / abs(early) * 100) if early != 0 else 0
            
            ax3.bar(['Early\n(First 20%)', 'Late\n(Last 20%)'], [early, late], 
                   color=['lightcoral', 'lightgreen'], alpha=0.7)
            ax3.set_ylabel('Average Total Reward')
            ax3.set_title(f'Learning Progress\n({improvement:+.1f}% change)')
            ax3.grid(True, alpha=0.3, axis='y')
        
        # Buffer和噪声
        ax4 = fig.add_subplot(gs[2, 0])
        ax4_twin = ax4.twinx()
        ax4.plot(episodes, logger.buffer_sizes, 'r-', alpha=0.6, label='Buffer Size')
        ax4_twin.plot(episodes, logger.noise_levels, 'purple', alpha=0.6, label='Noise Level')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Buffer Size', color='r')
        ax4_twin.set_ylabel('Noise Level', color='purple')
        ax4.set_title('Buffer & Exploration')
        ax4.grid(True, alpha=0.3)
        
        # 奖励分布
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(logger.episode_rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax5.axvline(np.mean(logger.episode_rewards), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(logger.episode_rewards):.2f}')
        ax5.set_xlabel('Total Reward')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Reward Distribution')
        ax5.legend()
        
        # 统计摘要
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        stats_text = f"""
        Training Summary
        {'='*30}
        Total Episodes: {len(episodes)}
        
        Final Avg Reward: {np.mean(logger.episode_rewards[-10:]):.2f}
        Best Reward: {np.max(logger.episode_rewards):.2f}
        Worst Reward: {np.min(logger.episode_rewards):.2f}
        
        Avg Steps/Episode: {np.mean(logger.episode_steps):.1f}
        Final Buffer Size: {logger.buffer_sizes[-1]}
        Final Noise: {logger.noise_levels[-1]:.4f}
        """
        ax6.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.suptitle(f'Training Summary: {logger.config.exp_name}', 
                    fontsize=16, fontweight='bold', y=0.995)
        
        plot_path = f"{self.config.plot_dir}/summary_final.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path


# ============================================================================
# 调试训练器
# ============================================================================

class DebugTrainer:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config
        
        # 基础trainer
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
        
        # 日志和可视化
        self.logger = DetailedLogger(config, env.agent_names)
        self.plotter = Plotter(config, env.agent_names)
        
        # 噪声
        self.current_noise = config.noise_scale
    
    def rollout_episode(self, add_noise=True):
        """执行一个episode并收集详细信息"""
        obs = self.env.reset()
        episode_reward = np.zeros(self.env.num_agents)
        step = 0
        
        # 收集动作统计
        actions_collected = [[] for _ in range(self.env.num_agents)]
        
        for _ in range(self.config.max_steps):
            acts_list = []
            for i, agent in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)
                
                # 添加噪声
                if add_noise:
                    noise = np.random.randn(*act.shape) * self.current_noise
                    act = np.clip(act + noise, -1, 1)
                
                acts_list.append(act)
                actions_collected[i].append(act[0])  # 第一个环境的动作
            
            # 构造动作
            max_act_size = max(act.shape[1] for act in acts_list)
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
        
        # 衰减噪声
        self.current_noise = max(
            self.config.noise_min,
            self.current_noise * self.config.noise_decay
        )
        
        # 转换动作为numpy数组
        actions_stats = [np.array(acts) for acts in actions_collected]
        
        return episode_reward, step, actions_stats
    
    def train_step(self, batch_size):
        """训练一步"""
        if len(self.trainer.replay_buffer) < batch_size:
            return None
        
        self.trainer.train_agents(rollout_steps=0, batch_size=batch_size)
    
    def train(self):
        """主训练循环"""
        self.logger.log("\n" + "="*70)
        self.logger.log("🚀 开始调试训练")
        self.logger.log("="*70)
        
        # 预热
        self.logger.log(f"\n🔥 预热阶段 ({self.config.warmup_episodes} episodes)")
        for ep in range(self.config.warmup_episodes):
            self.rollout_episode(add_noise=True)
            self.logger.log(f"  预热 {ep+1}/{self.config.warmup_episodes} - "
                          f"Buffer: {len(self.trainer.replay_buffer)}")
        
        # 训练循环
        self.logger.log(f"\n🏋️  训练阶段")
        self.logger.log("="*70)
        
        for episode in range(1, self.config.total_episodes + 1):
            # Rollout
            episode_rewards, episode_steps, actions_stats = self.rollout_episode(add_noise=True)
            
            # 训练
            for _ in range(episode_steps):
                self.train_step(self.config.batch_size)
            
            # 记录数据
            episode_data = {
                'total_reward': episode_rewards.sum(),
                'rewards_per_agent': episode_rewards,
                'steps': episode_steps,
                'buffer_size': len(self.trainer.replay_buffer),
                'noise_level': self.current_noise,
                'actions': actions_stats,
            }
            
            self.logger.log_episode(episode, episode_data)
            
            # 绘图
            if episode % self.config.plot_interval == 0:
                self.logger.log(f"\n📊 生成可视化图表...")
                plot1 = self.plotter.plot_training_curves(self.logger, episode)
                plot2 = self.plotter.plot_agent_analysis(self.logger, episode)
                self.logger.log(f"  ✓ 训练曲线: {plot1}")
                self.logger.log(f"  ✓ Agent分析: {plot2}")
            
            # 保存检查点
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        # 最终总结
        self.logger.log("\n" + "="*70)
        self.logger.log("🎉 训练完成！")
        self.logger.log("="*70)
        
        # 保存统计和最终图表
        self.logger.save_stats()
        summary_plot = self.plotter.create_summary_plot(self.logger)
        self.logger.log(f"📊 总结图表: {summary_plot}")
        
        # 保存最终模型
        self.save_checkpoint(self.config.total_episodes, is_final=True)
        
        self.logger.close()
    
    def save_checkpoint(self, episode, is_final=False):
        """保存检查点"""
        suffix = 'final' if is_final else f'ep{episode:04d}'
        save_path = f"{self.config.save_dir}/{self.config.exp_name}_{suffix}.pt"
        
        torch.save({
            'episode': episode,
            'agents': [agent.actor.state_dict() for agent in self.agents],
            'config': vars(self.config),
        }, save_path)
        
        self.logger.log(f"💾 保存检查点: {save_path}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    config = DebugConfig()
    
    print("="*70)
    print("🔍 MADDPG 调试训练")
    print("="*70)
    print(f"实验名称: {config.exp_name}")
    print(f"日志目录: {config.log_dir}/")
    print(f"图表目录: {config.plot_dir}/")
    print("="*70)
    
    # 创建环境
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    print(f"\n✓ 环境创建成功")
    print(f"  Agents: {env.agent_names}")
    
    # 创建Agents
    agents = []
    max_act_size = 2
    for i, agent_name in enumerate(env.agent_names):
        agent = Agent(
            obs_size=5,
            act_size=env.act_sizes[agent_name],
            num_agents=env.num_agents,
            max_act_size=max_act_size,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        agents.append(agent)
    
    print(f"✓ Agents创建成功")
    
    # 创建调试训练器
    debug_trainer = DebugTrainer(env, agents, config)
    
    # 开始训练
    debug_trainer.train()
    
    print(f"\n✅ 训练完成！")
    print(f"   - 日志: {config.log_dir}/{config.exp_name}.log")
    print(f"   - 统计: {config.log_dir}/{config.exp_name}_stats.json")
    print(f"   - 图表: {config.plot_dir}/")


if __name__ == "__main__":
    main()
