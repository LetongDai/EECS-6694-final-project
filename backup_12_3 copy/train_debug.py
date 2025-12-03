#!/usr/bin/env python3
"""
Debug Training Script with Main Grid Trading Records
"""

import torch
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path
from datetime import datetime
from collections import deque

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer


class DebugConfig:
    exp_name = f"debug_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    num_envs = 2
    max_steps = 24
    
    total_episodes = 200
    batch_size = 128
    warmup_episodes = 5
    
    actor_lr = 5e-5
    critic_lr = 5e-4
    
    gamma = 0.95
    tau = 0.01
    
    buffer_capacity = 10000
    
    noise_scale = 0.1
    noise_decay = 0.999
    noise_min = 0.01
    
    log_interval = 1
    plot_interval = 10
    save_interval = 50
    
    save_dir = "debug_checkpoints"
    log_dir = "debug_logs"
    plot_dir = "debug_plots"
   
   
    warmup_episodes = 10
    def __init__(self):
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.plot_dir).mkdir(exist_ok=True)


class DetailedLogger:
    """增强版日志记录器 with Main Grid tracking"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        self.num_agents = len(agent_names)
        
        self.episode_rewards = []
        self.episode_rewards_per_agent = []
        self.episode_steps = []
        self.episode_losses = []
        self.buffer_sizes = []
        self.noise_levels = []
        
        # Main Grid 交易记录
        self.grid_imports = []      # 每个episode从grid购买的总电量
        self.grid_exports = []      # 每个episode卖给grid的总电量
        self.grid_net_trades = []   # 净交易量 (正=进口，负=出口)
        self.clearing_prices = []   # 市场出清价格
        
        self.recent_rewards = deque(maxlen=10)
        
        self.log_file = open(f"{config.log_dir}/{config.exp_name}.log", 'w')
        
    def log(self, message, print_console=True):
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_msg = f"[{timestamp}] {message}"
        self.log_file.write(log_msg + '\n')
        self.log_file.flush()
        if print_console:
            print(log_msg)
    
    def log_episode(self, episode, episode_data):
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_rewards_per_agent.append(episode_data['rewards_per_agent'])
        self.episode_steps.append(episode_data['steps'])
        self.buffer_sizes.append(episode_data['buffer_size'])
        self.noise_levels.append(episode_data['noise_level'])
        
        # 记录 Main Grid 数据
        self.grid_imports.append(episode_data.get('grid_import', 0))
        self.grid_exports.append(episode_data.get('grid_export', 0))
        self.grid_net_trades.append(episode_data.get('grid_net_trade', 0))
        self.clearing_prices.append(episode_data.get('avg_clearing_price', 0))
        
        if 'losses' in episode_data:
            self.episode_losses.append(episode_data['losses'])
        
        self.recent_rewards.append(episode_data['total_reward'])
        
        # 详细日志
        self.log(f"\n{'='*70}")
        self.log(f"Episode {episode}/{self.config.total_episodes}")
        self.log(f"{'='*70}")
        self.log(f"总奖励: {episode_data['total_reward']:8.2f} | 步数: {episode_data['steps']:2d}")
        self.log(f"最近10个episodes平均: {np.mean(self.recent_rewards):8.2f}")
        
        self.log(f"\n各Agent奖励:")
        for i, name in enumerate(self.agent_names):
            reward = episode_data['rewards_per_agent'][i]
            self.log(f"  {name:10s}: {reward:8.3f}")
        
        # Main Grid 交易信息
        self.log(f"\n🔌 Main Grid 交易:")
        self.log(f"  进口电量: {episode_data.get('grid_import', 0):8.2f} kWh")
        self.log(f"  出口电量: {episode_data.get('grid_export', 0):8.2f} kWh")
        self.log(f"  净交易量: {episode_data.get('grid_net_trade', 0):8.2f} kWh")
        self.log(f"  平均出清价: {episode_data.get('avg_clearing_price', 0):6.2f} cents/kWh")
        self.log(f"  市场独立度: {episode_data.get('market_independence', 0):6.2f}%")
        
        self.log(f"\nBuffer大小: {episode_data['buffer_size']:5d}")
        self.log(f"噪声水平: {episode_data['noise_level']:.4f}")
        
        if 'actions' in episode_data:
            self.log(f"\n动作统计:")
            for i, name in enumerate(self.agent_names):
                actions = episode_data['actions'][i]
                self.log(f"  {name:10s}: mean={np.mean(actions):6.3f}, "
                        f"std={np.std(actions):6.3f}")
    
    def save_stats(self):
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_rewards_per_agent': [r.tolist() for r in self.episode_rewards_per_agent],
            'episode_steps': self.episode_steps,
            'buffer_sizes': self.buffer_sizes,
            'noise_levels': self.noise_levels,
            'episode_losses': self.episode_losses,
            'agent_names': self.agent_names,
            
            # Main Grid 数据
            'grid_imports': self.grid_imports,
            'grid_exports': self.grid_exports,
            'grid_net_trades': self.grid_net_trades,
            'clearing_prices': self.clearing_prices,
        }
        
        stats_path = f"{self.config.log_dir}/{self.config.exp_name}_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.log(f"\n💾 统计数据已保存: {stats_path}")
    
    def close(self):
        self.log_file.close()


class Plotter:
    """增强版可视化 with Main Grid charts"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        self.colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
    
    def plot_training_curves(self, logger, episode):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        
        # 1. 总奖励
        ax = axes[0, 0]
        ax.plot(episodes, logger.episode_rewards, 'b-', alpha=0.3, label='Raw')
        if len(logger.episode_rewards) > 10:
            smoothed = self._smooth(logger.episode_rewards, window=10)
            ax.plot(episodes, smoothed, 'b-', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各Agent奖励
        ax = axes[0, 1]
        rewards_per_agent = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            ax.plot(episodes, rewards_per_agent[:, i], alpha=0.6, 
                   color=self.colors[i], label=name)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Agent Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Main Grid 进出口
        ax = axes[0, 2]
        ax.plot(episodes, logger.grid_imports, 'r-', alpha=0.6, label='Import')
        ax.plot(episodes, logger.grid_exports, 'g-', alpha=0.6, label='Export')
        ax.plot(episodes, logger.grid_net_trades, 'b-', linewidth=2, label='Net')
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title('Main Grid Trading')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 出清价格
        ax = axes[1, 0]
        ax.plot(episodes, logger.clearing_prices, 'purple', alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Price (cents/kWh)')
        ax.set_title('Clearing Price')
        ax.grid(True, alpha=0.3)
        
        # 5. 噪声水平
        ax = axes[1, 1]
        ax.plot(episodes, logger.noise_levels, 'orange', alpha=0.6)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Noise Level')
        ax.set_title('Exploration Noise')
        ax.grid(True, alpha=0.3)
        
        # 6. 市场独立度
        ax = axes[1, 2]
        if logger.grid_imports:
            total_trades = np.array(logger.grid_imports) + np.array(logger.grid_exports)
            independence = 100 * (1 - total_trades / (total_trades.max() + 1e-6))
            ax.plot(episodes, independence, 'teal', alpha=0.6)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Independence (%)')
            ax.set_title('Market Independence')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f"{self.config.plot_dir}/training_ep{episode:04d}.png"
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _smooth(self, data, window=10):
        if len(data) < window:
            return data
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            smoothed.append(np.mean(data[start:i+1]))
        return smoothed
    
    def create_summary_plot(self, logger):
        """Main Grid 专项分析图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Main Grid Analysis: {logger.config.exp_name}', fontsize=16)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        
        # 1. 进出口趋势
        ax = axes[0, 0]
        ax.plot(episodes, logger.grid_imports, 'r-', label='Import', linewidth=2)
        ax.plot(episodes, logger.grid_exports, 'g-', label='Export', linewidth=2)
        ax.fill_between(episodes, 0, logger.grid_imports, alpha=0.3, color='red')
        ax.fill_between(episodes, 0, logger.grid_exports, alpha=0.3, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (kWh)')
        ax.set_title('Grid Import vs Export')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 净交易量
        ax = axes[0, 1]
        colors = ['red' if x > 0 else 'green' for x in logger.grid_net_trades]
        ax.bar(episodes, logger.grid_net_trades, color=colors, alpha=0.6)
        ax.axhline(0, color='black', linewidth=1)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Net Trade (kWh)')
        ax.set_title('Net Grid Position (+ import, - export)')
        ax.grid(True, alpha=0.3, axis='y')
        
        # 3. 交易量分布
        ax = axes[1, 0]
        all_trades = logger.grid_imports + logger.grid_exports
        ax.hist(all_trades, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(np.mean(all_trades), color='red', linestyle='--', 
                  linewidth=2, label=f'Mean: {np.mean(all_trades):.2f}')
        ax.set_xlabel('Total Grid Trade (kWh)')
        ax.set_ylabel('Frequency')
        ax.set_title('Grid Dependency Distribution')
        ax.legend()
        
        # 4. 统计摘要
        ax = axes[1, 1]
        ax.axis('off')
        stats_text = f"""
Main Grid Statistics
{'='*30}
Total Episodes: {len(episodes)}

Avg Import: {np.mean(logger.grid_imports):.2f} kWh
Avg Export: {np.mean(logger.grid_exports):.2f} kWh
Avg Net: {np.mean(logger.grid_net_trades):.2f} kWh

Total Imported: {np.sum(logger.grid_imports):.2f} kWh
Total Exported: {np.sum(logger.grid_exports):.2f} kWh

Avg Clearing Price: {np.mean(logger.clearing_prices):.2f} ¢/kWh
Price Range: {np.min(logger.clearing_prices):.2f}-{np.max(logger.clearing_prices):.2f}

Market Independence: 
{100 * (1 - np.mean(logger.grid_imports + logger.grid_exports) / 
 (max(logger.grid_imports + logger.grid_exports) + 1e-6)):.1f}%
        """
        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center')
        
        plt.tight_layout()
        plot_path = f"{self.config.plot_dir}/grid_analysis_final.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return plot_path


class DebugTrainer:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config
        
        self.trainer = Trainer(
            envs=env, agents=agents, obs_size=None, act_size=None,
            buffer_capacity=config.buffer_capacity,
            lr=config.actor_lr, critic_lr=config.critic_lr,
            gamma=config.gamma, tau=config.tau
        )
        
        self.logger = DetailedLogger(config, env.agent_names)
        self.plotter = Plotter(config, env.agent_names)
        
        self.current_noise = config.noise_scale
    
    def rollout_episode(self, add_noise=True):
        obs = self.env.reset()
        episode_reward = np.zeros(self.env.num_agents)
        
        actions_collected = [[] for _ in range(self.env.num_agents)]
        
        # 记录 Grid 交易
        grid_import_total = 0
        grid_export_total = 0
        clearing_prices = []
        
        for step in range(self.config.max_steps):
            acts_list = []
            for i, agent in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)
                
                if add_noise:
                    noise = np.random.randn(*act.shape) * self.current_noise
                    act = np.clip(act + noise, -1, 1)
                
                acts_list.append(act)
                actions_collected[i].append(act[0])
            
            max_act_size = max(act.shape[1] for act in acts_list)
            acts = np.zeros((self.config.num_envs, self.env.num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            next_obs, rewards, dones = self.env.step(acts)
            self.trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            # 获取 auction 结果（需要访问环境内部数据）
            env_data = self.env.latest_env_data[0]
            auction_results = self.env._run_auction(0, acts[0], env_data)
            
            grid_import_total += auction_results.get('grid_import', 0)
            grid_export_total += auction_results.get('grid_export', 0)
            clearing_prices.append(auction_results.get('clearing_price', 0))
            
            episode_reward += rewards[0]
            obs = next_obs
            
            if dones[0].all():
                break
        
        self.current_noise = max(self.config.noise_min, 
                                 self.current_noise * self.config.noise_decay)
        
        actions_stats = [np.array(acts) for acts in actions_collected]
        
        # 计算市场独立度
        total_trade = grid_import_total + grid_export_total
        market_independence = 100 * (1 - total_trade / (self.config.max_steps * 150))
        
        return (episode_reward, step + 1, actions_stats, 
                grid_import_total, grid_export_total,
                np.mean(clearing_prices), market_independence)
    
    def train_step(self, batch_size):
        if len(self.trainer.replay_buffer) < batch_size:
            return
        self.trainer.train_agents(rollout_steps=0, batch_size=batch_size)
    
    def train(self):
        self.logger.log("\n" + "="*70)
        self.logger.log("🚀 开始调试训练")
        self.logger.log("="*70)
        
        self.logger.log(f"\n🔥 预热阶段 ({self.config.warmup_episodes} episodes)")
        for ep in range(self.config.warmup_episodes):
            self.rollout_episode(add_noise=True)
            self.logger.log(f"  预热 {ep+1}/{self.config.warmup_episodes}")
        
        self.logger.log(f"\n🏋️  训练阶段")
        self.logger.log("="*70)
        
        for episode in range(1, self.config.total_episodes + 1):
            (episode_rewards, episode_steps, actions_stats,
             grid_import, grid_export, avg_price, independence) = self.rollout_episode(add_noise=True)
            
            for _ in range(episode_steps):
                self.train_step(self.config.batch_size)
            
            episode_data = {
                'total_reward': episode_rewards.sum(),
                'rewards_per_agent': episode_rewards,
                'steps': episode_steps,
                'buffer_size': len(self.trainer.replay_buffer),
                'noise_level': self.current_noise,
                'actions': actions_stats,
                'grid_import': grid_import,
                'grid_export': grid_export,
                'grid_net_trade': grid_import - grid_export,
                'avg_clearing_price': avg_price,
                'market_independence': independence,
            }
            
            self.logger.log_episode(episode, episode_data)
            
            if episode % self.config.plot_interval == 0:
                self.logger.log(f"\n📊 生成可视化图表...")
                plot1 = self.plotter.plot_training_curves(self.logger, episode)
                self.logger.log(f"  ✓ {plot1}")
            
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        self.logger.log("\n" + "="*70)
        self.logger.log("🎉 训练完成！")
        self.logger.log("="*70)
        
        self.logger.save_stats()
        summary_plot = self.plotter.create_summary_plot(self.logger)
        self.logger.log(f"📊 Main Grid分析: {summary_plot}")
        
        self.save_checkpoint(self.config.total_episodes, is_final=True)
        self.logger.close()
    
    def save_checkpoint(self, episode, is_final=False):
        suffix = 'final' if is_final else f'ep{episode:04d}'
        save_path = f"{self.config.save_dir}/{self.config.exp_name}_{suffix}.pt"
        
        torch.save({
            'episode': episode,
            'agents': [agent.actor.state_dict() for agent in self.agents],
            'config': vars(self.config),
        }, save_path)
        
        self.logger.log(f"💾 保存checkpoint: {save_path}")


def main():
    config = DebugConfig()
    
    print("="*70)
    print("🔍 MADDPG 调试训练 with Main Grid Tracking")
    print("="*70)
    
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    print(f"\n✓ 环境创建成功: {env.agent_names}")
    
    agents = []
    for i, agent_name in enumerate(env.agent_names):
        agent = Agent(
            obs_size=5, act_size=env.act_sizes[agent_name],
            num_agents=env.num_agents, max_act_size=2,
            lr=config.actor_lr, critic_lr=config.critic_lr,
            gamma=config.gamma, tau=config.tau
        )
        agents.append(agent)
    
    print(f"✓ Agents创建成功")
    
    debug_trainer = DebugTrainer(env, agents, config)
    debug_trainer.train()
    
    print(f"\n✅ 训练完成！")


if __name__ == "__main__":
    main()