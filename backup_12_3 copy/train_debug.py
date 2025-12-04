#!/usr/bin/env python3
"""
Debug Training Script with Comprehensive Visualization & Strategy Logging
整合功能:
1. Transformer 参数适配
2. Main Grid 进出口分析
3. 每日报价与发电策略日志
4. 多维度绘图系统
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

# 确保 environment.py 已更新 step() 返回 4 个值
from environment import MicrogridEnv
from agent import Agent 
from trainer import Trainer


class DebugConfig:
    exp_name = f"debug_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # --- 环境配置 ---
    num_envs = 2
    max_steps = 24
    
    # --- 训练参数 (适配 Transformer) ---
    total_episodes = 200    # 增加到 1000 以便观察长期收敛
    batch_size = 128         # Transformer 需要更大的 Batch 稳定梯度
    warmup_episodes = 20     # 给更多时间收集初始数据
    
    actor_lr = 5e-5          # 降低 LR 防止震荡
    critic_lr = 5e-4
    
    gamma = 0.95             # 关注长期收益
    tau = 0.01
    
    buffer_capacity = 50000
    
    # --- 噪声参数 ---
    noise_scale = 0.3
    noise_decay = 0.996      # 衰减更慢，保持探索
    noise_min = 0.02
    
    # --- 记录参数 ---
    log_interval = 1
    plot_interval = 20       # 每 20 轮画一次图
    save_interval = 100
    
    save_dir = "debug_checkpoints"
    log_dir = "debug_logs"
    plot_dir = "debug_plots"
    
    def __init__(self):
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.plot_dir).mkdir(exist_ok=True)


class DetailedLogger:
    """增强版日志：支持 Grid 分析和 策略详情"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        
        # RL 基础数据
        self.episode_rewards = []
        self.episode_rewards_per_agent = []
        self.episode_steps = []
        self.buffer_sizes = []
        self.noise_levels = []
        
        # Grid 数据
        self.grid_imports = []
        self.grid_exports = []
        self.grid_net_trades = []
        self.clearing_prices = []
        
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
        # 1. 归档数据
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_rewards_per_agent.append(episode_data['rewards_per_agent'])
        self.episode_steps.append(episode_data['steps'])
        self.buffer_sizes.append(episode_data['buffer_size'])
        self.noise_levels.append(episode_data['noise_level'])
        
        self.grid_imports.append(episode_data.get('grid_import', 0))
        self.grid_exports.append(episode_data.get('grid_export', 0))
        self.grid_net_trades.append(episode_data.get('grid_net_trade', 0))
        self.clearing_prices.append(episode_data.get('clearing_price', 0))
        
        self.recent_rewards.append(episode_data['total_reward'])
        
        # 2. 控制台输出 - 基础信息
        self.log(f"\n{'='*60}")
        self.log(f"Episode {episode}/{self.config.total_episodes} | Steps: {episode_data['steps']}")
        self.log(f"总奖励: {episode_data['total_reward']:8.2f} (Avg10: {np.mean(self.recent_rewards):8.2f})")
        
        self.log(f"Agent奖励:")
        for i, name in enumerate(self.agent_names):
            self.log(f"  {name:10s}: {episode_data['rewards_per_agent'][i]:8.3f}")
            
        self.log(f"Grid交易: Import={episode_data.get('grid_import',0):.1f} | Export={episode_data.get('grid_export',0):.1f}")
        
        # 3. 控制台输出 - 策略详情 (报价 & 发电)
        if 'avg_bids' in episode_data and 'total_gen' in episode_data:
            self.log("-" * 60)
            self.log(f"每日策略详情 (Avg Bid / Total Gen):")
            bids = episode_data['avg_bids']
            gen = episode_data['total_gen']
            
            # Wind
            self.log(f"Wind    : Bid {bids['wind']:5.1f} cents | Gen {gen['wind']:6.1f} kWh")
            # Solar
            self.log(f"Solar   : Bid {bids['solar']:5.1f} cents | Gen {gen['solar']:6.1f} kWh")
            # Diesel
            self.log(f"Diesel  : Bid {bids['diesel']:5.1f} cents | Gen {gen['diesel']:6.1f} kWh")
            # Battery (显示充放电状态)
            bat_net = gen['battery']
            action_str = "Dischg" if bat_net > 0 else "Charge"
            if abs(bat_net) < 0.1: action_str = "Idle"
            self.log(f"  Battery : Bid {bids['battery']:5.1f} cents | Net {bat_net:6.1f} kWh ({action_str})")
            
            self.log("-" * 60)
    
    def save_stats(self):
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_rewards_per_agent': [r.tolist() for r in self.episode_rewards_per_agent],
            'grid_imports': self.grid_imports,
            'grid_exports': self.grid_exports,
            'clearing_prices': self.clearing_prices,
            'buffer_sizes': self.buffer_sizes,
            'noise_levels': self.noise_levels
        }
        with open(f"{self.config.log_dir}/{self.config.exp_name}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
    def close(self):
        self.log_file.close()


class Plotter:
    """全能绘图器 (Training Curves + Agent Details + Grid Analysis)"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        self.colors = ['#5D9CC9', '#70C168', '#A88679', '#999999', '#54CEDE']

    def _smooth(self, data, window=10):
        if len(data) < window: return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    def plot_training_status(self, logger, episode):
        """Standard RL Curves"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        
        # Total Reward
        ax = axes[0, 0]
        ax.plot(episodes, logger.episode_rewards, 'b-', alpha=0.3, label='Raw')
        if len(episodes) > 10:
            smoothed = self._smooth(logger.episode_rewards)
            ax.plot(episodes[-len(smoothed):], smoothed, 'b-', linewidth=2)
        ax.set_title('Total Reward')
        ax.grid(True, alpha=0.3)
        
        # Agent Rewards
        ax = axes[0, 1]
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 0:
                ax.plot(episodes, rewards_arr[:, i], alpha=0.6, color=self.colors[i], label=name)
        ax.set_title('Agent Rewards')
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # Clearing Price
        ax = axes[0, 2]
        ax.plot(episodes, logger.clearing_prices, 'purple', alpha=0.6)
        ax.set_title('Clearing Price')
        ax.grid(True, alpha=0.3)
        
        # Buffer
        ax = axes[1, 0]
        ax.plot(episodes, logger.buffer_sizes, 'r-', alpha=0.6)
        ax.set_title('Buffer Size')
        ax.grid(True, alpha=0.3)
        
        # Noise
        ax = axes[1, 1]
        ax.plot(episodes, logger.noise_levels, 'orange', alpha=0.6)
        ax.set_title('Noise Level')
        ax.grid(True, alpha=0.3)
        
        # Reward Dist
        ax = axes[1, 2]
        recent = logger.episode_rewards[-50:] if len(logger.episode_rewards) > 0 else []
        ax.hist(recent, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Recent Reward Dist')
        
        plt.tight_layout()
        path = f"{self.config.plot_dir}/training_status_ep{episode:04d}.png"
        plt.savefig(path, dpi=100)
        plt.close()
        return path

    def plot_grid_status(self, logger, episode):
        """Main Grid Import/Export Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Grid Analysis - Episode {episode}', fontsize=16)
        
        episodes = list(range(1, len(logger.grid_imports) + 1))
        
        # Import vs Export
        ax = axes[0, 0]
        ax.plot(episodes, logger.grid_imports, 'r-', label='Import', alpha=0.7)
        ax.plot(episodes, logger.grid_exports, 'g-', label='Export', alpha=0.7)
        ax.fill_between(episodes, 0, logger.grid_imports, color='red', alpha=0.1)
        ax.fill_between(episodes, 0, logger.grid_exports, color='green', alpha=0.1)
        ax.set_title('Grid Energy Exchange (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Net Position
        ax = axes[0, 1]
        net = np.array(logger.grid_net_trades)
        colors = ['red' if x > 0 else 'green' for x in net]
        ax.bar(episodes, net, color=colors, alpha=0.5, width=1.0)
        ax.axhline(0, color='k', linewidth=1)
        ax.set_title('Net Position (+Import / -Export)')
        
        # Clearing Price Trend
        ax = axes[1, 0]
        ax.plot(episodes, logger.clearing_prices, 'purple', alpha=0.6)
        ax.set_title('Avg Clearing Price')
        ax.grid(True, alpha=0.3)
        
        # Stats Text
        ax = axes[1, 1]
        ax.axis('off')
        txt = f"Total Episodes: {len(episodes)}\n\n"
        txt += f"Avg Import: {np.mean(logger.grid_imports[-50:]):.1f} kWh\n"
        txt += f"Avg Export: {np.mean(logger.grid_exports[-50:]):.1f} kWh\n"
        txt += f"Avg Price : {np.mean(logger.clearing_prices[-50:]):.2f} ¢/kWh"
        ax.text(0.1, 0.5, txt, family='monospace', fontsize=12)
        
        plt.tight_layout()
        path = f"{self.config.plot_dir}/grid_status_ep{episode:04d}.png"
        plt.savefig(path, dpi=100)
        plt.close()
        return path

    def create_final_summary(self, logger):
        """生成最终总结大图 (3x3)"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        
        # 1. Total Reward Trend
        ax = fig.add_subplot(gs[0, :])
        ax.plot(episodes, logger.episode_rewards, 'b-', alpha=0.2)
        if len(episodes) > 20:
            smoothed = self._smooth(logger.episode_rewards, window=20)
            ax.plot(episodes[-len(smoothed):], smoothed, 'b-', linewidth=2)
        ax.set_title('Total Reward Progression')
        ax.grid(True, alpha=0.3)
        
        # 2. Agent Rewards
        ax = fig.add_subplot(gs[1, 0])
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 10:
                smoothed = self._smooth(rewards_arr[:, i])
                ax.plot(episodes[-len(smoothed):], smoothed, color=self.colors[i], label=name)
        ax.set_title('Agent Rewards (Smoothed)')
        ax.legend(fontsize='small')
        
        # 3. Grid Interaction
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(episodes, logger.grid_imports, 'r', alpha=0.5, label='Imp')
        ax.plot(episodes, logger.grid_exports, 'g', alpha=0.5, label='Exp')
        ax.set_title('Grid Interaction')
        ax.legend()
        
        # 4. Final Performance Bar
        ax = fig.add_subplot(gs[1, 2])
        if len(rewards_arr) > 10:
            final_avg = np.mean(rewards_arr[-10:], axis=0)
            ax.barh(self.agent_names, final_avg, color=self.colors, alpha=0.7)
            ax.set_title('Final Avg Rewards (Last 10)')
            
        plt.suptitle(f'Experiment Summary: {logger.config.exp_name}', fontsize=16)
        path = f"{self.config.plot_dir}/final_summary.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return path


class DebugTrainer:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config
        
        # 初始化 Trainer
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
        
        # Grid stats containers
        grid_import_total = 0
        grid_export_total = 0
        clearing_prices = []
        
        # 🌟 每日策略统计容器
        daily_stats = {
            'bids_sum': {'wind': 0, 'solar': 0, 'diesel': 0, 'battery': 0},
            'gen_sum': {'wind': 0, 'solar': 0, 'diesel': 0, 'battery': 0}
        }
        
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
            
            # Combine actions
            max_act_size = max(act.shape[1] for act in acts_list)
            acts = np.zeros((self.config.num_envs, self.env.num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            # 🌟 Step Environment (Expect 4 returns now)
            # next_obs, rewards, dones = self.env.step(acts) # Old
            next_obs, rewards, dones, infos = self.env.step(acts) # New
            
            # Buffer Add
            self.trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            # 🌟 Extract Data from Info
            info = infos[0] # Take first env
            
            # Accumulate Strategy Data
            for agent in ['wind', 'solar', 'diesel', 'battery']:
                daily_stats['bids_sum'][agent] += info['bids'][agent]
                daily_stats['gen_sum'][agent] += info['generation'][agent]
            
            # Accumulate Grid Data
            grid_import_total += info['generation']['main_grid_import']
            grid_export_total += info['generation']['main_grid_export']
            
            # Note: We need clearing price from environment usually, 
            # assuming it's available or we can approximate. 
            # For now, let's assume we can get it from info if added, 
            # otherwise skip precise price logging per step here or modify env to pass it.
            # Simplified: Pass 0 or modify env to return price in info['clearing_price']
            # Assuming env info has it:
            if 'clearing_price' in info:
                clearing_prices.append(info['clearing_price'])
            
            episode_reward += rewards[0]
            obs = next_obs
            
            if dones[0].all():
                break
        
        # Decay Noise
        self.current_noise = max(self.config.noise_min, 
                                 self.current_noise * self.config.noise_decay)
        
        # Process Stats
        actions_stats = [np.array(acts) for acts in actions_collected]
        net_trade = grid_import_total - grid_export_total
        avg_price = np.mean(clearing_prices) if clearing_prices else 0
        
        # Calculate Average Bids
        avg_bids = {k: v / (step + 1) for k, v in daily_stats['bids_sum'].items()}
        total_gen = daily_stats['gen_sum']
        
        return (episode_reward, step + 1, actions_stats, 
                grid_import_total, grid_export_total, net_trade,
                avg_price, avg_bids, total_gen)
    
    def train_step(self, batch_size):
        if len(self.trainer.replay_buffer) < batch_size:
            return
        self.trainer.train_agents(rollout_steps=0, batch_size=batch_size)
    
    def train(self):
        self.logger.log("\n" + "="*70)
        self.logger.log("开始训练 (Transformer Agents + Strategy Logging)")
        self.logger.log("="*70)
        
        # 🌟 关键安全检查: 确保是在 30/15 的配置下运行
        try:
            w_cap = self.env.microgrids[0].components['wind'].capacity
            pv_cap = self.env.microgrids[0].components['pv'].capacity
            self.logger.log(f"DEBUG: Wind Capacity = {w_cap} kW")
            self.logger.log(f"DEBUG: PV Capacity = {pv_cap} kW")
            
            if w_cap > 30 or pv_cap > 20:
                self.logger.log("警告: 检测到大容量配置！Main Grid 将会出现大量 Export。")
                self.logger.log("如果要复现论文缺电场景，请修改 environment.py 为 30/15。")
        except:
            pass

        # Warmup
        for ep in range(self.config.warmup_episodes):
            self.rollout_episode(add_noise=True)
            if ep % 5 == 0: self.logger.log(f"Warmup {ep}/{self.config.warmup_episodes}")

        # Main Loop
        for episode in range(1, self.config.total_episodes + 1):
            # Rollout
            (ep_rewards, ep_steps, acts_stats, 
             g_imp, g_exp, g_net, avg_price,
             avg_bids, total_gen) = self.rollout_episode(add_noise=True)
            
            # Train
            for _ in range(ep_steps):
                self.train_step(self.config.batch_size)
            
            # Log
            ep_data = {
                'total_reward': ep_rewards.sum(),
                'rewards_per_agent': ep_rewards,
                'steps': ep_steps,
                'buffer_size': len(self.trainer.replay_buffer),
                'noise_level': self.current_noise,
                'grid_import': g_imp,
                'grid_export': g_exp,
                'grid_net_trade': g_net,
                'clearing_price': avg_price,
                'avg_bids': avg_bids,   # 传入 Strategy
                'total_gen': total_gen  # 传入 Strategy
            }
            self.logger.log_episode(episode, ep_data)
            
            # Plot
            if episode % self.config.plot_interval == 0:
                self.logger.log(f"\n生成图表...")
                p1 = self.plotter.plot_training_status(self.logger, episode)
                p2 = self.plotter.plot_grid_status(self.logger, episode)
                self.logger.log(f"  {p1}")
                self.logger.log(f"  {p2}")
            
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        # End
        self.logger.log("="*70)
        self.logger.log("训练完成")
        self.logger.save_stats()
        final_plot = self.plotter.create_final_summary(self.logger)
        self.logger.log(f"最终总结图: {final_plot}")
        self.logger.close()

    def save_checkpoint(self, episode):
        path = f"{self.config.save_dir}/{self.config.exp_name}_ep{episode}.pt"
        torch.save({
            'episode': episode,
            'agents': [a.actor.state_dict() for a in self.agents],
            'config': vars(self.config)
        }, path)
        self.logger.log(f"Checkpoint saved: {path}")


def main():
    config = DebugConfig()
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    
    print(f"Environment initialized with {env.num_agents} agents.")
    
    # Init Agents (Transformer)
    agents = []
    for i, name in enumerate(env.agent_names):
        agent = Agent(
            obs_size=5, 
            act_size=env.act_sizes[name],
            num_agents=env.num_agents,
            max_act_size=2,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        agents.append(agent)
    
    trainer = DebugTrainer(env, agents, config)
    trainer.train()

if __name__ == "__main__":
    main()