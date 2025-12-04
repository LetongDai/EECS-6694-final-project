#!/usr/bin/env python3
"""
Debug Training Script with Comprehensive Visualization
(Merged: Standard RL Plots + Main Grid Analysis)
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
# 确保你也更新了 models.py 和 agent.py 为 Transformer 版本
from agent import Agent 
from trainer import Trainer


class DebugConfig:
    exp_name = f"debug_maddpg_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 环境配置
    num_envs = 2
    max_steps = 24
    
    # 训练参数
    total_episodes = 200     # 可以根据需要增加
    batch_size = 128         # Transformer 建议大一点
    warmup_episodes = 10
    
    # Transformer 敏感参数
    actor_lr = 5e-5
    critic_lr = 5e-4
    
    gamma = 0.95
    tau = 0.01
    
    buffer_capacity = 20000
    
    # 噪声参数
    noise_scale = 0.3      # 初始噪声大一点
    noise_decay = 0.995    # 衰减慢一点
    noise_min = 0.02
    
    # 记录参数
    log_interval = 1
    plot_interval = 10     # 每10轮画一次图
    save_interval = 50
    
    save_dir = "debug_checkpoints"
    log_dir = "debug_logs"
    plot_dir = "debug_plots"
    
    def __init__(self):
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.plot_dir).mkdir(exist_ok=True)


class DetailedLogger:
    """详细日志记录器 (包含 Grid 和 RL 统计)"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        
        # 标准 RL 统计
        self.episode_rewards = []
        self.episode_rewards_per_agent = []
        self.episode_steps = []
        self.episode_losses = []
        self.buffer_sizes = []
        self.noise_levels = []
        
        # Main Grid 专项统计
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
        # 记录基础数据
        self.episode_rewards.append(episode_data['total_reward'])
        self.episode_rewards_per_agent.append(episode_data['rewards_per_agent'])
        self.episode_steps.append(episode_data['steps'])
        self.buffer_sizes.append(episode_data['buffer_size'])
        self.noise_levels.append(episode_data['noise_level'])
        
        # 记录 Grid 数据
        self.grid_imports.append(episode_data.get('grid_import', 0))
        self.grid_exports.append(episode_data.get('grid_export', 0))
        self.grid_net_trades.append(episode_data.get('grid_net_trade', 0))
        self.clearing_prices.append(episode_data.get('avg_clearing_price', 0))
        
        if 'losses' in episode_data:
            self.episode_losses.append(episode_data['losses'])
        
        self.recent_rewards.append(episode_data['total_reward'])
        
        # 控制台输出
        self.log(f"\n{'='*60}")
        self.log(f"Episode {episode}/{self.config.total_episodes} | Steps: {episode_data['steps']}")
        self.log(f"总奖励: {episode_data['total_reward']:8.2f} (Avg10: {np.mean(self.recent_rewards):8.2f})")
        
        self.log(f"Agent奖励:")
        for i, name in enumerate(self.agent_names):
            self.log(f"  {name:10s}: {episode_data['rewards_per_agent'][i]:8.3f}")
            
        self.log(f"Grid交易: Import={episode_data.get('grid_import',0):.1f} | Export={episode_data.get('grid_export',0):.1f}")
    
    def save_stats(self):
        # 将 numpy 数据转为 list 以便 JSON 序列化
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_rewards_per_agent': [r.tolist() for r in self.episode_rewards_per_agent],
            'episode_steps': self.episode_steps,
            'buffer_sizes': self.buffer_sizes,
            'noise_levels': self.noise_levels,
            'grid_imports': self.grid_imports,
            'grid_exports': self.grid_exports,
            'clearing_prices': self.clearing_prices
        }
        with open(f"{self.config.log_dir}/{self.config.exp_name}_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
            
    def close(self):
        self.log_file.close()


class Plotter:
    """全能绘图器 (融合了你的绘图代码 + Grid分析代码)"""
    
    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names
        # 使用你喜欢的颜色方案
        self.colors = ['#5D9CC9', '#70C168', '#A88679', '#999999', '#54CEDE']
        # Fallback if more agents than colors
        if len(agent_names) > len(self.colors):
            self.colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))

    def _smooth(self, data, window=10):
        """平滑曲线辅助函数"""
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')

    # =========================================================
    # 1. 标准训练曲线 (来自你提供的代码)
    # =========================================================
    def plot_training_status(self, logger, episode):
        """绘制标准 RL 训练状态 (Reward, Steps, Buffer, Noise)"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        
        # 1. 总奖励
        ax = axes[0, 0]
        ax.plot(episodes, logger.episode_rewards, 'b-', alpha=0.3, label='Raw')
        if len(logger.episode_rewards) > 10:
            smoothed = self._smooth(logger.episode_rewards, window=10)
            x_smooth = episodes[len(episodes)-len(smoothed):]
            ax.plot(x_smooth, smoothed, 'b-', linewidth=2, label='Smoothed (10)')
        ax.set_title('Total Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 各Agent奖励
        ax = axes[0, 1]
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 0:
                ax.plot(episodes, rewards_arr[:, i], alpha=0.6, color=self.colors[i], label=name)
        ax.set_title('Agent Rewards')
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.3)
        
        # 3. 步数 (Steps)
        ax = axes[0, 2]
        ax.plot(episodes, logger.episode_steps, 'g-', alpha=0.6)
        ax.set_title('Episode Length')
        ax.grid(True, alpha=0.3)
        
        # 4. Buffer Size
        ax = axes[1, 0]
        ax.plot(episodes, logger.buffer_sizes, 'r-', alpha=0.6)
        ax.set_title('Replay Buffer Size')
        ax.grid(True, alpha=0.3)
        
        # 5. Noise Level
        ax = axes[1, 1]
        ax.plot(episodes, logger.noise_levels, 'purple', alpha=0.6)
        ax.set_title('Exploration Noise')
        ax.grid(True, alpha=0.3)
        
        # 6. 最近奖励分布
        ax = axes[1, 2]
        recent_n = min(50, len(logger.episode_rewards))
        if recent_n > 0:
            recent = logger.episode_rewards[-recent_n:]
            ax.hist(recent, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'Reward Dist (Last {recent_n})')
        
        plt.tight_layout()
        save_path = f"{self.config.plot_dir}/training_status_ep{episode:04d}.png"
        plt.savefig(save_path, dpi=100)
        plt.close()
        return save_path

    # =========================================================
    # 2. 详细 Agent 分析 (来自你提供的代码)
    # =========================================================
    def plot_agent_detail(self, logger, episode):
        """绘制各Agent的详细分析 (平滑曲线 + 柱状图)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Agent Analysis - Episode {episode}', fontsize=16)
        
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        episodes = list(range(1, len(rewards_arr) + 1))
        
        # 1. 平滑趋势
        ax = axes[0, 0]
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 10:
                smoothed = self._smooth(rewards_arr[:, i], window=10)
                x_smooth = episodes[len(episodes)-len(smoothed):]
                ax.plot(x_smooth, smoothed, linewidth=2, color=self.colors[i], label=name)
        ax.set_title('Agent Rewards (Smoothed)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 最新一轮奖励
        ax = axes[0, 1]
        if len(rewards_arr) > 0:
            latest = rewards_arr[-1]
            bars = ax.bar(self.agent_names, latest, color=self.colors, alpha=0.7)
            ax.set_title('Latest Episode Rewards')
            ax.axhline(0, color='k', linewidth=0.5)
            # 标注数值
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}', ha='center', va='bottom' if h>0 else 'top')
        
        # 3. 全局平均奖励
        ax = axes[1, 0]
        if len(rewards_arr) > 0:
            means = np.mean(rewards_arr, axis=0)
            bars = ax.bar(self.agent_names, means, color=self.colors, alpha=0.7)
            ax.set_title('Average Rewards (All Eps)')
            ax.axhline(0, color='k', linewidth=0.5)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}', ha='center', va='bottom' if h>0 else 'top')
                
        # 4. 波动性 (Std Dev)
        ax = axes[1, 1]
        if len(rewards_arr) > 0:
            stds = np.std(rewards_arr, axis=0)
            bars = ax.bar(self.agent_names, stds, color=self.colors, alpha=0.7)
            ax.set_title('Reward Variability (Std)')
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x()+bar.get_width()/2, h, f'{h:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        save_path = f"{self.config.plot_dir}/agent_detail_ep{episode:04d}.png"
        plt.savefig(save_path, dpi=100)
        plt.close()
        return save_path

    # =========================================================
    # 3. Main Grid 专项分析 (保留之前的关键功能)
    # =========================================================
    def plot_grid_status(self, logger, episode):
        """绘制微网与主网交互状态 (进出口 + 电价)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Grid Interaction - Episode {episode}', fontsize=16)
        
        episodes = list(range(1, len(logger.grid_imports) + 1))
        
        # 1. Import vs Export
        ax = axes[0, 0]
        ax.plot(episodes, logger.grid_imports, 'r-', label='Import', alpha=0.7)
        ax.plot(episodes, logger.grid_exports, 'g-', label='Export', alpha=0.7)
        ax.fill_between(episodes, 0, logger.grid_imports, color='red', alpha=0.1)
        ax.fill_between(episodes, 0, logger.grid_exports, color='green', alpha=0.1)
        ax.set_title('Grid Energy Exchange (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Net Position
        ax = axes[0, 1]
        net = np.array(logger.grid_net_trades)
        colors = ['red' if x > 0 else 'green' for x in net]
        ax.bar(episodes, net, color=colors, alpha=0.5, width=1.0)
        ax.axhline(0, color='k', linewidth=1)
        ax.set_title('Net Position (+Import / -Export)')
        ax.grid(True, alpha=0.3)
        
        # 3. Clearing Price
        ax = axes[1, 0]
        ax.plot(episodes, logger.clearing_prices, 'purple', alpha=0.6)
        ax.set_title('Avg Clearing Price (cents/kWh)')
        ax.grid(True, alpha=0.3)
        
        # 4. Market Independence
        ax = axes[1, 1]
        imports = np.array(logger.grid_imports)
        exports = np.array(logger.grid_exports)
        total_trade = imports + exports
        # 简单定义独立性：trade 越少越独立
        max_trade = np.max(total_trade) if len(total_trade) > 0 else 1
        independence = 100 * (1 - total_trade / (max_trade + 1e-6))
        ax.plot(episodes, independence, 'teal', alpha=0.6)
        ax.set_title('Market Independence Index')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = f"{self.config.plot_dir}/grid_status_ep{episode:04d}.png"
        plt.savefig(save_path, dpi=100)
        plt.close()
        return save_path

    # =========================================================
    # 4. 最终总结报告 (融合版)
    # =========================================================
    def create_final_summary(self, logger):
        """创建最终的 3x3 综合分析图 (来自你提供的代码)"""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        episodes = list(range(1, len(logger.episode_rewards) + 1))
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        
        # Row 1: 全局奖励趋势
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(episodes, logger.episode_rewards, 'b-', alpha=0.2)
        if len(episodes) > 20:
            smoothed = self._smooth(logger.episode_rewards, window=20)
            x_smooth = episodes[len(episodes)-len(smoothed):]
            ax_main.plot(x_smooth, smoothed, 'b-', linewidth=3, label='Trend')
        ax_main.set_title('Total Reward Progression')
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)
        
        # Row 2, Col 1: Agent 趋势
        ax1 = fig.add_subplot(gs[1, 0])
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 10:
                smoothed = self._smooth(rewards_arr[:, i], window=10)
                x_smooth = episodes[len(episodes)-len(smoothed):]
                ax1.plot(x_smooth, smoothed, color=self.colors[i], label=name)
        ax1.set_title('Agent Rewards (Smoothed)')
        ax1.legend(fontsize=8)
        
        # Row 2, Col 2: 最终性能
        ax2 = fig.add_subplot(gs[1, 1])
        if len(rewards_arr) > 10:
            final_avg = np.mean(rewards_arr[-10:], axis=0)
            ax2.barh(self.agent_names, final_avg, color=self.colors, alpha=0.7)
            ax2.set_title('Final Performance (Last 10 Eps)')
            ax2.axvline(0, color='k', linewidth=0.5)

        # Row 2, Col 3: 学习进度对比
        ax3 = fig.add_subplot(gs[1, 2])
        if len(episodes) > 20:
            w = len(episodes) // 5
            early = np.mean(logger.episode_rewards[:w])
            late = np.mean(logger.episode_rewards[-w:])
            change = ((late-early)/abs(early)*100) if early !=0 else 0
            ax3.bar(['Early', 'Late'], [early, late], color=['gray', 'green'])
            ax3.set_title(f'Improvement: {change:+.1f}%')

        # Row 3, Col 1: Grid 概览 (插入之前的 Grid 逻辑)
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(episodes, logger.grid_imports, 'r', label='Imp', alpha=0.5)
        ax4.plot(episodes, logger.grid_exports, 'g', label='Exp', alpha=0.5)
        ax4.legend()
        ax4.set_title('Main Grid Interaction')
        ax4.grid(True, alpha=0.3)

        # Row 3, Col 2: 奖励分布
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.hist(logger.episode_rewards, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax5.set_title('Reward Distribution')

        # Row 3, Col 3: 文本统计
        ax6 = fig.add_subplot(gs[2, 2])
        ax6.axis('off')
        txt = f"""
        SUMMARY STATS
        -------------
        Episodes: {len(episodes)}
        Avg Reward (Last 20): {np.mean(logger.episode_rewards[-20:]):.2f}
        
        Grid Import Total: {sum(logger.grid_imports):.0f} kWh
        Grid Export Total: {sum(logger.grid_exports):.0f} kWh
        Avg Clearing Price: {np.mean(logger.clearing_prices):.2f}
        """
        ax6.text(0.1, 0.5, txt, family='monospace', fontsize=10)

        plt.suptitle(f'Experiment Summary: {logger.config.exp_name}', fontsize=16)
        save_path = f"{self.config.plot_dir}/final_summary.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        return save_path


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
        
        # Grid stats for this episode
        grid_import_total = 0
        grid_export_total = 0
        clearing_prices = []
        
        for step in range(self.config.max_steps):
            acts_list = []
            for i, agent in enumerate(self.agents):
                # 转为 Tensor
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)
                
                # 加噪声
                if add_noise:
                    noise = np.random.randn(*act.shape) * self.current_noise
                    act = np.clip(act + noise, -1, 1)
                
                acts_list.append(act)
                actions_collected[i].append(act[0])
            
            # 组合动作
            max_act_size = max(act.shape[1] for act in acts_list)
            acts = np.zeros((self.config.num_envs, self.env.num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            next_obs, rewards, dones = self.env.step(acts)
            
            # 存入 Buffer
            self.trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            # 获取 Auction 结果用于统计
            # 注意：这里需要访问 env 内部状态
            env_data = self.env.latest_env_data[0]
            auction_res = self.env._run_auction(0, acts[0], env_data)
            
            grid_import_total += auction_res.get('grid_import', 0)
            grid_export_total += auction_res.get('grid_export', 0)
            clearing_prices.append(auction_res.get('clearing_price', 0))
            
            episode_reward += rewards[0]
            obs = next_obs
            
            if dones[0].all():
                break
        
        # 噪声衰减
        self.current_noise = max(self.config.noise_min, 
                                 self.current_noise * self.config.noise_decay)
        
        # 整理统计数据
        actions_stats = [np.array(acts) for acts in actions_collected]
        total_trade = grid_import_total + grid_export_total
        net_trade = grid_import_total - grid_export_total
        
        return (episode_reward, step + 1, actions_stats, 
                grid_import_total, grid_export_total, net_trade,
                np.mean(clearing_prices))
    
    def train_step(self, batch_size):
        if len(self.trainer.replay_buffer) < batch_size:
            return
        self.trainer.train_agents(rollout_steps=0, batch_size=batch_size)
    
    def train(self):
        self.logger.log("\n" + "="*70)
        self.logger.log("🚀 开始融合版训练 (Standard RL + Grid Analysis)")
        self.logger.log("="*70)
        
        # 打印容量参数验证
        try:
            w_cap = self.env.microgrids[0].components['wind'].capacity
            pv_cap = self.env.microgrids[0].components['pv'].capacity
            self.logger.log(f"DEBUG: Wind Capacity = {w_cap} kW")
            self.logger.log(f"DEBUG: PV Capacity = {pv_cap} kW")
        except:
            pass

        # 预热
        for ep in range(self.config.warmup_episodes):
            self.rollout_episode(add_noise=True)
            if ep % 5 == 0: self.logger.log(f"Warmup {ep}/{self.config.warmup_episodes}")

        # 主循环
        for episode in range(1, self.config.total_episodes + 1):
            (ep_rewards, ep_steps, acts_stats, 
             g_imp, g_exp, g_net, avg_price) = self.rollout_episode(add_noise=True)
            
            # 训练更新
            for _ in range(ep_steps):
                self.train_step(self.config.batch_size)
            
            # 记录数据
            ep_data = {
                'total_reward': ep_rewards.sum(),
                'rewards_per_agent': ep_rewards,
                'steps': ep_steps,
                'buffer_size': len(self.trainer.replay_buffer),
                'noise_level': self.current_noise,
                'grid_import': g_imp,
                'grid_export': g_exp,
                'grid_net_trade': g_net,
                'avg_clearing_price': avg_price
            }
            self.logger.log_episode(episode, ep_data)
            
            # 📊 绘图 (融合版)
            if episode % self.config.plot_interval == 0:
                self.logger.log(f"\n📊 生成多维分析图表...")
                
                # 1. 基础 RL 状态图
                p1 = self.plotter.plot_training_status(self.logger, episode)
                
                # 2. 详细 Agent 分析图
                p2 = self.plotter.plot_agent_detail(self.logger, episode)
                
                # 3. Main Grid 专项图
                p3 = self.plotter.plot_grid_status(self.logger, episode)
                
                self.logger.log(f"  ✓ Training Status: {p1}")
                self.logger.log(f"  ✓ Agent Details:   {p2}")
                self.logger.log(f"  ✓ Grid Analysis:   {p3}")
            
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        # 结束
        self.logger.log("="*70)
        self.logger.log("🎉 训练完成")
        self.logger.save_stats()
        
        # 生成最终总结图
        final_plot = self.plotter.create_final_summary(self.logger)
        self.logger.log(f"📊 最终总结图: {final_plot}")
        self.logger.close()

    def save_checkpoint(self, episode):
        path = f"{self.config.save_dir}/{self.config.exp_name}_ep{episode}.pt"
        torch.save({
            'episode': episode,
            'agents': [a.actor.state_dict() for a in self.agents],
            'config': vars(self.config)
        }, path)
        self.logger.log(f"💾 Checkpoint saved: {path}")


def main():
    config = DebugConfig()
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    
    # 打印环境信息
    print(f"Environment: {env.num_agents} agents")
    
    # 初始化 Agents (Transformer 版)
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