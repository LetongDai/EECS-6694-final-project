import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from datetime import datetime


class DetailedLogger:
    """增强版日志：支持 Grid 分析和 策略详情"""

    def __init__(self, config, agent_names):
        self.config = config
        self.agent_names = agent_names

        # RL 基础数据
        self.episode_rewards = []
        self.episode_rewards_per_agent = []

        # Grid 数据
        self.grid_imports = []
        self.grid_exports = []
        self.clearing_prices = []

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

        self.grid_imports.append(episode_data['grid_import'])
        self.grid_exports.append(episode_data['grid_export'])
        self.clearing_prices.append(episode_data['clearing_price'])

        # 2. 控制台输出 - 基础信息
        self.log(f"\n{'=' * 60}")
        self.log(f"Episode {episode}/{self.config.total_episodes}")
        self.log(f"总奖励: {episode_data['total_reward']:8.2f}")

        self.log(f"Agent奖励:")
        for i, name in enumerate(self.agent_names):
            self.log(f"  {name:15s}: {episode_data['rewards_per_agent'][i]:8.3f}")

        self.log(
            f"Grid交易: Import={episode_data['grid_import']:.1f} | Export={episode_data['grid_export']:.1f}")

        # 3. 控制台输出 - 策略详情 (报价 & 发电) - 动态
        if 'avg_bids' in episode_data and 'total_gen' in episode_data:
            self.log("-" * 60)
            self.log(f"每日策略详情 (Avg Bid / Total Gen):")
            bids = episode_data['avg_bids']
            gen = episode_data['total_gen']

            # 动态输出所有agents
            for name in self.agent_names:
                if name == 'battery':
                    bat_net = gen[name]
                    action_str = "Dischg" if bat_net > 0 else "Charge" if bat_net < 0 else "Idle"
                    self.log(
                        f"  {name:12s}: Bid {bids[name]:5.1f} cents | Net {bat_net:6.1f} kWh ({action_str})")
                elif name != 'customer' and name in bids:
                    self.log(f"{name:14s}: Bid {bids[name]:5.1f} cents | Gen {gen[name]:6.1f} kWh")

            self.log("-" * 60)

    def save_stats(self):
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_rewards_per_agent': [r.tolist() for r in self.episode_rewards_per_agent],
            'grid_imports': self.grid_imports,
            'grid_exports': self.grid_exports,
            'clearing_prices': self.clearing_prices,
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
        self.num_agents = len(agent_names)
        base_colors = ['#5D9CC9', '#70C168', '#A88679', '#999999', '#54CEDE',
                       '#E87A5D', '#F4D35E', '#9B59B6', '#3498DB', '#E74C3C']
        self.colors = [base_colors[i % len(base_colors)] for i in range(self.num_agents)]

    def plot_training_status(self, logger, episode):
        """Standard RL Curves"""
        fig, axes = plt.subplots(1, 3, figsize=(10, 10))
        fig.suptitle(f'Training Progress - Episode {episode}', fontsize=16)

        episodes = list(range(1, len(logger.episode_rewards) + 1))

        # Total Reward
        ax = axes[0]
        ax.plot(episodes, logger.episode_rewards, 'b-', alpha=0.3, label='Raw')
        if len(episodes) > 10:
            ax.plot(episodes[-len(logger.episode_rewards):], logger.episode_rewards, 'b-', linewidth=2)
        ax.set_title('Total Reward')
        ax.grid(True, alpha=0.3)

        # Agent Rewards
        ax = axes[1]
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 0:
                ax.plot(episodes, rewards_arr[:, i], alpha=0.6,
                        color=self.colors[i], label=name)
        ax.set_title('Agent Rewards')
        ax.legend(fontsize='small', ncol=2 if self.num_agents > 5 else 1)
        ax.grid(True, alpha=0.3)

        # Clearing Price
        ax = axes[2]
        ax.plot(episodes, logger.clearing_prices, 'purple', alpha=0.6)
        ax.set_title('Clearing Price')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        path = f"{self.config.plot_dir}/training_status_ep{episode:04d}.png"
        plt.savefig(path, dpi=100)
        plt.close()
        return path

    def plot_grid_status(self, logger, episode):
        """Main Grid Import/Export Analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle(f'Grid Analysis - Episode {episode}', fontsize=16)

        episodes = list(range(1, len(logger.grid_imports) + 1))

        # Import vs Export
        ax = axes[0]
        ax.plot(episodes, logger.grid_imports, 'r-', label='Import', alpha=0.7)
        ax.plot(episodes, logger.grid_exports, 'g-', label='Export', alpha=0.7)
        ax.fill_between(episodes, 0, logger.grid_imports, color='red', alpha=0.1)
        ax.fill_between(episodes, 0, logger.grid_exports, color='green', alpha=0.1)
        ax.set_title('Grid Energy Exchange (kWh)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Clearing Price Trend
        ax = axes[1]
        ax.plot(episodes, logger.clearing_prices, 'purple', alpha=0.6)
        ax.set_title('Avg Clearing Price')
        ax.grid(True, alpha=0.3)

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
            ax.plot(episodes[-len(logger.episode_rewards):], logger.episode_rewards, 'b-', linewidth=2)
        ax.set_title('Total Reward Progression')
        ax.grid(True, alpha=0.3)

        # 2. Agent Rewards
        ax = fig.add_subplot(gs[1, 0])
        rewards_arr = np.array(logger.episode_rewards_per_agent)
        for i, name in enumerate(self.agent_names):
            if len(rewards_arr) > 10:
                ax.plot(episodes[-len(rewards_arr[:, i]):], rewards_arr[:, i], color=self.colors[i], label=name)
        ax.set_title('Agent Rewards')
        ax.legend(fontsize='small')

        # 3. Grid Interaction
        ax = fig.add_subplot(gs[1, 1])
        ax.plot(episodes, logger.grid_imports, 'r', alpha=0.5, label='Imp')
        ax.plot(episodes, logger.grid_exports, 'g', alpha=0.5, label='Exp')
        ax.set_title('Grid Interaction')
        ax.legend()

        plt.suptitle(f'Experiment Summary: {logger.config.exp_name}', fontsize=16)
        path = f"{self.config.plot_dir}/final_summary.png"
        plt.savefig(path, dpi=150)
        plt.close()
        return path
