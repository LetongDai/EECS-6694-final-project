#!/usr/bin/env python3
"""
Training with Auction Results Logging
通过环境包装器记录拍卖结果
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from environment_fixed import MicrogridEnv
from agent import Agent
from trainer import Trainer


class AuctionLoggingEnv:
    """环境包装器 - 捕获并记录拍卖结果"""
    
    def __init__(self, base_env):
        self.env = base_env
        self.last_auction_results = []
        
    def __getattr__(self, name):
        """代理所有其他属性到base_env"""
        return getattr(self.env, name)
    
    def step(self, actions):
        """重写step方法以捕获拍卖结果"""
        # 先清空上次的结果
        self.last_auction_results = []
        
        # 调用原始step，同时记录拍卖结果
        next_obs_list = []
        rewards_list = []
        dones_list = []
        
        for env_idx in range(self.env.num_envs):
            # Generate environmental data
            env_data = self.env._generate_env_data(env_idx)
            self.env.latest_env_data[env_idx] = env_data
            
            # Run auction with agent actions
            auction_results = self.env._run_auction(env_idx, actions[env_idx], env_data)
            
            # 🔥 记录拍卖结果
            self.last_auction_results.append({
                'allocated_power': auction_results['allocated_power'].copy(),
                'clearing_price': auction_results['clearing_price'],
                'actual_demand': auction_results['actual_demand'],
                'curtailed_load': auction_results['curtailed_load'],
                'grid_import': auction_results['grid_import'],
                'grid_export': auction_results['grid_export'],
                'total_supply': auction_results['total_supply'],
            })
            
            # Update battery SOC based on auction results
            battery_power = auction_results['allocated_power'].get('battery', 0)
            battery_component = self.env.microgrids[env_idx].components['battery']
            if battery_power > 0:  # Discharging
                energy_change = battery_power * 1.0  # 1 hour timestep
                battery_component.soc -= energy_change / battery_component.capacity_kwh
            elif battery_power < 0:  # Charging
                energy_change = abs(battery_power) * 1.0
                battery_component.soc += energy_change / battery_component.capacity_kwh
            battery_component.soc = np.clip(battery_component.soc, 0.0, 1.0)
            
            # Get next observation
            next_obs = self.env._get_observation(env_idx, env_data, auction_results)
            next_obs_list.append(next_obs)
            
            # Calculate rewards
            rewards = self.env._calculate_rewards(env_idx, auction_results, env_data, self.env.use_policy)
            rewards_list.append(rewards)
            
            # Check if done
            self.env.current_steps[env_idx] += 1
            done = self.env.current_steps[env_idx] >= self.env.max_steps
            dones = np.full(self.env.num_agents, done)
            dones_list.append(dones)
        
        # Stack results
        next_obs = np.stack(next_obs_list, axis=0)
        rewards = np.stack(rewards_list, axis=0)
        dones = np.stack(dones_list, axis=0)
        
        return next_obs, rewards, dones


class EnergyBalanceLogger:
    """记录能量平衡的详细日志"""
    
    def __init__(self):
        self.episode_data = []
        
    def log_episode(self, episode_num, rewards, auction_results):
        """记录一个episode的能量数据"""
        
        if not auction_results:
            return
        
        # 平均所有step的数据（因为是24小时）
        avg_allocated = {}
        for key in ['wind', 'solar', 'diesel', 'battery']:
            values = [r['allocated_power'].get(key, 0) for r in auction_results]
            avg_allocated[key] = np.mean(values) if values else 0
        
        episode_log = {
            'episode': episode_num,
            'total_reward': float(rewards.sum()),
            'rewards_per_agent': rewards.tolist(),
            
            # 能量数据（24小时平均）
            'wind_energy': float(avg_allocated['wind'] * 24),  # kWh/day
            'solar_energy': float(avg_allocated['solar'] * 24),
            'diesel_energy': float(avg_allocated['diesel'] * 24),
            'battery_energy': float(avg_allocated['battery'] * 24),
            
            # 其他指标
            'avg_clearing_price': float(np.mean([r['clearing_price'] for r in auction_results])),
            'avg_actual_demand': float(np.mean([r['actual_demand'] for r in auction_results]) * 24),
            'avg_curtailed_load': float(np.mean([r['curtailed_load'] for r in auction_results]) * 24),
            'avg_grid_import': float(np.mean([r['grid_import'] for r in auction_results]) * 24),
            'avg_grid_export': float(np.mean([r['grid_export'] for r in auction_results]) * 24),
        }
        
        self.episode_data.append(episode_log)
    
    def get_statistics(self):
        """获取统计数据"""
        if not self.episode_data:
            return {}
        
        # 取后20%
        n = len(self.episode_data)
        start_idx = int(n * 0.8)
        recent_data = self.episode_data[start_idx:]
        
        # 计算平均值
        stats = {
            'num_episodes': len(recent_data),
            'avg_total_reward': np.mean([d['total_reward'] for d in recent_data]),
            'avg_rewards_per_agent': np.mean([d['rewards_per_agent'] for d in recent_data], axis=0).tolist(),
            
            # 能量统计
            'avg_wind_energy': np.mean([d['wind_energy'] for d in recent_data]),
            'avg_solar_energy': np.mean([d['solar_energy'] for d in recent_data]),
            'avg_diesel_energy': np.mean([d['diesel_energy'] for d in recent_data]),
            'avg_battery_energy': np.mean([d['battery_energy'] for d in recent_data]),
            
            'avg_actual_demand': np.mean([d['avg_actual_demand'] for d in recent_data]),
            'avg_curtailed_load': np.mean([d['avg_curtailed_load'] for d in recent_data]),
            'avg_grid_import': np.mean([d['avg_grid_import'] for d in recent_data]),
            'avg_grid_export': np.mean([d['avg_grid_export'] for d in recent_data]),
            'avg_clearing_price': np.mean([d['avg_clearing_price'] for d in recent_data]),
        }
        
        return stats


def train_with_auction_logging(config):
    """带拍卖结果记录的训练"""
    
    print("=" * 70)
    print("🔋 MADDPG Training with Auction Results Logging")
    print("=" * 70)
    
    # 创建环境（用包装器）
    base_env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps, use_policy=True, use_time_of_export_pricing=True)
    env = AuctionLoggingEnv(base_env)
    print(f"\n✅ Environment created with logging wrapper")
    print(f"   Agents: {env.agent_names}")
    
    # 创建Agents
    agents = []
    for agent_name in env.agent_names:
        agent = Agent(
            obs_size=5,
            act_size=env.act_sizes[agent_name],
            num_agents=env.num_agents,
            max_act_size=2,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        agents.append(agent)
    
    print(f"✅ Agents created")
    
    # 创建Trainer
    trainer = Trainer(
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
    
    # 创建能量记录器
    energy_logger = EnergyBalanceLogger()
    current_noise = config.noise_scale
    
    print(f"\n🏋️  Starting training...")
    print("=" * 70)
    
    for episode in range(1, config.total_episodes + 1):
        # Rollout episode
        obs = env.reset()
        episode_reward = np.zeros(env.num_agents)
        episode_auction_results = []
        
        for step in range(config.max_steps):
            # 收集动作
            acts_list = []
            for i, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)
                
                # 添加噪声
                if episode > config.warmup_episodes:
                    noise = np.random.randn(*act.shape) * current_noise
                    act = np.clip(act + noise, 0, 1)
                
                acts_list.append(act)
            
            # 构造动作数组
            acts = np.zeros((config.num_envs, env.num_agents, 2))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            # Step环境（会记录拍卖结果到env.last_auction_results）
            next_obs, rewards, dones = env.step(acts)
            trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
            # 🔥 捕获这一步的拍卖结果
            if env.last_auction_results:
                episode_auction_results.append(env.last_auction_results[0])  # 取第一个环境
            
            episode_reward += rewards[0]
            obs = next_obs
            
            if dones[0].all():
                break
        
        # 训练
        if episode > config.warmup_episodes and len(trainer.replay_buffer) >= config.batch_size:
            for _ in range(config.max_steps):
                trainer.train_agents(rollout_steps=0, batch_size=config.batch_size)
            
            # 衰减噪声
            current_noise = max(config.noise_min, current_noise * config.noise_decay)
        
        # 🔥 记录能量数据
        energy_logger.log_episode(episode, episode_reward, episode_auction_results)
        
        # 打印日志
        if episode % config.log_interval == 0:
            print(f"\n📊 Episode {episode}/{config.total_episodes}")
            print(f"  Total Reward: {episode_reward.sum():8.2f}")
            print(f"  Per-agent: {', '.join([f'{r:.1f}' for r in episode_reward])}")
            print(f"  Noise: {current_noise:.4f}")
    
    # 保存统计数据
    print(f"\n💾 Saving logs...")
    
    stats = energy_logger.get_statistics()
    stats['config'] = vars(config)
    stats['exp_name'] = config.exp_name
    stats['all_episodes'] = energy_logger.episode_data  # 保存所有episode数据
    
    stats_path = f"{config.log_dir}/{config.exp_name}_energy_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"✅ Energy statistics saved: {stats_path}")
    
    # 打印能量平衡摘要
    print_energy_summary(stats)
    
    # 保存模型
    checkpoint_path = f"{config.save_dir}/{config.exp_name}_final.pt"
    torch.save({
        'episode': config.total_episodes,
        'agents': [agent.actor.state_dict() for agent in agents],
        'config': vars(config),
        'energy_stats': stats
    }, checkpoint_path)
    
    print(f"✅ Model saved: {checkpoint_path}")
    print(f"\n{'='*70}")
    print(f"🎉 Training complete!")
    print(f"{'='*70}")


def print_energy_summary(stats):
    """打印能量平衡摘要"""
    
    print(f"\n⚡ Energy Balance Summary (Last 20% episodes):")
    print("=" * 70)
    
    if 'avg_wind_energy' in stats:
        wind = stats['avg_wind_energy']
        solar = stats['avg_solar_energy']
        diesel = stats['avg_diesel_energy']
        battery = stats['avg_battery_energy']
        
        total_supply = wind + solar + diesel + max(0, battery)
        
        print(f"\n  Supply (kWh/day):")
        print(f"    Wind:      {wind:8.2f} kWh")
        print(f"    Solar:     {solar:8.2f} kWh")
        print(f"    Diesel:    {diesel:8.2f} kWh")
        print(f"    Battery:   {battery:8.2f} kWh {'(discharge)' if battery > 0 else '(charge)'}")
        print(f"    ──────────")
        print(f"    Total:     {total_supply:8.2f} kWh")
        
        demand = stats['avg_actual_demand']
        curtailed = stats['avg_curtailed_load']
        grid_import = stats['avg_grid_import']
        grid_export = stats['avg_grid_export']
        
        print(f"\n  Demand (kWh/day):")
        print(f"    Actual:    {demand:8.2f} kWh")
        print(f"    Curtailed: {curtailed:8.2f} kWh")
        print(f"    Total:     {demand + curtailed:8.2f} kWh")
        
        print(f"\n  Main Grid:")
        print(f"    Import:    {grid_import:8.2f} kWh")
        print(f"    Export:    {grid_export:8.2f} kWh")
        net = grid_import - grid_export
        print(f"    Net:       {net:8.2f} kWh {'(import)' if net > 0 else '(export)'}")
        
        # 验证能量平衡
        total_generation = total_supply + grid_import - grid_export
        total_consumption = demand
        balance_error = abs(total_generation - total_consumption)
        
        print(f"\n  Energy Balance Check:")
        print(f"    Generation: {total_generation:8.2f} kWh")
        print(f"    Consumption: {total_consumption:8.2f} kWh")
        print(f"    Balance Error: {balance_error:8.2f} kWh ({balance_error/total_consumption*100:.1f}%)")
        
        # 计算指标
        self_sufficiency = (total_supply / (total_supply + grid_import) * 100) if (total_supply + grid_import) > 0 else 0
        renewable_ratio = ((wind + solar) / total_supply * 100) if total_supply > 0 else 0
        
        print(f"\n  Metrics:")
        print(f"    Self-sufficiency: {self_sufficiency:6.1f}%")
        print(f"    Renewable ratio:  {renewable_ratio:6.1f}%")
        print(f"    Clearing price:   {stats['avg_clearing_price']:6.2f} cents/kWh")
    else:
        print("  ⚠️  No energy data available")


class Config:
    """训练配置"""
    exp_name = f"auction_logged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # 环境
    num_envs = 4
    max_steps = 24
    
    # 训练
    total_episodes = 200
    batch_size = 64
    warmup_episodes = 10
    
    # 学习率
    actor_lr = 1e-4
    critic_lr = 1e-3
    
    # RL参数
    gamma = 0.95
    tau = 0.01
    
    # Buffer
    buffer_capacity = 10000
    
    # 噪声
    noise_scale = 0.1
    noise_decay = 0.999
    noise_min = 0.01
    
    # 日志
    log_interval = 10
    
    # 路径
    save_dir = "debug_checkpoints"
    log_dir = "debug_logs"
    
    def __init__(self):
        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)


if __name__ == "__main__":
    config = Config()
    train_with_auction_logging(config)
