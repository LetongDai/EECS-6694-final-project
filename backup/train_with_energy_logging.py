#!/usr/bin/env python3
"""
Enhanced Training with Energy Balance Logging
带能量平衡记录的增强训练脚本
"""

import torch
import numpy as np
from pathlib import Path
from datetime import datetime
import json

from environment import MicrogridEnv
from agent import Agent
from trainer import Trainer


class EnergyBalanceLogger:
    """记录能量平衡的详细日志"""
    
    def __init__(self):
        self.episode_data = []
        
    def log_episode(self, episode_num, env, actions, rewards, auction_results=None):
        """记录一个episode的能量数据"""
        
        # 从最后一步的环境数据中提取
        if hasattr(env, 'latest_env_data') and env.latest_env_data[0]:
            env_data = env.latest_env_data[0]
            
            # 记录基本信息
            episode_log = {
                'episode': episode_num,
                'total_reward': rewards.sum(),
                'rewards_per_agent': rewards.tolist(),
                
                # 环境数据
                'wind_speed': env_data.get('wind_speed', 0),
                'solar_irradiance': env_data.get('solar_irradiance', 0),
                'base_load': env_data.get('base_load', 0),
            }
            
            # 如果有拍卖结果，记录能量分配
            if auction_results:
                episode_log['energy_allocated'] = auction_results.get('allocated_power', {})
                episode_log['clearing_price'] = auction_results.get('clearing_price', 0)
                episode_log['actual_demand'] = auction_results.get('actual_demand', 0)
                episode_log['curtailed_load'] = auction_results.get('curtailed_load', 0)
                episode_log['grid_import'] = auction_results.get('grid_import', 0)
                episode_log['grid_export'] = auction_results.get('grid_export', 0)
            
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
        avg_stats = {
            'num_episodes': len(recent_data),
            'avg_total_reward': np.mean([d['total_reward'] for d in recent_data]),
            'avg_rewards_per_agent': np.mean([d['rewards_per_agent'] for d in recent_data], axis=0).tolist(),
        }
        
        # 能量统计
        if 'energy_allocated' in recent_data[0]:
            energy_keys = ['wind', 'solar', 'diesel', 'battery']
            for key in energy_keys:
                values = [d['energy_allocated'].get(key, 0) for d in recent_data]
                avg_stats[f'avg_{key}_energy'] = np.mean(values)
            
            avg_stats['avg_actual_demand'] = np.mean([d.get('actual_demand', 0) for d in recent_data])
            avg_stats['avg_curtailed_load'] = np.mean([d.get('curtailed_load', 0) for d in recent_data])
            avg_stats['avg_grid_import'] = np.mean([d.get('grid_import', 0) for d in recent_data])
            avg_stats['avg_grid_export'] = np.mean([d.get('grid_export', 0) for d in recent_data])
            avg_stats['avg_clearing_price'] = np.mean([d.get('clearing_price', 0) for d in recent_data])
        
        return avg_stats
    
    def save(self, filepath):
        """保存日志"""
        with open(filepath, 'w') as f:
            json.dump({
                'episode_data': self.episode_data,
                'statistics': self.get_statistics()
            }, f, indent=2)


def train_with_energy_logging(config):
    """带能量记录的训练"""
    
    print("=" * 70)
    print("🔋 MADDPG Training with Energy Balance Logging")
    print("=" * 70)
    
    # 创建环境
    env = MicrogridEnv(num_envs=config.num_envs, max_steps=config.max_steps)
    print(f"\n✅ Environment created: {env.agent_names}")
    
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
            
            # Step环境
            next_obs, rewards, dones = env.step(acts)
            trainer.replay_buffer.add((obs, acts, rewards, next_obs, dones))
            
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
        
        # 记录能量数据
        # 注意：这里我们记录的是最后一步的数据作为代表
        # 实际应该记录整个episode的平均值
        energy_logger.log_episode(episode, env, acts, episode_reward)
        
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
        wind = stats.get('avg_wind_energy', 0)
        solar = stats.get('avg_solar_energy', 0)
        diesel = stats.get('avg_diesel_energy', 0)
        battery = stats.get('avg_battery_energy', 0)
        
        total_supply = wind + solar + diesel + max(0, battery)
        
        print(f"\n  Supply (kWh/day):")
        print(f"    Wind:      {wind:8.2f} kWh")
        print(f"    Solar:     {solar:8.2f} kWh")
        print(f"    Diesel:    {diesel:8.2f} kWh")
        print(f"    Battery:   {battery:8.2f} kWh {'(discharge)' if battery > 0 else '(charge)'}")
        print(f"    ──────────")
        print(f"    Total:     {total_supply:8.2f} kWh")
        
        demand = stats.get('avg_actual_demand', 0)
        curtailed = stats.get('avg_curtailed_load', 0)
        grid_import = stats.get('avg_grid_import', 0)
        grid_export = stats.get('avg_grid_export', 0)
        
        print(f"\n  Demand (kWh/day):")
        print(f"    Actual:    {demand:8.2f} kWh")
        print(f"    Curtailed: {curtailed:8.2f} kWh")
        print(f"    Total:     {demand + curtailed:8.2f} kWh")
        
        print(f"\n  Main Grid:")
        print(f"    Import:    {grid_import:8.2f} kWh")
        print(f"    Export:    {grid_export:8.2f} kWh")
        print(f"    Net:       {grid_import - grid_export:8.2f} kWh {'(import)' if grid_import > grid_export else '(export)'}")
        
        # 计算指标
        self_sufficiency = (demand / (demand + grid_import) * 100) if (demand + grid_import) > 0 else 0
        renewable_ratio = ((wind + solar) / total_supply * 100) if total_supply > 0 else 0
        
        print(f"\n  Metrics:")
        print(f"    Self-sufficiency: {self_sufficiency:6.1f}%")
        print(f"    Renewable ratio:  {renewable_ratio:6.1f}%")
        print(f"    Clearing price:   {stats.get('avg_clearing_price', 0):6.2f} cents/kWh")
    else:
        print("  ⚠️  No energy data available")
        print("  ⚠️  Make sure to record actual power allocations during training")


class Config:
    """训练配置"""
    exp_name = f"energy_logged_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
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
    train_with_energy_logging(config)
