#!/usr/bin/env python3
"""
从训练统计文件直接读取结果
"""
import json
import numpy as np
from pathlib import Path

def analyze_stats_file(stats_file):
    """分析训练统计文件"""
    
    print("="*70)
    print("📊 训练统计分析")
    print("="*70)
    
    stats_path = Path(stats_file)
    
    if not stats_path.exists():
        print(f"\n❌ 文件不存在: {stats_file}")
        
        # 尝试查找可能的文件
        log_dir = Path("debug_logs")
        if log_dir.exists():
            json_files = list(log_dir.glob("*_stats.json"))
            if json_files:
                print(f"\n💡 找到以下统计文件:")
                for i, f in enumerate(json_files, 1):
                    print(f"   {i}. {f.name}")
                print(f"\n   使用方法: python analyze_stats.py {json_files[0]}")
        return
    
    print(f"\n📁 读取: {stats_file}")
    
    try:
        with open(stats_path, 'r') as f:
            stats = json.load(f)
    except Exception as e:
        print(f"❌ 读取失败: {e}")
        return
    
    print(f"✅ 读取成功")
    
    # 基本信息
    print(f"\n📦 实验信息:")
    exp_name = stats.get('exp_name', 'N/A')
    print(f"  实验名称: {exp_name}")
    
    # 配置信息
    config = stats.get('config', {})
    if config:
        print(f"\n⚙️  训练配置:")
        key_params = ['total_episodes', 'batch_size', 'actor_lr', 'critic_lr', 
                     'gamma', 'tau', 'noise_scale']
        for key in key_params:
            if key in config:
                print(f"  {key:20s}: {config[key]}")
    
    # 训练结果
    episode_rewards = stats.get('episode_rewards', [])
    
    if not episode_rewards:
        print(f"\n❌ 没有找到训练数据")
        return
    
    rewards_array = np.array(episode_rewards)
    
    print(f"\n📈 训练统计 ({len(rewards_array)} episodes):")
    print(f"  平均奖励:   {np.mean(rewards_array):10.2f}")
    print(f"  最大奖励:   {np.max(rewards_array):10.2f}")
    print(f"  最小奖励:   {np.min(rewards_array):10.2f}")
    print(f"  标准差:     {np.std(rewards_array):10.2f}")
    
    # 分阶段统计
    n = len(rewards_array)
    
    # 前20%
    early = rewards_array[:n//5]
    print(f"\n  前期 (0-{n//5}):")
    print(f"    平均: {np.mean(early):10.2f}")
    print(f"    标准差: {np.std(early):10.2f}")
    
    # 后20%
    late = rewards_array[4*n//5:]
    print(f"\n  后期 ({4*n//5}-{n}):")
    print(f"    平均: {np.mean(late):10.2f}")
    print(f"    标准差: {np.std(late):10.2f}")
    
    # 进步
    improvement = ((np.mean(late) - np.mean(early)) / abs(np.mean(early)) * 100) if np.mean(early) != 0 else 0
    print(f"\n  📊 学习进度: {improvement:+.1f}%")
    
    # 各Agent分析
    episode_rewards_per_agent = stats.get('episode_rewards_per_agent', [])
    
    if episode_rewards_per_agent:
        agent_names = stats.get('agent_names', ['wind', 'solar', 'diesel', 'battery', 'customer'])
        
        # 取后20%
        recent_data = np.array(episode_rewards_per_agent[4*n//5:])
        avg_per_agent = np.mean(recent_data, axis=0)
        std_per_agent = np.std(recent_data, axis=0)
        
        print(f"\n🤖 各Agent表现 (后20%数据):")
        print(f"  {'Agent':12s}  {'平均奖励':>10s}  {'标准差':>8s}  {'占比':>6s}  {'状态'}")
        print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*6}  {'-'*6}")
        
        total = avg_per_agent.sum()
        
        for i, name in enumerate(agent_names):
            if i < len(avg_per_agent):
                reward = avg_per_agent[i]
                std = std_per_agent[i]
                percentage = (reward / total * 100) if total != 0 else 0
                
                status = "✅ 正" if reward > 0 else "⚠️  负"
                
                print(f"  {name:12s}  {reward:10.2f}  {std:8.2f}  {percentage:5.1f}%  {status}")
        
        print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*6}")
        print(f"  {'总计':12s}  {total:10.2f}")
        
        # 性能评级
        print(f"\n🎯 性能评级:")
        if total > 50000:
            print(f"  🏆 优秀! (>50k)")
            print(f"     模型已经达到很好的性能")
        elif total > 40000:
            print(f"  ✅ 良好  (40k-50k)")
            print(f"     可以继续fine-tune以提升")
        elif total > 20000:
            print(f"  ⚠️  一般  (20k-40k)")
            print(f"     建议调整奖励函数或超参数")
        else:
            print(f"  ❌ 较差  (<20k)")
            print(f"     需要检查环境和奖励函数设计")
        
        # 具体建议
        print(f"\n💡 优化建议:")
        
        # 检查负奖励
        negative_agents = [agent_names[i] for i in range(len(avg_per_agent)) if avg_per_agent[i] < 0]
        if negative_agents:
            print(f"  ⚠️  负奖励agents: {', '.join(negative_agents)}")
            
            if 'diesel' in negative_agents:
                print(f"     - Diesel负奖励是预期的（碳税+燃料成本）")
                print(f"     - 如果想提高，可以在environment.py中降低CARBON_TAX")
            
            if 'customer' in negative_agents:
                print(f"     - Customer负奖励说明电费成本过高")
                print(f"     - 检查curtailment策略是否合理")
        
        # 检查收敛性
        if np.std(late) > 10000:
            print(f"  ⚠️  后期标准差较大 ({np.std(late):.0f})")
            print(f"     - 说明训练还未完全收敛")
            print(f"     - 建议继续训练或降低探索噪声")
        
        # 检查进步
        if improvement < 100:
            print(f"  ⚠️  学习进度较慢 ({improvement:.1f}%)")
            print(f"     - 建议增加学习率或调整奖励缩放")
    
    print(f"\n" + "="*70)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        stats_file = sys.argv[1]
    else:
        # 默认文件
        stats_file = "debug_logs/debug_maddpg_20251124_031242_stats.json"
        
        # 如果不存在，尝试查找
        if not Path(stats_file).exists():
            log_dir = Path("debug_logs")
            if log_dir.exists():
                json_files = list(log_dir.glob("*_stats.json"))
                if json_files:
                    stats_file = str(json_files[0])
                    print(f"💡 自动使用: {stats_file}\n")
    
    analyze_stats_file(stats_file)
