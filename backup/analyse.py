#!/usr/bin/env python3
"""
简单稳健的训练结果分析脚本
"""
import torch
import json
import numpy as np
from pathlib import Path

def safe_format(value, format_str=".2f", default="N/A"):
    """安全的格式化函数"""
    if value is None:
        return default
    try:
        return f"{value:{format_str}}"
    except:
        return str(value)

def analyze_checkpoint_simple(checkpoint_path):
    """简单稳健的checkpoint分析"""
    
    print("=" * 70)
    print("📊 训练结果分析")
    print("=" * 70)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        print(f"✅ 成功加载: {checkpoint_path}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 基本信息
    print(f"\n📦 基本信息:")
    print(f"  Checkpoint包含的keys: {list(checkpoint.keys())}")
    
    config = checkpoint.get('config', {})
    if config:
        print(f"  实验名称: {config.get('exp_name', 'N/A')}")
    
    episode = checkpoint.get('episode')
    if episode:
        print(f"  训练到: Episode {episode}")
    
    eval_reward = checkpoint.get('eval_reward')
    if eval_reward is not None:
        print(f"  评估奖励: {safe_format(eval_reward)}")
    
    eval_rewards_per_agent = checkpoint.get('eval_rewards_per_agent')
    if eval_rewards_per_agent is not None:
        print(f"\n🤖 各Agent奖励:")
        agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
        rewards = np.array(eval_rewards_per_agent)
        for i, name in enumerate(agent_names):
            if i < len(rewards):
                print(f"    {name:10s}: {safe_format(rewards[i])}")
    
    # 分析统计文件
    if config:
        log_dir = config.get('log_dir', 'debug_logs')
        exp_name = config.get('exp_name', '')
        stats_file = Path(log_dir) / f"{exp_name}_stats.json"
        
        if stats_file.exists():
            print(f"\n📁 统计文件: {stats_file}")
            
            try:
                with open(stats_file, 'r') as f:
                    stats = json.load(f)
                
                episode_rewards = stats.get('episode_rewards', [])
                if episode_rewards:
                    rewards_array = np.array(episode_rewards)
                    
                    print(f"\n📈 训练统计:")
                    print(f"  总Episodes: {len(rewards_array)}")
                    print(f"  平均奖励: {safe_format(np.mean(rewards_array))}")
                    print(f"  最大奖励: {safe_format(np.max(rewards_array))}")
                    print(f"  最小奖励: {safe_format(np.min(rewards_array))}")
                    print(f"  标准差: {safe_format(np.std(rewards_array))}")
                    
                    # 最近100 episodes
                    recent = min(100, len(rewards_array))
                    recent_rewards = rewards_array[-recent:]
                    print(f"\n  最近{recent}episodes:")
                    print(f"    平均: {safe_format(np.mean(recent_rewards))}")
                    print(f"    标准差: {safe_format(np.std(recent_rewards))}")
                
                # 各Agent统计
                episode_rewards_per_agent = stats.get('episode_rewards_per_agent', [])
                if episode_rewards_per_agent:
                    agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
                    
                    # 取最后20%的数据
                    n = len(episode_rewards_per_agent)
                    recent_start = int(n * 0.8)
                    recent_data = np.array(episode_rewards_per_agent[recent_start:])
                    
                    print(f"\n🤖 各Agent表现 (后20%数据):")
                    avg_per_agent = np.mean(recent_data, axis=0)
                    
                    for i, name in enumerate(agent_names):
                        if i < len(avg_per_agent):
                            reward = avg_per_agent[i]
                            percentage = reward / avg_per_agent.sum() * 100 if avg_per_agent.sum() != 0 else 0
                            
                            status = "✅" if reward > 0 else "⚠️ "
                            print(f"    {status} {name:10s}: {safe_format(reward):>8s}  ({safe_format(percentage, '.1f')}%)")
                    
                    print(f"    {'─'*10}")
                    print(f"       {'总计':10s}: {safe_format(avg_per_agent.sum()):>8s}")
                    
            except Exception as e:
                print(f"  ⚠️  读取统计文件时出错: {e}")
        else:
            print(f"\n📁 统计文件未找到: {stats_file}")
    
    # 关键配置
    if config:
        print(f"\n⚙️  关键配置:")
        important_keys = ['actor_lr', 'critic_lr', 'gamma', 'tau', 
                         'noise_scale', 'noise_decay', 'batch_size',
                         'buffer_capacity', 'total_episodes']
        
        for key in important_keys:
            if key in config:
                print(f"  {key:20s}: {config[key]}")
    
    # 简单建议
    print(f"\n💡 快速建议:")
    
    if eval_rewards_per_agent is not None:
        rewards = np.array(eval_rewards_per_agent)
        total = rewards.sum()
        
        if total > 50000:
            print(f"  ✅ 性能优秀! 总奖励 {safe_format(total)}")
        elif total > 40000:
            print(f"  ✅ 性能良好! 总奖励 {safe_format(total)}")
        elif total > 20000:
            print(f"  ⚠️  性能一般，总奖励 {safe_format(total)}")
        else:
            print(f"  ❌ 需要改进，总奖励 {safe_format(total)}")
        
        # 检查负奖励
        agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
        negative_agents = [agent_names[i] for i in range(len(rewards)) if rewards[i] < 0]
        
        if negative_agents:
            print(f"\n  ⚠️  以下agents有负奖励: {', '.join(negative_agents)}")
            if 'diesel' in negative_agents:
                print(f"     💡 Diesel负奖励是正常的（碳税+燃料成本）")
                print(f"        如果想提高，可以降低environment.py中的CARBON_TAX")
    
    print(f"\n" + "=" * 70)

def quick_comparison(checkpoint_paths):
    """快速比较多个checkpoints"""
    
    print("\n" + "=" * 70)
    print("📊 模型对比")
    print("=" * 70 + "\n")
    
    results = []
    
    for path in checkpoint_paths:
        try:
            checkpoint = torch.load(path, map_location='cpu')
            
            name = Path(path).name
            eval_reward = checkpoint.get('eval_reward')
            eval_rewards_per_agent = checkpoint.get('eval_rewards_per_agent')
            
            if eval_rewards_per_agent is not None:
                total = np.array(eval_rewards_per_agent).sum()
            elif eval_reward is not None:
                total = eval_reward
            else:
                total = None
            
            results.append({
                'name': name,
                'total': total,
                'path': path
            })
            
        except Exception as e:
            print(f"⚠️  {path}: 加载失败 ({e})")
    
    # 排序
    results = [r for r in results if r['total'] is not None]
    results.sort(key=lambda x: x['total'], reverse=True)
    
    # 显示
    print(f"{'排名':<6} {'总奖励':>10} {'文件名':<50}")
    print("-" * 70)
    
    for i, result in enumerate(results, 1):
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "  "
        print(f"{emoji} #{i:<3} {safe_format(result['total']):>10} {result['name']:<50}")
    
    if results:
        print(f"\n🏆 最佳模型: {results[0]['name']}")
        print(f"   路径: {results[0]['path']}")

if __name__ == "__main__":
    import sys
    
    # 默认路径
    default_checkpoint = "debug_checkpoints/debug_maddpg_20251124_031242_final.pt"
    
    if len(sys.argv) > 1:
        # 从命令行参数读取
        checkpoint_path = sys.argv[1]
    else:
        checkpoint_path = default_checkpoint
    
    # 分析单个checkpoint
    analyze_checkpoint_simple(checkpoint_path)
    
    # 如果有多个参数，进行对比
    if len(sys.argv) > 2:
        print("\n")
        quick_comparison(sys.argv[1:])
    else:
        # 尝试找到所有checkpoints进行对比
        checkpoint_dir = Path("debug_checkpoints")
        if checkpoint_dir.exists():
            all_checkpoints = list(checkpoint_dir.glob("*.pt"))
            if len(all_checkpoints) > 1:
                quick_comparison([str(p) for p in all_checkpoints])