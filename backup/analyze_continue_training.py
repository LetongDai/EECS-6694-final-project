#!/usr/bin/env python3
"""
分析继续训练的结果
Analyze Continue Training Results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def analyze_continue_training(stats_path, checkpoint_path=None):
    """分析继续训练的结果"""
    
    print("=" * 70)
    print("📊 分析继续训练结果")
    print("=" * 70)
    
    # 1. 加载统计数据
    print(f"\n1️⃣  加载统计数据...")
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    print(f"✅ 统计数据加载成功")
    print(f"   实验名称: {stats['exp_name']}")
    print(f"   原始checkpoint: {stats.get('original_checkpoint', 'N/A')}")
    
    # 2. 基本信息
    print(f"\n2️⃣  训练信息")
    print(f"   原始训练: {stats['original_episodes']} episodes")
    print(f"   继续训练: {stats['additional_episodes']} episodes")
    print(f"   总episodes: {stats['original_episodes'] + stats['additional_episodes']}")
    
    # 3. 性能分析
    print(f"\n3️⃣  性能分析")
    
    episode_rewards = np.array(stats['episode_rewards'])
    agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
    
    # 总奖励统计
    total_rewards = np.sum(episode_rewards, axis=1)
    print(f"\n   📈 总奖励统计:")
    print(f"      最终平均 (最后20 eps): {np.mean(total_rewards[-20:]):.2f}")
    print(f"      最佳: {np.max(total_rewards):.2f}")
    print(f"      最差: {np.min(total_rewards):.2f}")
    print(f"      平均: {np.mean(total_rewards):.2f}")
    
    # 学习进度
    n = len(total_rewards)
    if n >= 20:
        early_avg = np.mean(total_rewards[:n//5])
        late_avg = np.mean(total_rewards[4*n//5:])
        improvement = ((late_avg - early_avg) / abs(early_avg)) * 100
        print(f"\n   📊 学习进度:")
        print(f"      早期平均 (前20%): {early_avg:.2f}")
        print(f"      后期平均 (后20%): {late_avg:.2f}")
        print(f"      提升: {improvement:+.1f}%")
    
    # 各Agent表现
    print(f"\n   🤖 各Agent表现 (最后20 episodes):")
    final_rewards = np.mean(episode_rewards[-20:], axis=0)
    final_total = np.sum(final_rewards)
    
    for i, name in enumerate(agent_names):
        if i < len(final_rewards):
            reward = final_rewards[i]
            percentage = (reward / final_total * 100) if final_total != 0 else 0
            print(f"      {name:10s}: {reward:8.2f} ({percentage:5.1f}%)")
    
    # 4. 对比原始训练
    if 'original_checkpoint' in stats and Path(stats['original_checkpoint']).exists():
        print(f"\n4️⃣  与原始训练对比")
        try:
            import torch
            original_ckpt = torch.load(stats['original_checkpoint'], 
                                      map_location='cpu', weights_only=False)
            
            if 'eval_reward' in original_ckpt:
                original_reward = original_ckpt['eval_reward']
                improvement = ((final_total - original_reward) / abs(original_reward)) * 100
                print(f"      原始模型奖励: {original_reward:.2f}")
                print(f"      继续训练后: {final_total:.2f}")
                print(f"      改进: {improvement:+.1f}%")
            else:
                print(f"      原始checkpoint没有评估数据")
        except Exception as e:
            print(f"      无法加载原始checkpoint: {e}")
    
    # 5. 评估数据
    if 'eval_rewards' in stats and stats['eval_rewards']:
        print(f"\n5️⃣  评估数据")
        eval_data = stats['eval_rewards']
        print(f"      评估次数: {len(eval_data)}")
        
        best_eval = max(eval_data, key=lambda x: x[1])
        print(f"      最佳评估: Episode {best_eval[0]}, 奖励 {best_eval[1]:.2f}")
        
        latest_eval = eval_data[-1]
        print(f"      最新评估: Episode {latest_eval[0]}, 奖励 {latest_eval[1]:.2f}")
    
    # 6. 生成可视化
    print(f"\n6️⃣  生成可视化...")
    plot_path = plot_continue_training_results(stats, stats_path)
    print(f"✅ 图表已保存: {plot_path}")
    
    # 7. 评级
    print(f"\n7️⃣  性能评级")
    if final_total > 50000:
        rating = "🏆 Excellent (>50k)"
        status = "优秀"
    elif final_total > 40000:
        rating = "✅ Very Good (40k-50k)"
        status = "良好"
    elif final_total > 20000:
        rating = "⚠️  Good (20k-40k)"
        status = "一般"
    else:
        rating = "❌ Needs Improvement (<20k)"
        status = "需要改进"
    
    print(f"      评级: {rating}")
    print(f"      状态: {status}")
    
    # 8. 建议
    print(f"\n8️⃣  训练建议")
    
    if improvement > 0:
        print(f"      ✅ 继续训练有效！奖励提升了 {improvement:.1f}%")
        if improvement < 5:
            print(f"      💡 提升较小，可能接近收敛")
            print(f"      💡 建议：尝试调整环境参数（如政策补贴）")
    else:
        print(f"      ⚠️  继续训练后性能下降")
        print(f"      💡 建议：")
        print(f"         - 降低学习率")
        print(f"         - 增加训练episodes")
        print(f"         - 检查是否过拟合")
    
    # Agent特定建议
    if final_rewards[3] < 100:  # Battery
        print(f"      💡 Battery性能较低 ({final_rewards[3]:.2f})")
        print(f"         建议降低charge/discharge cost")
    
    if final_rewards[4] < -500:  # Customer
        print(f"      💡 Customer成本过高 ({final_rewards[4]:.2f})")
        print(f"         建议增加GREEN_DISCOUNT")
    
    print(f"\n" + "=" * 70)
    print(f"✅ 分析完成！")
    print(f"=" * 70)
    
    return stats


def plot_continue_training_results(stats, stats_path):
    """绘制继续训练结果"""
    
    episode_rewards = np.array(stats['episode_rewards'])
    agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
    total_rewards = np.sum(episode_rewards, axis=1)
    
    original_episodes = stats['original_episodes']
    episodes = np.arange(1, len(total_rewards) + 1)
    actual_episodes = episodes + original_episodes  # 实际episode编号
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f'Continue Training Analysis: {stats["exp_name"]}', 
                 fontsize=16, fontweight='bold')
    
    # 1. 总奖励曲线
    ax = axes[0, 0]
    ax.plot(actual_episodes, total_rewards, 'b-', alpha=0.3, linewidth=1, label='Raw')
    
    # 平滑曲线
    if len(total_rewards) > 10:
        window = min(20, len(total_rewards) // 5)
        smoothed = np.convolve(total_rewards, np.ones(window)/window, mode='valid')
        smooth_episodes = actual_episodes[:len(smoothed)]
        ax.plot(smooth_episodes, smoothed, 'b-', linewidth=2, label=f'Smoothed ({window})')
    
    # 标注原始训练结束位置
    ax.axvline(original_episodes, color='red', linestyle='--', 
               linewidth=2, alpha=0.5, label='Continue from here')
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.set_title('Total Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 各Agent奖励
    ax = axes[0, 1]
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for i, name in enumerate(agent_names):
        if i < episode_rewards.shape[1]:
            agent_rewards = episode_rewards[:, i]
            if len(agent_rewards) > 10:
                window = min(20, len(agent_rewards) // 5)
                smoothed = np.convolve(agent_rewards, np.ones(window)/window, mode='valid')
                smooth_episodes = actual_episodes[:len(smoothed)]
                ax.plot(smooth_episodes, smoothed, color=colors[i], 
                       linewidth=2, label=name)
    
    ax.axvline(original_episodes, color='red', linestyle='--', 
               linewidth=2, alpha=0.5, label='Continue point')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Agent Rewards (Smoothed)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 最终性能分布
    ax = axes[1, 0]
    final_rewards = np.mean(episode_rewards[-20:], axis=0)
    bars = ax.bar(agent_names, final_rewards, color=colors, alpha=0.7)
    
    # 添加数值标签
    for bar, reward in zip(bars, final_rewards):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{reward:.0f}',
               ha='center', va='bottom' if reward > 0 else 'top')
    
    ax.set_ylabel('Average Reward (Last 20 Episodes)')
    ax.set_title('Final Agent Performance')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 4. 学习进度对比
    ax = axes[1, 1]
    
    n = len(total_rewards)
    if n >= 20:
        # 分成5个阶段
        stages = 5
        stage_size = n // stages
        stage_rewards = []
        stage_labels = []
        
        for i in range(stages):
            start_idx = i * stage_size
            end_idx = (i + 1) * stage_size if i < stages - 1 else n
            stage_reward = np.mean(total_rewards[start_idx:end_idx])
            stage_rewards.append(stage_reward)
            
            start_ep = original_episodes + start_idx + 1
            end_ep = original_episodes + end_idx
            stage_labels.append(f'Eps\n{start_ep}-{end_ep}')
        
        bars = ax.bar(stage_labels, stage_rewards, color='skyblue', alpha=0.7)
        
        # 添加趋势线
        x = np.arange(len(stage_rewards))
        z = np.polyfit(x, stage_rewards, 1)
        p = np.poly1d(z)
        ax.plot(x, p(x), "r--", linewidth=2, label='Trend')
        
        # 标注数值
        for bar, reward in zip(bars, stage_rewards):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{reward:.0f}',
                   ha='center', va='bottom')
        
        ax.set_ylabel('Average Reward')
        ax.set_title('Learning Progress by Stage')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # 保存
    save_path = Path(stats_path).parent / f"{stats['exp_name']}_analysis.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return save_path


def compare_with_original(continue_stats_path, original_stats_path):
    """对比继续训练和原始训练"""
    
    print("\n" + "=" * 70)
    print("🔀 对比继续训练 vs 原始训练")
    print("=" * 70)
    
    # 加载数据
    with open(continue_stats_path, 'r') as f:
        continue_stats = json.load(f)
    
    with open(original_stats_path, 'r') as f:
        original_stats = json.load(f)
    
    # 对比
    print(f"\n📊 性能对比:")
    
    original_rewards = np.array(original_stats['episode_rewards'])
    continue_rewards = np.array(continue_stats['episode_rewards'])
    
    original_final = np.mean(np.sum(original_rewards[-20:], axis=1))
    continue_final = continue_stats['final_total']
    
    improvement = ((continue_final - original_final) / abs(original_final)) * 100
    
    print(f"   原始训练最终奖励: {original_final:.2f}")
    print(f"   继续训练最终奖励: {continue_final:.2f}")
    print(f"   改进: {improvement:+.1f}%")
    
    # 各Agent对比
    print(f"\n   各Agent对比:")
    original_agent_rewards = np.mean(original_rewards[-20:], axis=0)
    continue_agent_rewards = continue_stats['final_rewards_per_agent']
    
    agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
    
    for i, name in enumerate(agent_names):
        if i < len(original_agent_rewards) and i < len(continue_agent_rewards):
            orig = original_agent_rewards[i]
            cont = continue_agent_rewards[i]
            change = ((cont - orig) / abs(orig)) * 100 if orig != 0 else 0
            print(f"      {name:10s}: {orig:8.2f} → {cont:8.2f} ({change:+6.1f}%)")


def main():
    """主函数"""
    
    import sys
    
    if len(sys.argv) > 1:
        stats_path = sys.argv[1]
    else:
        # 查找最新的continue training stats
        continue_logs = Path("continue_logs")
        if continue_logs.exists():
            stats_files = list(continue_logs.glob("continue_*_stats.json"))
            if stats_files:
                stats_path = str(max(stats_files, key=lambda p: p.stat().st_mtime))
                print(f"💡 使用最新的统计文件: {stats_path}")
            else:
                print("❌ 未找到统计文件")
                print("用法: python analyze_continue_training.py <stats_path>")
                return
        else:
            print("❌ continue_logs 目录不存在")
            return
    
    # 分析
    stats = analyze_continue_training(stats_path)
    
    # 如果有原始训练统计，进行对比
    if len(sys.argv) > 2:
        original_stats_path = sys.argv[2]
        compare_with_original(stats_path, original_stats_path)


if __name__ == "__main__":
    main()
