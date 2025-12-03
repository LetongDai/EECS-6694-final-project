#!/usr/bin/env python3
"""
Enhanced analysis with Main Grid statistics and Energy Balance
包含主电网统计和能量平衡的增强分析
"""
import torch
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


def analyze_checkpoint_with_grid(checkpoint_path, stats_path=None):
    """完整分析包括主电网统计"""
    
    print("=" * 70)
    print("📊 增强版分析 - 包含主电网统计")
    print("=" * 70)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        print(f"✅ 成功加载: {checkpoint_path}")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 基本信息
    print(f"\n📦 基本信息:")
    config = checkpoint.get('config', {})
    print(f"  实验名称: {config.get('exp_name', 'N/A')}")
    print(f"  训练Episode: {checkpoint.get('episode', 'N/A')}")
    
    eval_reward = checkpoint.get('eval_reward')
    if eval_reward is not None:
        print(f"  评估总奖励: {eval_reward:.2f} cents")
    
    eval_rewards_per_agent = checkpoint.get('eval_rewards_per_agent')
    if eval_rewards_per_agent is not None:
        print(f"\n🤖 各Agent奖励:")
        agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
        rewards = np.array(eval_rewards_per_agent)
        
        for i, name in enumerate(agent_names):
            if i < len(rewards):
                print(f"    {name:10s}: {rewards[i]:8.2f}")
        
        print(f"    {'─'*10}")
        print(f"    {'总计':10s}: {rewards.sum():8.2f}")
    
    # 分析统计文件（包含主电网数据）
    if stats_path and Path(stats_path).exists():
        print(f"\n📁 加载统计文件: {stats_path}")
        
        try:
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            
            analyze_grid_statistics(stats)
            
            # 生成可视化
            plot_path = plot_grid_analysis(stats, checkpoint_path)
            print(f"\n📊 可视化已保存: {plot_path}")
            
        except Exception as e:
            print(f"  ⚠️  读取统计文件时出错: {e}")
    else:
        print(f"\n⚠️  未找到统计文件，无法分析主电网数据")
        print(f"    需要在训练时保存详细的episode数据")
    
    print(f"\n" + "=" * 70)


def analyze_grid_statistics(stats):
    """分析主电网统计数据"""
    
    print(f"\n⚡ 主电网与能量平衡分析:")
    print("=" * 70)
    
    episode_rewards_per_agent = stats.get('episode_rewards_per_agent', [])
    
    if not episode_rewards_per_agent:
        print("  ❌ 没有足够的数据进行分析")
        return
    
    # 转换为numpy数组
    rewards_array = np.array(episode_rewards_per_agent)
    n_episodes = len(rewards_array)
    
    # 取后20%的数据作为稳定期
    stable_start = int(n_episodes * 0.8)
    stable_rewards = rewards_array[stable_start:]
    
    # 各Agent平均奖励
    avg_rewards = np.mean(stable_rewards, axis=0)
    agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
    
    print(f"\n1️⃣  各Agent表现 (后20%数据):")
    print(f"  {'Agent':12s}  {'平均奖励':>10s}  {'占比':>6s}  {'状态'}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}  {'-'*10}")
    
    total_reward = avg_rewards.sum()
    
    for i, name in enumerate(agent_names):
        if i < len(avg_rewards):
            reward = avg_rewards[i]
            percentage = (reward / total_reward * 100) if total_reward != 0 else 0
            status = "✅ 正收益" if reward > 0 else "⚠️  负收益"
            print(f"  {name:12s}  {reward:10.2f}  {percentage:5.1f}%  {status}")
    
    print(f"  {'-'*12}  {'-'*10}  {'-'*6}")
    print(f"  {'总计':12s}  {total_reward:10.2f}")
    
    # 能量平衡分析
    print(f"\n2️⃣  能量平衡估算:")
    print("  " + "─" * 66)
    
    # 基于奖励推算能量
    # 假设平均电价为20 cents/kWh
    avg_price = 20.0  # cents/kWh
    
    # 供应侧（发电）
    wind_energy = avg_rewards[0] / avg_price if avg_rewards[0] > 0 else 0
    solar_energy = avg_rewards[1] / avg_price if avg_rewards[1] > 0 else 0
    diesel_energy = (avg_rewards[2] + 0.08 * 75 * 24) / avg_price  # 考虑成本
    battery_discharge = avg_rewards[3] / avg_price if avg_rewards[3] > 0 else 0
    
    # 需求侧
    customer_cost = abs(avg_rewards[4])  # 客户成本（负值）
    customer_demand = customer_cost / avg_price
    
    total_supply = wind_energy + solar_energy + diesel_energy + battery_discharge
    
    print(f"  {'供应侧 (每日估算)':30s}")
    print(f"    Wind:      {wind_energy:8.2f} kWh")
    print(f"    Solar:     {solar_energy:8.2f} kWh")
    print(f"    Diesel:    {diesel_energy:8.2f} kWh")
    print(f"    Battery:   {battery_discharge:8.2f} kWh (放电)")
    print(f"    {'─'*10}")
    print(f"    总供应:    {total_supply:8.2f} kWh")
    
    print(f"\n  {'需求侧 (每日估算)':30s}")
    print(f"    Customer:  {customer_demand:8.2f} kWh")
    
    # 主电网平衡
    grid_balance = total_supply - customer_demand
    
    print(f"\n  {'主电网平衡':30s}")
    if grid_balance > 0:
        print(f"    向主电网输出: {grid_balance:8.2f} kWh")
        print(f"    状态: ✅ 微电网有盈余")
    elif grid_balance < 0:
        print(f"    从主电网购买: {abs(grid_balance):8.2f} kWh")
        print(f"    状态: ⚠️  需要主电网支持")
    else:
        print(f"    完美平衡: {grid_balance:8.2f} kWh")
        print(f"    状态: ✅ 自给自足")
    
    # 自给率
    self_sufficiency = (min(total_supply, customer_demand) / customer_demand * 100) if customer_demand > 0 else 0
    print(f"\n  自给自足率: {self_sufficiency:.1f}%")
    
    # 可再生能源占比
    renewable_energy = wind_energy + solar_energy
    renewable_ratio = (renewable_energy / total_supply * 100) if total_supply > 0 else 0
    print(f"  可再生能源占比: {renewable_ratio:.1f}%")
    
    # 性能评级
    print(f"\n3️⃣  性能评级:")
    
    if self_sufficiency >= 95:
        rating = "🏆 优秀 (≥95%)"
    elif self_sufficiency >= 85:
        rating = "✅ 良好 (85-95%)"
    elif self_sufficiency >= 70:
        rating = "⚠️  一般 (70-85%)"
    else:
        rating = "❌ 较差 (<70%)"
    
    print(f"  自给自足评级: {rating}")
    
    if renewable_ratio >= 60:
        green_rating = "🌱 清洁能源为主"
    elif renewable_ratio >= 40:
        green_rating = "♻️  混合能源"
    else:
        green_rating = "⚡ 传统能源为主"
    
    print(f"  能源结构评级: {green_rating}")
    
    # 改进建议
    print(f"\n4️⃣  优化建议:")
    
    if self_sufficiency < 90:
        print(f"  💡 提高自给自足率:")
        print(f"     - 增加可再生能源容量")
        print(f"     - 优化电池储能策略")
        print(f"     - 实施需求响应管理")
    
    if renewable_ratio < 50:
        print(f"  💡 提高清洁能源占比:")
        print(f"     - 增加风能/太阳能发电")
        print(f"     - 减少柴油发电依赖")
        print(f"     - 考虑引入GREEN_SUBSIDY政策")
    
    if grid_balance < -50:
        print(f"  💡 降低主电网依赖:")
        print(f"     - 优化负荷曲线")
        print(f"     - 增加本地发电容量")
        print(f"     - 改进储能调度策略")


def plot_grid_analysis(stats, checkpoint_path):
    """生成主电网分析可视化"""
    
    episode_rewards_per_agent = stats.get('episode_rewards_per_agent', [])
    if not episode_rewards_per_agent:
        return None
    
    rewards_array = np.array(episode_rewards_per_agent)
    agent_names = ['Wind', 'Solar', 'Diesel', 'Battery', 'Customer']
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Microgrid Energy Balance Analysis with Main Grid', 
                 fontsize=16, fontweight='bold')
    
    # 1. 各Agent奖励趋势
    ax = axes[0, 0]
    episodes = np.arange(1, len(rewards_array) + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, 5))
    
    for i, name in enumerate(agent_names):
        agent_rewards = rewards_array[:, i]
        # 平滑处理
        if len(agent_rewards) > 20:
            window = 20
            smoothed = np.convolve(agent_rewards, np.ones(window)/window, mode='valid')
            smooth_episodes = episodes[:len(smoothed)]
            ax.plot(smooth_episodes, smoothed, color=colors[i], 
                   linewidth=2, label=name)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward (cents)')
    ax.set_title('Agent Rewards Over Time (Smoothed)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # 2. 能量平衡估算
    ax = axes[0, 1]
    
    # 取后20%数据
    stable_start = int(len(rewards_array) * 0.8)
    stable_rewards = rewards_array[stable_start:]
    avg_rewards = np.mean(stable_rewards, axis=0)
    
    # 估算能量（简化）
    avg_price = 20.0
    energy_production = [
        avg_rewards[0] / avg_price if avg_rewards[0] > 0 else 0,  # Wind
        avg_rewards[1] / avg_price if avg_rewards[1] > 0 else 0,  # Solar
        (avg_rewards[2] + 0.08 * 75 * 24) / avg_price,  # Diesel
        avg_rewards[3] / avg_price if avg_rewards[3] > 0 else 0,  # Battery
    ]
    
    energy_names = ['Wind', 'Solar', 'Diesel', 'Battery']
    
    bars = ax.bar(energy_names, energy_production, color=colors[:4], alpha=0.7)
    ax.set_ylabel('Energy (kWh/day)')
    ax.set_title('Average Energy Production (Last 20%)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, energy in zip(bars, energy_production):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{energy:.1f}',
               ha='center', va='bottom')
    
    # 3. 供需平衡
    ax = axes[1, 0]
    
    total_supply = sum(energy_production)
    customer_demand = abs(avg_rewards[4]) / avg_price
    grid_balance = total_supply - customer_demand
    
    categories = ['Total\nSupply', 'Customer\nDemand', 'Grid\nBalance']
    values = [total_supply, customer_demand, grid_balance]
    colors_balance = ['green', 'orange', 'red' if grid_balance < 0 else 'blue']
    
    bars = ax.bar(categories, values, color=colors_balance, alpha=0.7)
    ax.set_ylabel('Energy (kWh/day)')
    ax.set_title('Energy Balance')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}',
               ha='center', va='bottom' if value > 0 else 'top')
    
    # 4. 能源结构饼图
    ax = axes[1, 1]
    
    renewable_energy = energy_production[0] + energy_production[1]
    fossil_energy = energy_production[2]
    storage_energy = energy_production[3]
    
    energy_types = ['Renewable\n(Wind+Solar)', 'Fossil\n(Diesel)', 'Storage\n(Battery)']
    energy_values = [renewable_energy, fossil_energy, storage_energy]
    colors_pie = ['green', 'gray', 'blue']
    
    # 过滤掉零值
    energy_values_filtered = [v for v in energy_values if v > 0]
    energy_types_filtered = [t for t, v in zip(energy_types, energy_values) if v > 0]
    colors_pie_filtered = [c for c, v in zip(colors_pie, energy_values) if v > 0]
    
    if energy_values_filtered:
        wedges, texts, autotexts = ax.pie(energy_values_filtered, 
                                           labels=energy_types_filtered,
                                           colors=colors_pie_filtered,
                                           autopct='%1.1f%%',
                                           startangle=90)
        ax.set_title('Energy Mix')
        
        # 设置文本样式
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    plt.tight_layout()
    
    # 保存
    checkpoint_name = Path(checkpoint_path).stem
    plot_path = f"debug_plots/{checkpoint_name}_grid_analysis.png"
    Path("debug_plots").mkdir(exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path


if __name__ == "__main__":
    import sys
    
    # 默认路径
    default_checkpoint = "/Users/ezslaptop/Desktop/MADDPG/final_models/maddpg_microgrid_best_20251124_164650/model_checkpoint.pt"
    default_stats = "final_models/maddpg_microgrid_best_20251124_164650/training_stats.json"
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        stats_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        checkpoint_path = default_checkpoint
        stats_path = default_stats
    
    analyze_checkpoint_with_grid(checkpoint_path, stats_path)
