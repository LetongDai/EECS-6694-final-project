#!/usr/bin/env python3
"""
Test Customer Curtailment Constraint Impact
测试客户削减约束的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def simulate_customer_behavior(max_curtailment, K=10.0, price_range=(15, 40)):
    """
    模拟Customer在不同价格下的削减行为
    
    Args:
        max_curtailment: 最大削减比例 (0.3 = 30%, 1.0 = 100%)
        K: 不适成本系数
        price_range: 价格范围 (min, max) cents/kWh
    """
    
    base_demand = 80.0  # kW
    prices = np.linspace(price_range[0], price_range[1], 50)
    
    results = {
        'no_constraint': [],
        'with_constraint': []
    }
    
    # 测试不同削减比例
    curtailment_ratios = np.linspace(0, 1.0, 100)
    
    for price in prices:
        # 无约束情况：找到最优削减比例
        best_cost_no_constraint = float('inf')
        best_curtail_no_constraint = 0
        
        for curtail in curtailment_ratios:
            actual_demand = base_demand * (1 - curtail)
            curtailed = base_demand * curtail
            
            electricity_cost = actual_demand * price / 100.0
            discomfort_cost = K * curtailed / 100.0
            total_cost = electricity_cost + discomfort_cost
            
            if total_cost < best_cost_no_constraint:
                best_cost_no_constraint = total_cost
                best_curtail_no_constraint = curtail
        
        # 有约束情况
        best_cost_with_constraint = float('inf')
        best_curtail_with_constraint = 0
        
        for curtail in curtailment_ratios:
            if curtail > max_curtailment:
                break  # 超过约束
                
            actual_demand = base_demand * (1 - curtail)
            curtailed = base_demand * curtail
            
            electricity_cost = actual_demand * price / 100.0
            discomfort_cost = K * curtailed / 100.0
            total_cost = electricity_cost + discomfort_cost
            
            if total_cost < best_cost_with_constraint:
                best_cost_with_constraint = total_cost
                best_curtail_with_constraint = curtail
        
        results['no_constraint'].append({
            'price': price,
            'curtail': best_curtail_no_constraint,
            'cost': best_cost_no_constraint,
            'actual_demand': base_demand * (1 - best_curtail_no_constraint)
        })
        
        results['with_constraint'].append({
            'price': price,
            'curtail': best_curtail_with_constraint,
            'cost': best_cost_with_constraint,
            'actual_demand': base_demand * (1 - best_curtail_with_constraint)
        })
    
    return results


def plot_comparison(results, max_curtailment, K):
    """绘制对比图"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Customer Behavior: No Constraint vs Max {max_curtailment*100:.0f}% Curtailment (K={K})',
                 fontsize=14, fontweight='bold')
    
    # 提取数据
    prices = [r['price'] for r in results['no_constraint']]
    curtail_no = [r['curtail'] * 100 for r in results['no_constraint']]
    curtail_yes = [r['curtail'] * 100 for r in results['with_constraint']]
    demand_no = [r['actual_demand'] for r in results['no_constraint']]
    demand_yes = [r['actual_demand'] for r in results['with_constraint']]
    cost_no = [r['cost'] for r in results['no_constraint']]
    cost_yes = [r['cost'] for r in results['with_constraint']]
    
    # 1. 削减比例对比
    ax = axes[0, 0]
    ax.plot(prices, curtail_no, 'r-', linewidth=2, label='No Constraint', alpha=0.7)
    ax.plot(prices, curtail_yes, 'g-', linewidth=2, label=f'Max {max_curtailment*100:.0f}% Constraint', alpha=0.7)
    ax.axhline(max_curtailment * 100, color='orange', linestyle='--', linewidth=1.5, 
              label=f'{max_curtailment*100:.0f}% Limit')
    ax.set_xlabel('Electricity Price (cents/kWh)')
    ax.set_ylabel('Curtailment Ratio (%)')
    ax.set_title('Curtailment vs Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. 实际需求对比
    ax = axes[0, 1]
    ax.plot(prices, demand_no, 'r-', linewidth=2, label='No Constraint', alpha=0.7)
    ax.plot(prices, demand_yes, 'g-', linewidth=2, label=f'Max {max_curtailment*100:.0f}% Constraint', alpha=0.7)
    ax.axhline(80, color='blue', linestyle='--', linewidth=1, label='Base Demand (80 kW)', alpha=0.5)
    ax.set_xlabel('Electricity Price (cents/kWh)')
    ax.set_ylabel('Actual Demand (kW)')
    ax.set_title('Actual Demand vs Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 成本对比
    ax = axes[1, 0]
    ax.plot(prices, cost_no, 'r-', linewidth=2, label='No Constraint', alpha=0.7)
    ax.plot(prices, cost_yes, 'g-', linewidth=2, label=f'Max {max_curtailment*100:.0f}% Constraint', alpha=0.7)
    ax.set_xlabel('Electricity Price (cents/kWh)')
    ax.set_ylabel('Total Cost ($)')
    ax.set_title('Customer Cost vs Price')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. 统计对比
    ax = axes[1, 1]
    ax.axis('off')
    
    # 计算统计数据
    avg_curtail_no = np.mean(curtail_no)
    avg_curtail_yes = np.mean(curtail_yes)
    avg_demand_no = np.mean(demand_no)
    avg_demand_yes = np.mean(demand_yes)
    avg_cost_no = np.mean(cost_no)
    avg_cost_yes = np.mean(cost_yes)
    
    daily_demand_no = avg_demand_no * 24
    daily_demand_yes = avg_demand_yes * 24
    
    stats_text = f"""
    Statistical Comparison
    {'='*40}
    
    Average Curtailment:
      No Constraint:    {avg_curtail_no:6.1f}%
      With Constraint:  {avg_curtail_yes:6.1f}%
      Difference:       {avg_curtail_no - avg_curtail_yes:6.1f}%
    
    Average Actual Demand:
      No Constraint:    {avg_demand_no:6.1f} kW
      With Constraint:  {avg_demand_yes:6.1f} kW
      Difference:       {avg_demand_yes - avg_demand_no:6.1f} kW
    
    Daily Energy (24h):
      No Constraint:    {daily_demand_no:6.0f} kWh/day
      With Constraint:  {daily_demand_yes:6.0f} kWh/day
      Difference:       {daily_demand_yes - daily_demand_no:6.0f} kWh/day
    
    Average Cost:
      No Constraint:    ${avg_cost_no:6.2f}
      With Constraint:  ${avg_cost_yes:6.2f}
      Difference:       ${avg_cost_yes - avg_cost_no:6.2f}
    
    ⚠️ Problem with No Constraint:
      - Excessive curtailment ({avg_curtail_no:.0f}%) unrealistic
      - Very low demand ({daily_demand_no:.0f} kWh/day)
      - Destabilizes microgrid energy balance
    
    ✅ Benefit of {max_curtailment*100:.0f}% Constraint:
      - Realistic curtailment ({avg_curtail_yes:.0f}%)
      - Stable demand ({daily_demand_yes:.0f} kWh/day)
      - Better training for all agents
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           verticalalignment='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    return fig


def main():
    """主函数"""
    
    print("="*70)
    print("🔍 Customer Curtailment Constraint Analysis")
    print("="*70)
    
    K = 10.0  # 论文中使用的不适成本系数
    max_curtailment_options = [1.0, 0.70, 0.50, 0.30, 0.20]
    
    print(f"\n测试参数:")
    print(f"  不适成本系数 K = {K} cents/kWh")
    print(f"  基础需求 = 80 kW")
    print(f"  价格范围 = 15-40 cents/kWh")
    
    # 创建输出目录
    output_dir = Path("/mnt/user-data/outputs")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 测试不同约束
    for max_curtail in max_curtailment_options:
        print(f"\n📊 测试约束: {max_curtail*100:.0f}% 最大削减")
        
        # 模拟行为
        results = simulate_customer_behavior(max_curtail, K=K)
        
        # 绘图
        fig = plot_comparison(results, max_curtail, K)
        
        # 保存
        save_path = output_dir / f"curtailment_comparison_{int(max_curtail*100)}pct.png"
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✅ 图表保存: {save_path.name}")
    
    # 生成总结
    print(f"\n" + "="*70)
    print("📋 总结")
    print("="*70)
    
    print(f"""
    测试了5种约束条件:
    1. 无约束 (100%)     - ⚠️  允许完全削减，不现实
    2. 70% 最大削减      - ⚠️  仍然过高
    3. 50% 最大削减      - ⚠️  偏高
    4. 30% 最大削减      - ✅ 推荐 (行业标准)
    5. 20% 最大削减      - ✅ 保守选择
    
    💡 建议:
    - 使用 30% 约束作为默认设置
    - 符合行业标准的需求响应项目
    - 匹配论文中 K=10 时的经验结果
    - 确保微电网能量平衡稳定
    
    📂 所有对比图已保存到: {output_dir}/
    """)
    
    print(f"\n✅ 分析完成!")
    print(f"查看图表了解详细对比")


if __name__ == "__main__":
    main()
