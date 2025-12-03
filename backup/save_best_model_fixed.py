#!/usr/bin/env python3
"""
完整保存和记录最佳模型 - 修复版
Save Best Model with Complete Documentation - Fixed Version
"""
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime
import numpy as np


def safe_float_format(value, format_spec=".2f", default="N/A"):
    """安全的浮点数格式化"""
    if value is None:
        return default
    try:
        return f"{float(value):{format_spec}}"
    except (ValueError, TypeError):
        return default


def save_best_model_complete(
    checkpoint_path,
    stats_path=None,
    save_dir="final_models",
    model_name="maddpg_microgrid_best"
):
    """
    完整保存模型
    """
    
    print("=" * 70)
    print("💾 保存最佳模型")
    print("=" * 70)
    
    # 创建保存目录
    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = save_path / f"{model_name}_{timestamp}"
    model_dir.mkdir(exist_ok=True)
    
    print(f"\n📁 保存目录: {model_dir}")
    
    # 1. 保存checkpoint
    print(f"\n1️⃣  保存Checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    checkpoint_save_path = model_dir / "model_checkpoint.pt"
    torch.save(checkpoint, checkpoint_save_path)
    print(f"   ✅ Checkpoint: {checkpoint_save_path.name}")
    
    # 2. 保存配置
    print(f"\n2️⃣  保存配置...")
    config_save_path = model_dir / "config.json"
    clean_config = {}
    for key, value in config.items():
        try:
            json.dumps(value)
            clean_config[key] = value
        except:
            clean_config[key] = str(value)
    
    with open(config_save_path, 'w') as f:
        json.dump(clean_config, f, indent=2)
    print(f"   ✅ 配置文件: {config_save_path.name}")
    
    # 3. 保存统计
    print(f"\n3️⃣  保存训练统计...")
    stats_data = None
    if stats_path and Path(stats_path).exists():
        with open(stats_path, 'r') as f:
            stats_data = json.load(f)
        
        stats_save_path = model_dir / "training_stats.json"
        with open(stats_save_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"   ✅ 训练统计: {stats_save_path.name}")
    else:
        print(f"   ⚠️  未找到统计文件")
    
    # 4. 生成性能报告
    print(f"\n4️⃣  生成性能报告...")
    performance_report = generate_performance_report(checkpoint, stats_data)
    
    report_path = model_dir / "performance_report.json"
    with open(report_path, 'w') as f:
        json.dump(performance_report, f, indent=2)
    print(f"   ✅ 性能报告: {report_path.name}")
    
    # 5. 生成README
    print(f"\n5️⃣  生成README文档...")
    readme_content = generate_readme_safe(model_name, checkpoint, performance_report, stats_data)
    
    readme_path = model_dir / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    print(f"   ✅ README: {readme_path.name}")
    
    # 6. 生成使用示例
    print(f"\n6️⃣  生成使用示例...")
    example_code = generate_usage_example(model_name, clean_config)
    
    example_path = model_dir / "usage_example.py"
    with open(example_path, 'w') as f:
        f.write(example_code)
    print(f"   ✅ 示例代码: {example_path.name}")
    
    # 7. 复制图表
    print(f"\n7️⃣  复制训练图表...")
    plot_dir = Path("debug_plots")
    if plot_dir.exists():
        plots_save_dir = model_dir / "plots"
        plots_save_dir.mkdir(exist_ok=True)
        
        exp_name = config.get('exp_name', '')
        if exp_name:
            for plot_file in plot_dir.glob(f"*{exp_name.split('_')[-1]}*.png"):
                shutil.copy(plot_file, plots_save_dir / plot_file.name)
                print(f"   ✅ 复制图表: {plot_file.name}")
        
        summary_files = list(plot_dir.glob("summary*.png"))
        if summary_files:
            for f in summary_files[:1]:
                shutil.copy(f, plots_save_dir / "training_summary.png")
                print(f"   ✅ 复制图表: training_summary.png")
    
    # 8. 创建轻量级版本
    print(f"\n8️⃣  创建简化版模型...")
    lightweight_checkpoint = {
        'agents': checkpoint['agents'],
        'config': clean_config,
        'performance': performance_report
    }
    
    lightweight_path = model_dir / "model_weights_only.pt"
    torch.save(lightweight_checkpoint, lightweight_path)
    print(f"   ✅ 轻量级模型: {lightweight_path.name}")
    
    # 完成
    print(f"\n" + "=" * 70)
    print(f"✅ 模型保存完成！")
    print(f"=" * 70)
    
    print(f"\n📦 保存内容:")
    print(f"   📂 {model_dir}/")
    print(f"      ├── model_checkpoint.pt")
    print(f"      ├── model_weights_only.pt")
    print(f"      ├── config.json")
    print(f"      ├── training_stats.json")
    print(f"      ├── performance_report.json")
    print(f"      ├── README.md")
    print(f"      ├── usage_example.py")
    print(f"      └── plots/")
    
    print(f"\n💡 下一步:")
    print(f"   1. 查看README: {readme_path}")
    print(f"   2. 测试模型: python {example_path}")
    print(f"   3. 分享模型: 打包 {model_dir}/ 目录")
    
    return model_dir


def generate_performance_report(checkpoint, stats_data):
    """生成性能报告"""
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'episode': checkpoint.get('episode'),
            'eval_reward': checkpoint.get('eval_reward'),
        }
    }
    
    if 'eval_rewards_per_agent' in checkpoint:
        rewards = np.array(checkpoint['eval_rewards_per_agent'])
        agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
        
        report['agent_performance'] = {}
        for i, name in enumerate(agent_names):
            if i < len(rewards):
                report['agent_performance'][name] = {
                    'reward': float(rewards[i]),
                    'percentage': float(rewards[i] / rewards.sum() * 100) if rewards.sum() != 0 else 0
                }
    
    if stats_data:
        episode_rewards = stats_data.get('episode_rewards', [])
        if episode_rewards:
            rewards_array = np.array(episode_rewards)
            n = len(rewards_array)
            
            report['training_progress'] = {
                'total_episodes': n,
                'final_avg_reward': float(np.mean(rewards_array[-20:])) if n >= 20 else float(np.mean(rewards_array)),
                'best_reward': float(np.max(rewards_array)),
                'worst_reward': float(np.min(rewards_array)),
                'improvement': {
                    'early_avg': float(np.mean(rewards_array[:n//5])) if n >= 5 else float(rewards_array[0]),
                    'late_avg': float(np.mean(rewards_array[4*n//5:])) if n >= 5 else float(rewards_array[-1]),
                    'percentage': float(((np.mean(rewards_array[4*n//5:]) - np.mean(rewards_array[:n//5])) / 
                                        abs(np.mean(rewards_array[:n//5])) * 100)) if n >= 5 else 0
                }
            }
    
    return report


def generate_readme_safe(model_name, checkpoint, performance_report, stats_data):
    """生成README文档 - 安全版本"""
    
    config = checkpoint.get('config', {})
    
    # 安全获取奖励数据
    eval_reward = checkpoint.get('eval_reward')
    if eval_reward is not None:
        eval_reward_cents = safe_float_format(eval_reward, ".2f")
        eval_reward_dollar = safe_float_format(eval_reward/100, ".2f", "$N/A")
        if eval_reward_dollar != "$N/A":
            eval_reward_dollar = f"${eval_reward_dollar}"
    else:
        eval_reward_cents = "N/A"
        eval_reward_dollar = "N/A"
    
    agent_perf = performance_report.get('agent_performance', {})
    training_prog = performance_report.get('training_progress', {})
    
    # 构建README
    lines = []
    lines.append(f"# {model_name.replace('_', ' ').title()}")
    lines.append("")
    lines.append(f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 📊 Model Performance")
    lines.append("")
    lines.append("### Overall Performance")
    lines.append(f"- **Total Reward**: {eval_reward_cents} cents ≈ {eval_reward_dollar}")
    lines.append(f"- **Episodes Trained**: {checkpoint.get('episode', 'N/A')}")
    
    if training_prog:
        improvement = training_prog.get('improvement', {})
        lines.append(f"- **Learning Progress**: {safe_float_format(improvement.get('percentage', 0), '.1f')}% improvement")
        lines.append(f"- **Best Reward**: {safe_float_format(training_prog.get('best_reward', 0), '.2f')}")
        lines.append(f"- **Final Average**: {safe_float_format(training_prog.get('final_avg_reward', 0), '.2f')}")
    
    lines.append("")
    lines.append("### Agent Performance Breakdown")
    lines.append("")
    lines.append("| Agent | Reward (cents) | Reward ($) | Contribution |")
    lines.append("|-------|----------------|------------|--------------|")
    
    for agent_name in ['wind', 'solar', 'diesel', 'battery', 'customer']:
        if agent_name in agent_perf:
            perf = agent_perf[agent_name]
            reward = perf['reward']
            percentage = perf['percentage']
            reward_cents = safe_float_format(reward, ".2f")
            reward_dollar = safe_float_format(reward/100, ".2f")
            percentage_str = safe_float_format(percentage, ".1f")
            lines.append(f"| {agent_name.capitalize():8s} | {reward_cents:>14s} | ${reward_dollar:>10s} | {percentage_str:>11s}% |")
    
    lines.append("")
    lines.append("### Performance Ratings")
    lines.append("")
    
    # 评级
    if eval_reward is not None:
        total = eval_reward
    elif training_prog and 'final_avg_reward' in training_prog:
        total = training_prog['final_avg_reward']
    else:
        total = 0
    
    if total > 50000:
        rating = "🏆 Excellent (>50k)"
    elif total > 40000:
        rating = "✅ Very Good (40k-50k)"
    elif total > 20000:
        rating = "⚠️  Good (20k-40k)"
    else:
        rating = "❌ Needs Improvement (<20k)"
    
    lines.append(f"**Overall Rating**: {rating}")
    lines.append("")
    
    # Agent评级
    for agent_name, emoji in [('wind', '🌬️'), ('solar', '☀️'), ('diesel', '⛽'), 
                               ('battery', '🔋'), ('customer', '👥')]:
        if agent_name in agent_perf:
            reward = agent_perf[agent_name]['reward']
            if agent_name in ['wind', 'solar', 'diesel']:
                status = "✅ Profitable" if reward > 5000 else "⚠️ Low Profit"
            elif agent_name == 'battery':
                status = "✅ Positive" if reward > 0 else "❌ Negative"
            else:
                status = "✅ Cost Savings" if reward > -500 else "⚠️ High Cost"
            
            lines.append(f"- {emoji} **{agent_name.capitalize()}**: {status} ({safe_float_format(reward, '.2f')} cents)")
    
    lines.append("")
    lines.append("## ⚙️  Training Configuration")
    lines.append("")
    lines.append("### Hyperparameters")
    lines.append("```json")
    lines.append("{")
    lines.append(f'  "actor_lr": {config.get("actor_lr", "N/A")},')
    lines.append(f'  "critic_lr": {config.get("critic_lr", "N/A")},')
    lines.append(f'  "gamma": {config.get("gamma", "N/A")},')
    lines.append(f'  "tau": {config.get("tau", "N/A")},')
    lines.append(f'  "batch_size": {config.get("batch_size", "N/A")},')
    lines.append(f'  "buffer_capacity": {config.get("buffer_capacity", "N/A")}')
    lines.append("}")
    lines.append("```")
    lines.append("")
    
    lines.append("## 🚀 Quick Start")
    lines.append("")
    lines.append("See `usage_example.py` for complete testing code.")
    lines.append("")
    lines.append("---")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)


def generate_usage_example(model_name, config):
    """生成使用示例"""
    
    return f'''#!/usr/bin/env python3
"""Usage Example for {model_name}"""

import torch
import numpy as np
from environment import MicrogridEnv
from agent import Agent


def main():
    # Load model
    checkpoint = torch.load('model_checkpoint.pt', weights_only=False)
    
    # Create environment
    env = MicrogridEnv(num_envs=1, max_steps=24)
    
    # Load agents
    agents = []
    for i, agent_name in enumerate(env.agent_names):
        agent = Agent(
            obs_size=5,
            act_size=env.act_sizes[agent_name],
            num_agents=env.num_agents,
            max_act_size=2,
            lr=1e-4, critic_lr=1e-3, gamma=0.95, tau=0.01
        )
        agent.actor.load_state_dict(checkpoint['agents'][i])
        agents.append(agent)
    
    print("✅ Model loaded successfully!")
    
    # Run test
    obs = env.reset()
    episode_reward = np.zeros(env.num_agents)
    
    for step in range(24):
        acts_list = []
        for i, agent in enumerate(agents):
            obs_tensor = torch.FloatTensor(obs[:, i])
            with torch.no_grad():
                act = agent.predict(obs_tensor)
            acts_list.append(act)
        
        acts = np.zeros((1, env.num_agents, 2))
        for i, act in enumerate(acts_list):
            acts[:, i, :act.shape[1]] = act
        
        next_obs, rewards, dones = env.step(acts)
        episode_reward += rewards[0]
        obs = next_obs
    
    print(f"\\nTotal Reward: {{episode_reward.sum():.2f}}")
    for i, name in enumerate(env.agent_names):
        print(f"  {{name}}: {{episode_reward[i]:.2f}}")


if __name__ == "__main__":
    main()
'''


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        stats_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        checkpoint_path = "debug_checkpoints/debug_maddpg_20251124_045324_final_evaluated.pt"
        stats_path = "debug_logs/debug_maddpg_20251124_045324_stats.json"
    
    try:
        model_dir = save_best_model_complete(
            checkpoint_path=checkpoint_path,
            stats_path=stats_path,
            model_name="maddpg_microgrid_best"
        )
        print(f"\n🎉 Success! Model saved to: {model_dir}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
