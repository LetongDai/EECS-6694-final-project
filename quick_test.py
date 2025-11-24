#!/usr/bin/env python3
"""
快速测试checkpoint的性能
"""
import torch
import numpy as np
from environment import MicrogridEnv
from agent import Agent

def quick_test(checkpoint_path, num_episodes=200):
    """
    快速测试checkpoint性能
    
    Args:
        checkpoint_path: checkpoint文件路径
        num_episodes: 测试的episode数量
    """
    
    print("="*70)
    print(f"🧪 快速性能测试")
    print("="*70)
    
    # 加载checkpoint
    print(f"\n📥 加载: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    config = checkpoint.get('config', {})
    print(f"✅ 加载成功 (Episode {checkpoint.get('episode', 'Unknown')})")
    
    # 创建环境
    num_envs = 1  # 测试用1个环境
    max_steps = config.get('max_steps', 24)
    
    print(f"\n🌍 创建环境...")
    env = MicrogridEnv(num_envs=num_envs, max_steps=max_steps)
    print(f"✅ 环境创建成功: {env.agent_names}")
    
    # 创建agents并加载权重
    print(f"\n🤖 加载Agents...")
    agents = []
    for i, agent_name in enumerate(env.agent_names):
        agent = Agent(
            obs_size=5,
            act_size=env.act_sizes[agent_name],
            num_agents=env.num_agents,
            max_act_size=2,
            lr=config.get('actor_lr', 1e-4),
            critic_lr=config.get('critic_lr', 1e-3),
            gamma=config.get('gamma', 0.95),
            tau=config.get('tau', 0.01)
        )
        
        # 加载训练好的权重
        agent.actor.load_state_dict(checkpoint['agents'][i])
        agent.target_actor.load_state_dict(checkpoint['agents'][i])
        agent.actor.eval()
        
        agents.append(agent)
    
    print(f"✅ 所有agents加载完成")
    
    # 测试
    print(f"\n🎯 开始测试 ({num_episodes} episodes)...")
    print("-"*70)
    
    episode_rewards_list = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_reward = np.zeros(env.num_agents)
        
        for step in range(max_steps):
            # 收集动作（不加噪声）
            acts_list = []
            for i, agent in enumerate(agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                with torch.no_grad():
                    act = agent.predict(obs_tensor)
                acts_list.append(act)
            
            # 构造动作数组
            max_act_size = 2
            acts = np.zeros((num_envs, env.num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            
            # 执行
            next_obs, rewards, dones = env.step(acts)
            episode_reward += rewards[0]
            obs = next_obs
            
            if dones[0].all():
                break
        
        episode_rewards_list.append(episode_reward)
        
        # 显示进度
        total = episode_reward.sum()
        if (episode + 1) % 5 == 0 or episode == 0:
            print(f"Episode {episode+1:2d}: Total={total:8.2f} | "
                  f"[{', '.join([f'{r:6.1f}' for r in episode_reward])}]")
    
    # 统计结果
    print("\n" + "="*70)
    print("📊 测试结果")
    print("="*70)
    
    episode_rewards_array = np.array(episode_rewards_list)
    avg_rewards = np.mean(episode_rewards_array, axis=0)
    std_rewards = np.std(episode_rewards_array, axis=0)
    avg_total = avg_rewards.sum()
    
    print(f"\n总体表现:")
    print(f"  平均总奖励: {avg_total:8.2f}")
    print(f"  标准差:     {np.std([r.sum() for r in episode_rewards_list]):8.2f}")
    print(f"  最大总奖励: {np.max([r.sum() for r in episode_rewards_list]):8.2f}")
    print(f"  最小总奖励: {np.min([r.sum() for r in episode_rewards_list]):8.2f}")
    
    print(f"\n各Agent表现:")
    print(f"  {'Agent':12s}  {'平均奖励':>10s}  {'标准差':>8s}  {'占比':>6s}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*6}")
    
    for i, name in enumerate(env.agent_names):
        percentage = (avg_rewards[i] / avg_total * 100) if avg_total != 0 else 0
        status = "✅" if avg_rewards[i] > 0 else "⚠️ "
        print(f"  {status}{name:10s}  {avg_rewards[i]:10.2f}  {std_rewards[i]:8.2f}  {percentage:5.1f}%")
    
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*6}")
    print(f"  {'总计':12s}  {avg_total:10.2f}")
    
    # 性能评估
    print(f"\n🎯 性能评级:")
    if avg_total > 50000:
        print(f"  🏆 优秀! (>50k)")
    elif avg_total > 40000:
        print(f"  ✅ 良好  (40k-50k)")
    elif avg_total > 20000:
        print(f"  ⚠️  一般  (20k-40k)")
    else:
        print(f"  ❌ 较差  (<20k)")
    
    # 保存结果
    results = {
        'checkpoint_path': checkpoint_path,
        'episode': checkpoint.get('episode'),
        'test_episodes': num_episodes,
        'avg_total_reward': float(avg_total),
        'avg_rewards_per_agent': avg_rewards.tolist(),
        'std_rewards_per_agent': std_rewards.tolist(),
        'agent_names': env.agent_names,
    }
    
    # 保存到checkpoint中
    checkpoint_updated = checkpoint.copy()
    checkpoint_updated['eval_reward'] = float(avg_total)
    checkpoint_updated['eval_rewards_per_agent'] = avg_rewards.tolist()
    
    # 保存更新后的checkpoint
    output_path = checkpoint_path.replace('.pt', '_evaluated.pt')
    torch.save(checkpoint_updated, output_path)
    print(f"\n💾 已保存评估结果到: {output_path}")
    
    print("\n" + "="*70)
    
    return results, episode_rewards_list

if __name__ == "__main__":
    import sys
    
    # 从命令行获取路径，或使用默认值
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    else:
        checkpoint_path = "debug_checkpoints/debug_maddpg_20251124_031242_final.pt"
        num_episodes = 200
    
    results, episode_rewards = quick_test(checkpoint_path, num_episodes)
    
    print(f"\n💡 提示:")
    print(f"   - 现在可以用 analyze_simple.py 分析 _evaluated.pt 文件了")
    print(f"   - 或者继续用这个脚本测试其他checkpoints")
