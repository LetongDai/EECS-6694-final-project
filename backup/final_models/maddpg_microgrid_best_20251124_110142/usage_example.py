#!/usr/bin/env python3
"""Usage Example for maddpg_microgrid_best"""

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
    
    print(f"\nTotal Reward: {episode_reward.sum():.2f}")
    for i, name in enumerate(env.agent_names):
        print(f"  {name}: {episode_reward[i]:.2f}")


if __name__ == "__main__":
    main()
