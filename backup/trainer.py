import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replaybuffer import ReplayBuffer
from agent import Agent


class Trainer:
  def __init__(self, envs, agents, obs_size, act_size, buffer_capacity=1e5,
                 lr=1e-3, critic_lr=None, gamma=0.95, tau=0.05):
    self.envs = envs
    self.agents = agents
    self.replay_buffer = ReplayBuffer(buffer_capacity)

  @torch.no_grad()
  def rollout(self, max_steps):
    obs = self.envs.reset()
    num_envs = obs.shape[0]
    num_agents = len(self.agents)
    
    for _ in range(max_steps):
      # 🔧 FIX: Collect actions from each agent
      acts_list = []
      for i, agent in enumerate(self.agents):
        # Convert numpy obs to torch tensor before passing to agent
        obs_tensor = torch.FloatTensor(obs[:, i])
        act = agent.predict(obs_tensor)  # shape: [num_envs, act_size_i]
        acts_list.append(act)
      
      # Get max action size for padding
      max_act_size = max(act.shape[1] for act in acts_list)
      
      # Create padded action array: [num_envs, num_agents, max_act_size]
      acts = np.zeros((num_envs, num_agents, max_act_size))
      for i, act in enumerate(acts_list):
        act_size = act.shape[1]
        acts[:, i, :act_size] = act
      
      next_obs, rewards, dones = self.envs.step(acts)
      self.replay_buffer.add((obs, acts, rewards, next_obs, dones))
      obs = next_obs
    
  def train_agents(self, rollout_steps, batch_size):
    # Only rollout if rollout_steps > 0
    if rollout_steps > 0:
      self.rollout(rollout_steps)
    
    # Check if buffer has enough samples
    if len(self.replay_buffer) < batch_size:
      return
    
    # Sample from replay buffer
    # Shapes after sampling: [batch_size, num_envs, num_agents, feature_size]
    obs, acts, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
    
    # Reshape: [batch_size, num_envs, num_agents, ...] -> [batch_size*num_envs, num_agents, ...]
    batch_size_actual = obs.shape[0]
    num_envs = obs.shape[1]
    num_agents = len(self.agents)
    
    # Flatten batch and env dimensions
    obs = obs.reshape(-1, num_agents, obs.shape[-1])  # [batch*envs, agents, obs_size]
    next_obs = next_obs.reshape(-1, num_agents, next_obs.shape[-1])
    acts = acts.reshape(-1, num_agents, acts.shape[-1])
    rewards = rewards.reshape(-1, num_agents)
    dones = dones.reshape(-1, num_agents)
    
    # Generate next actions for each agent
   # Generate next actions for each agent (with padding)
    batch_size_flat = next_obs.shape[0]
    num_agents = len(self.agents)
    max_act_size = max(agent.act_size for agent in self.agents)

    next_acts = torch.zeros(batch_size_flat, num_agents, max_act_size)
    for i, agent in enumerate(self.agents):
      next_act = agent.target_actor(next_obs[:, i]).detach()
      next_acts[:, i, :agent.act_size] = next_act
        
    sample = (obs, acts, rewards, next_obs, dones, next_acts)
    for i, agent in enumerate(self.agents):
      agent.train_on(sample, i)

  def eval_agents(self):
    # TODO
    raise NotImplementedError("not implemented")