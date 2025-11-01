import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from replaybuffer import ReplayBuffer
from agent import Agent


class Trainer:
  def __init__(self, envs, agents, obs_size, act_size, buffer_capacity=1e5,
                 lr=1e-3, critic_lr=None, gamma=0.95, tau=0.05):
    # the envs represents an array of parallel environments
    # each environment needs to have two methods: reset and step
    # reset takes no parameters and return an initial state
    # step takes all agent actions as input and return the following: next state, reward, done
    self.envs = envs
    self.agents = agents
    self.replay_buffer = ReplayBuffer(buffer_capacity)

  @torch.no_grad()
  def rollout(self, max_steps):
    obs = envs.reset()
    for _ in range(max_steps):
      acts = acts=np.array([agent(obs[:, i]) for i, agent in enumerate(self.agents)])
      next_obs, rewards, dones = self.envs.step(acts)
      self.replay_buffer.add((obs, acts, rewards, next_obs, dones))
      obs = next_obs
    
  def train_agents(self, rollout_steps, batch_size):
    self.rollout(rollout_steps)
    obs, acts, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)
    next_acts = torch.FloatTensor([agent(next_obs[:, i]) for i, agent in enumerate(self.agents)])
    sample = (obs, acts, rewards, next_obs, dones, next_acts)
    for i, agent in enumerate(self.agents):
      agent.train_on(sample, i)

  def eval_agents(self):
    # TODO
    raise NotImplementedError("not implemented")
