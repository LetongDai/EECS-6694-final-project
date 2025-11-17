import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Actor, Critic

class Agent:
    def __init__(self, obs_size, act_size, num_agents, max_act_size=2,
             lr=1e-3, critic_lr=None, gamma=0.95, tau=0.05):
        # networks
        self.actor = Actor(obs_size, act_size)
        self.critic = Critic(obs_size * num_agents, max_act_size * num_agents)
        self.target_actor = Actor(obs_size, act_size)
        self.target_critic = Critic(obs_size * num_agents, max_act_size * num_agents)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        if critic_lr == None:
          critic_lr = lr
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # initialization
        
        self.gamma = gamma
        self.tau = tau
        self.act_size = act_size
        self.polyak_avg()  # initialize the target networks by copying the actor and critic networks
        
    def polyak_avg(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_on(self, sample, idx):
      obs, acts, rewards, next_obs, dones, next_acts = sample
      
      # 🔧 FIX: Flatten observations and actions across agents dimension
      # obs shape: [batch, num_agents, obs_size] -> [batch, num_agents * obs_size]
      batch_size = obs.shape[0]
      num_agents = obs.shape[1]
      
      obs_flat = obs.reshape(batch_size, -1)
      next_obs_flat = next_obs.reshape(batch_size, -1)
      acts_flat = acts.reshape(batch_size, -1)
      next_acts_flat = next_acts.reshape(batch_size, -1)
      
      # Get rewards and dones for this agent (idx)
      rewards_i = rewards[:, idx].unsqueeze(1)  # [batch, 1]
      dones_i = dones[:, idx].unsqueeze(1)      # [batch, 1]

      # Target actions and Q-values
      target_q = self.target_critic(next_obs_flat, next_acts_flat)
      y = rewards_i + self.gamma * (1 - dones_i) * target_q.detach()

      # Critic update
      current_q = self.critic(obs_flat, acts_flat)
      critic_loss = F.mse_loss(current_q, y)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # Actor update
      # Generate new action for this agent
      new_act_i = self.actor(obs[:, idx])  # [batch, act_size]
      
      # Create new actions array with this agent's new action
      new_acts = acts.clone()
      new_acts[:, idx, :self.act_size] = new_act_i
      new_acts_flat = new_acts.reshape(batch_size, -1)
      
      actor_loss = -self.critic(obs_flat, new_acts_flat).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      self.polyak_avg()

    def predict(self, obs):
      return self.target_actor(obs).detach().cpu().numpy()