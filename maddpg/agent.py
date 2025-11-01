import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent:
    def __init__(self, obs_size, act_size, num_agents,
                 lr=1e-3, critic_lr=None, gamma=0.95, tau=0.05):
        # networks
        self.actor = Actor(obs_size, act_size)
        self.critic = Critic(obs_size * num_agents, act_size * num_agents)
        self.target_actor = Actor(obs_size, act_size)
        self.target_critic = Critic(obs_size * num_agents, act_size * num_agents)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        if critic_lr == None:
          critic_lr = lr
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # initialization
        self.polyak_avg()
        self.gamma = gamma
        self.tau = tau

    def polyak_avg(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_on(self, sample, idx):
      obs, acts, rewards, next_obs, dones, next_acts = sample

      # Target actions and Q-values
      target_q = self.target_critic(torch.cat(next_obs, dim=1), torch.cat(next_acts, dim=1))
      y = rewards + self.gamma * target_q.detach()

      # Critic update
      current_q = self.critic(torch.cat(obs, dim=1), torch.cat(acts, dim=1))
      critic_loss = F.mse_loss(current_q, y)
      self.critic_optimizer.zero_grad()
      critic_loss.backward()
      self.critic_optimizer.step()

      # Actor update
      new_acts = acts.clone()
      new_acts[:, idx] = self.actor(obs[:, idx])
      actor_loss = -self.critic(torch.cat(obs, dim=1), torch.cat(new_acts, dim=1)).mean()
      self.actor_optimizer.zero_grad()
      actor_loss.backward()
      self.actor_optimizer.step()

      self.update_targets()

    def predict(self, obs):
      return self.target_actor(obs).detach().cpu().numpy()
