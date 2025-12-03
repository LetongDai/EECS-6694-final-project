import torch
import torch.nn as nn
import torch.nn.functional as F
# 确保这里引用的是刚刚修改过的 models
from models import Actor, Critic 

class Agent:
    def __init__(self, obs_size, act_size, num_agents, max_act_size=2,
             lr=1e-4, critic_lr=1e-3, gamma=0.95, tau=0.01):
        # 注意：这里调整了默认 lr，Transformer 通常需要比 MLP 更小的 lr
        
        # networks
        # Critic 的输入维度是所有 Agent 的 obs + act 总和
        self.actor = Actor(obs_size, act_size)
        self.critic = Critic(obs_size * num_agents, max_act_size * num_agents)
        self.target_actor = Actor(obs_size, act_size)
        self.target_critic = Critic(obs_size * num_agents, max_act_size * num_agents)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        if critic_lr is None:
            critic_lr = lr
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.act_size = act_size
        
        # 硬更新初始化 Target 网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def polyak_avg(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train_on(self, sample, idx):
        obs, acts, rewards, next_obs, dones, next_acts = sample
        
        batch_size = obs.shape[0]
        
        # Flatten observations and actions
        obs_flat = obs.reshape(batch_size, -1)
        next_obs_flat = next_obs.reshape(batch_size, -1)
        acts_flat = acts.reshape(batch_size, -1)
        next_acts_flat = next_acts.reshape(batch_size, -1)
        
        rewards_i = rewards[:, idx].unsqueeze(1)  # [batch, 1]
        dones_i = dones[:, idx].unsqueeze(1)      # [batch, 1]

        # ----------------------------
        # Critic Update
        # ----------------------------
        with torch.no_grad():
            target_q = self.target_critic(next_obs_flat, next_acts_flat)
            # 确保维度匹配，防止广播错误
            y = rewards_i + self.gamma * (1 - dones_i) * target_q

        current_q = self.critic(obs_flat, acts_flat)
        critic_loss = F.mse_loss(current_q, y)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 梯度裁剪：Transformer 训练时梯度容易爆炸，加上这行更安全
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

        # ----------------------------
        # Actor Update
        # ----------------------------
        # 重新计算当前 Agent 的动作
        new_act_i = self.actor(obs[:, idx])
        
        # 组合新动作集
        new_acts = acts.clone()
        new_acts[:, idx, :self.act_size] = new_act_i
        new_acts_flat = new_acts.reshape(batch_size, -1)
        
        # Actor Loss: 最大化 Critic 的评分 (即最小化 -Q)
        actor_loss = -self.critic(obs_flat, new_acts_flat).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

        self.polyak_avg()

    def predict(self, obs):
        # 增加维度检查
        if obs.dim() == 1:
            obs = obs.unsqueeze(0) # [1, obs_size]
        
        return self.target_actor(obs).detach().cpu().numpy()