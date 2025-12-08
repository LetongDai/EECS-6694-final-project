import torch
import numpy as np
from replaybuffer import ReplayBuffer


class Trainer:
    def __init__(self, envs, agents, config, buffer_capacity=1e5):
        self.envs = envs
        self.agents = agents
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.config=config
        self.current_noise = self.config.noise_scale

    @torch.no_grad()
    def rollout(self, add_noise=True):
        obs = self.envs.reset()
        num_envs = obs.shape[0]
        num_agents = len(self.agents)
        agent_names = self.envs.agent_names
        episode_reward = np.zeros(num_agents)
        actions_collected = []

        # statistics, doesn't used in training
        daily_stats = {
            'bids_sum': {name: 0.0 for name in agent_names if name != 'customer'},
            'gen_sum': {name: 0.0 for name in agent_names if name != 'customer'}
        }
        grid_import_total = 0.0
        grid_export_total = 0.0
        clearing_prices = []

        for step in range(self.config.max_steps):
            acts_list = []
            for i, agent in enumerate(self.agents):
                obs_tensor = torch.FloatTensor(obs[:, i])
                act = agent.predict(obs_tensor)  # shape: [num_envs, act_size_i]
                if add_noise:
                    noise = np.random.normal(0, self.current_noise, act.shape)
                    act = np.clip(act + noise, -1, 1)
                acts_list.append(act)

            # Combine actions
            max_act_size = max(act.shape[1] for act in acts_list)
            acts = np.zeros((num_envs, num_agents, max_act_size))
            for i, act in enumerate(acts_list):
                acts[:, i, :act.shape[1]] = act
            actions_collected.append([act.copy() for act in acts_list])

            next_obs, rewards, dones, infos = self.envs.step(acts)
            self.replay_buffer.add((obs, acts, rewards, next_obs, dones))

            info = infos[0]

            for name in agent_names:
                if name != 'customer':
                    if name in info['bids']:
                        daily_stats['bids_sum'][name] += info['bids'][name]
                    if name in info['allocated_power']:
                        daily_stats['gen_sum'][name] += info['allocated_power'][name]

            grid_import_total += info['grid_import']
            grid_export_total += info['grid_export']

            if 'clearing_price' in info:
                clearing_prices.append(info['clearing_price'])

            episode_reward += rewards[0]
            obs = next_obs

            if dones[0].all():
                break

        self.current_noise = max(self.config.noise_min,
                                 self.current_noise * self.config.noise_decay)

        actions_stats = actions_collected
        avg_price = np.mean(clearing_prices) if clearing_prices else 0
        avg_bids = {k: v / (step + 1) for k, v in daily_stats['bids_sum'].items()}
        total_gen = daily_stats['gen_sum']

        return (episode_reward, step + 1, actions_stats,
                grid_import_total, grid_export_total,
                avg_price, avg_bids, total_gen)
    
    def train_agents(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return

        # Shapes: [batch_size, num_envs, num_agents, feature_size]
        obs, acts, rewards, next_obs, dones = self.replay_buffer.sample(batch_size)

        num_agents = len(self.agents)

        # Flatten batch and env dimensions
        obs = obs.reshape(-1, num_agents, obs.shape[-1])  # [batch*envs, agents, obs_size]
        next_obs = next_obs.reshape(-1, num_agents, next_obs.shape[-1])
        acts = acts.reshape(-1, num_agents, acts.shape[-1])
        rewards = rewards.reshape(-1, num_agents)
        dones = dones.reshape(-1, num_agents)

        # Generate next actions for each agent (with padding)
        batch_size_flat = next_obs.shape[0]
        num_agents = len(self.agents)
        max_act_size = max(agent.act_size for agent in self.agents)

        next_acts = torch.zeros(batch_size_flat, num_agents, max_act_size)
        for i, agent in enumerate(self.agents):
            next_act = agent.target_actor(next_obs[:, i]).detach()
            next_acts[:, i, :agent.act_size] = next_act

        # Training
        sample = (obs, acts, rewards, next_obs, dones, next_acts)
        for i, agent in enumerate(self.agents):
            agent.train_on(sample, i)