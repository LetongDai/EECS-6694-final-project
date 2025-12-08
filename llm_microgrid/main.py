import torch
import json
from pathlib import Path
from datetime import datetime

from environment import MicrogridEnv
from agent import Agent 
from trainer import Trainer
from llm_reward_generator import LLMRewardGenerator
from utility import DetailedLogger, Plotter


class DebugConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        self._load_from_dict(config_dict)

        Path(self.save_dir).mkdir(exist_ok=True)
        Path(self.log_dir).mkdir(exist_ok=True)
        Path(self.plot_dir).mkdir(exist_ok=True)

    def _load_from_dict(self, config_dict):
        """load user configuration"""
        # LLM
        llm = config_dict.get('llm', {})
        self.llm_enabled = llm.get('enabled', True)
        self.llm_api_key = llm.get('api_key', 'your_anthropic_api_key')
        self.llm_model = llm.get('model', 'claude-sonnet-4-20250514')
        self.llm_max_tokens = llm.get('max_tokens', 2000)
        self.llm_provider = llm.get('provider', 'gemini')
        self.llm_policy_description = llm.get('policy_description', '')

        # Experiment
        exp = config_dict.get('experiment', {})
        exp_name_base = exp.get('exp_name', 'debug_maddpg')
        self.exp_name = f"{exp_name_base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = exp.get('save_dir', 'debug_checkpoints')
        self.log_dir = exp.get('log_dir', 'debug_logs')
        self.plot_dir = exp.get('plot_dir', 'debug_plots')

        # Environment
        env = config_dict.get('environment', {})
        self.num_envs = env.get('num_envs', 2)
        self.max_steps = env.get('max_steps', 24)

        # Training
        train = config_dict.get('training', {})
        self.total_episodes = train.get('total_episodes', 200)
        self.batch_size = train.get('batch_size', 128)
        self.warmup_episodes = train.get('warmup_episodes', 20)
        self.actor_lr = train.get('actor_lr', 5e-5)
        self.critic_lr = train.get('critic_lr', 5e-4)
        self.gamma = train.get('gamma', 0.95)
        self.tau = train.get('tau', 0.01)
        self.buffer_capacity = train.get('buffer_capacity', 50000)

        # Exploration
        explore = config_dict.get('exploration', {})
        self.noise_scale = explore.get('noise_scale', 0.3)
        self.noise_decay = explore.get('noise_decay', 0.996)
        self.noise_min = explore.get('noise_min', 0.02)

        # Logging
        log = config_dict.get('logging', {})
        self.log_interval = log.get('log_interval', 1)
        self.plot_interval = log.get('plot_interval', 20)
        self.save_interval = log.get('save_interval', 100)


class DetailedTrainer:
    def __init__(self, env, agents, config):
        self.env = env
        self.agents = agents
        self.config = config

        self.trainer = Trainer(env, agents, config, buffer_capacity=config.buffer_capacity)

        self.agent_names = env.agent_names
        self.num_agents = env.num_agents
        
        self.logger = DetailedLogger(config, env.agent_names)
        self.plotter = Plotter(config, env.agent_names)
    
    def train(self):
        self.logger.log("\n" + "="*70)
        self.logger.log("Training begin")
        self.logger.log("="*70)

        # Warmup
        for ep in range(self.config.warmup_episodes):
            self.trainer.rollout()
            if ep % 5 == 0: self.logger.log(f"Warmup {ep}/{self.config.warmup_episodes}")
        self.logger.log(f"Warmup {self.config.warmup_episodes}/{self.config.warmup_episodes}")

        # Main Loop
        for episode in range(1, self.config.total_episodes + 1):
            # Rollout
            (ep_rewards, ep_steps, acts_stats,
             g_imp, g_exp, avg_price,
             avg_bids, total_gen) = self.trainer.rollout()
            
            # Train
            for _ in range(ep_steps):
                self.trainer.train_agents(self.config.batch_size)
            
            # Log
            ep_data = {
                'total_reward': ep_rewards.sum(),
                'rewards_per_agent': ep_rewards,
                'grid_import': g_imp,
                'grid_export': g_exp,
                'clearing_price': avg_price,
                'avg_bids': avg_bids,
                'total_gen': total_gen
            }
            self.logger.log_episode(episode, ep_data)
            
            # Plot
            if episode % self.config.plot_interval == 0:
                self.logger.log(f"\nGenerating plots")
                p1 = self.plotter.plot_training_status(self.logger, episode)
                p2 = self.plotter.plot_grid_status(self.logger, episode)
                self.logger.log(f"  {p1}")
                self.logger.log(f"  {p2}")
            
            if episode % self.config.save_interval == 0:
                self.save_checkpoint(episode)
        
        # End
        self.logger.log("="*70)
        self.logger.log("Training complete")
        self.logger.save_stats()
        final_plot = self.plotter.create_final_summary(self.logger)
        self.logger.log(f"Final summary: {final_plot}")
        self.logger.close()

    def save_checkpoint(self, episode):
        path = f"{self.config.save_dir}/{self.config.exp_name}_ep{episode}.pt"
        torch.save({
            'episode': episode,
            'agents': [a.actor.state_dict() for a in self.agents],
            'config': vars(self.config)
        }, path)
        self.logger.log(f"Checkpoint saved: {path}")


def setup_llm_reward(env, policy_description, api_key, model, max_tokens, microgrid_config):
    llm_gen = LLMRewardGenerator(api_key=api_key, model=model, max_tokens=max_tokens, config_path=microgrid_config)

    policy_text = policy_description

    try:
        print("Generating reward function from policy description")
        reward_code = llm_gen.generate_reward_code(policy_text)
        print("\nGenerated reward code:")
        print(reward_code)

        print("\nValidating and compiling")
        reward_fn = llm_gen.validate_and_compile(reward_code)
        env.reward_function = reward_fn
        print("Custom reward function loaded successfully\n")
        return True
    except Exception as e:
        print(f"Failed to generate reward: {e}")
        print("Using default reward function\n")
        return False


def main(microgrid_config, user_config):
    print(f"Loading configuration from {user_config}")
    config = DebugConfig(user_config)

    env = MicrogridEnv(microgrid_config, num_envs=config.num_envs, max_steps=config.max_steps)

    if config.llm_enabled:
        setup_llm_reward(
            env,
            api_key=config.llm_api_key,
            model=config.llm_model,
            max_tokens=config.llm_max_tokens,
            policy_description=config.llm_policy_description,
            microgrid_config=microgrid_config
        )

    print(f"Environment initialized with {env.num_agents} agents.")

    # Init Agents
    agents = []
    max_act_size = max(env.act_sizes.values())

    for i, name in enumerate(env.agent_names):
        agent = Agent(
            obs_size=5,  # padding to the same dimension
            act_size=env.act_sizes[env.agent_types[i]],
            num_agents=env.num_agents,
            max_act_size=max_act_size,
            lr=config.actor_lr,
            critic_lr=config.critic_lr,
            gamma=config.gamma,
            tau=config.tau
        )
        agents.append(agent)

    trainer = DetailedTrainer(env, agents, config)
    trainer.train()


if __name__ == "__main__":
    import sys
    user_config = sys.argv[1] if len(sys.argv) > 1 else "user_config.json"
    microgrid_config = sys.argv[2] if len(sys.argv) > 2 else "microgrid_config.json"
    main(microgrid_config, user_config)