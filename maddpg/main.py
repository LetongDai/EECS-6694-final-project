from trainer import Trainer

envs = None
agents = None
obs_size = 0
act_size = 0

epochs = 1000
rollout_steps = 500
batch_size = 16

if __name__ == "__main__":
  trainer = Trainer(envs, agents, obs_size, act_size)
  for i in range(epochs):
    trainer.train_agents(self, rollout_steps, batch_size)
