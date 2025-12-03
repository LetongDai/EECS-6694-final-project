from environment import MicrogridEnv
import numpy as np

env = MicrogridEnv(num_envs=1, max_steps=24)
obs = env.reset()

# 测试场景1: 内部供应充足
actions = np.array([...])  # 低价bid
next_obs, rewards, dones = env.step(actions)

# 检查
assert env.latest_env_data[0]['main_grid_trade'] == 0  # Main Grid应该不成交

# 测试场景2: 内部供应不足
actions = np.array([...])  # 高价bid导致成交少
next_obs, rewards, dones = env.step(actions)

# 检查
assert env.latest_env_data[0]['main_grid_trade'] > 0  # Main Grid应该成交