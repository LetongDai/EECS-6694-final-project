"""
LLM Reward Generator Configuration
"""

# Safety Configuration
FORBIDDEN_IMPORTS = ['os', 'sys', 'subprocess', 'eval', 'exec', '__import__']
ALLOWED_IMPORTS = ['np', 'numpy']

# Function Signature Configuration
REQUIRED_FUNCTION_NAME = "reward_function"
REQUIRED_PARAMS = ['env_idx', 'auction_results', 'env_data', 'use_policy_incentive']
NUM_REQUIRED_PARAMS = 3
NUM_TOTAL_PARAMS = 4
EXPECTED_RETURN_SHAPE = (5,)

# Prompt Template
PROMPT_TEMPLATE = """Generate a Python reward function for a microgrid multi-agent system.

Policy Description: {policy_description}

EXACT function signature required:
def reward_function(env_idx: int, auction_results: Dict, env_data: Dict, use_policy_incentive: bool = True) -> np.ndarray:

Parameters:
- env_idx: int, current environment index
- auction_results: Dict with keys:
  * 'clearing_price': float, market clearing price in cents/kWh
  * 'allocated_power': dict mapping agent names to power (kW)
    - 'wind': float (positive if sold)
    - 'solar': float (positive if sold)
    - 'diesel': float (positive if sold)
    - 'battery': float (positive=discharge/sell, negative=charge/buy)
  * 'actual_demand': float, customer's actual consumption (kW)
  * 'curtailed_load': float, customer's curtailed load (kW)
  * 'matched_trades': list of trade dicts with 'buyer', 'seller', 'quantity', 'price'
- env_data: Dict with environment state (wind_speed, solar_irradiance, base_load, etc.)
- use_policy_incentive: bool, whether to apply policy incentives/penalties

Return:
- np.ndarray with shape (5,) containing rewards for [wind, solar, diesel, battery, customer]

Example structure:
def reward_function(env_idx: int, auction_results: Dict, env_data: Dict, use_policy_incentive: bool = True) -> np.ndarray:
    clearing_price = auction_results['clearing_price']
    allocated = auction_results['allocated_power']
    rewards = np.zeros(5)

    # Wind agent reward
    wind_power = allocated.get('wind', 0)
    if wind_power > 0:
        rewards[0] = wind_power * clearing_price / 100.0

    # ... other agents

    return rewards

Requirements:
- Only use numpy operations (np.*), no file I/O or dangerous imports
- Access allocated power using allocated.get('agent_name', 0)
- Agent order in return array: [wind, solar, diesel, battery, customer]
- Typical reward components: energy sales revenue, generation costs, policy incentives

Return ONLY the function code, no explanations or markdown."""

# Test Data for Validation
TEST_AUCTION_RESULTS = {
    'clearing_price': 25.0,
    'allocated_power': {
        'wind': 10.0,
        'solar': 15.0,
        'diesel': 5.0,
        'battery': -3.0
    },
    'actual_demand': 50.0,
    'curtailed_load': 5.0,
    'matched_trades': [
        {'buyer': 'customer', 'seller': 'wind', 'quantity': 10.0, 'price': 20.0},
        {'buyer': 'customer', 'seller': 'solar', 'quantity': 15.0, 'price': 22.0}
    ]
}

TEST_ENV_DATA = {
    'wind_speed': 8.0,
    'solar_irradiance': 800.0,
    'base_load': 55.0,
    'time_hour': 12
}

TEST_ENV_IDX = 0
TEST_USE_POLICY = True