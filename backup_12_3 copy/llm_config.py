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
PROMPT_TEMPLATE = """Generate a Python reward function for a microgrid energy trading system.

Policy Description: {policy_description}

Agent Configuration:
{agent_list}

EXACT function signature required:
def reward_function(env_idx: int, auction_results: Dict, env_data: Dict, use_policy_incentive: bool = True) -> np.ndarray:

Parameters:
- env_idx: int, environment index
- auction_results: Dict with keys:
  * 'clearing_price': float, market clearing price (cents/kWh, needs to divide by 100)
  * 'allocated_power': Dict[str, float] mapping agent names to power amounts (kW)
    Keys are agent names: {agent_names_dict_keys}
    Access using: allocated_power['agent_name'] or allocated_power.get('agent_name', 0.0)
  * 'actual_demand': float, customer's actual consumption (kW)
  * 'curtailed_load': float, amount of demand curtailed (kW)
  * 'matched_trades': list of trade dicts with 'buyer', 'seller', 'quantity' (kW), 'price' (cents/kWh)
- env_data: Dict with keys: 'time_hour', 'wind_speed', 'solar_irradiance', 'base_load' (kW)
- use_policy_incentive: bool, whether to apply policy incentives

Return:
- np.ndarray with shape ({n_agents},) containing rewards for agents in this EXACT order:
  {agent_names}

CRITICAL - Data Access:
- allocated_power is a DICTIONARY, not an array
- DO NOT use allocated_power[0], allocated_power[1], etc.
- CORRECT: allocated_power['WindTurbine_1'] or allocated_power.get('battery', 0.0)
- WRONG: allocated_power[WIND_IDX] or allocated_power[0]

Reward Design Guidelines:
- Generators: reward = power_sold * price - operating_costs (typical range: $0 to $20 per hour)
- Battery: reward = discharge_revenue - charge_cost - degradation_cost (typical: -$5 to $10)
- Customer: reward = -(electricity_cost + discomfort_penalty) (typical: -$10 to -$50)
- Policy incentives should be small bonuses/penalties (e.g., $0.01-0.05 per kW)

Important:
- Power values are in kW, timestep is 1 hour
- Return exactly {n_agents} rewards in the order: {agent_names}
- Use only numpy (np.*), no file I/O or system calls

Return ONLY the function code with NO markdown formatting."""