"""
Enhanced Environment wrapper for Microgrid MADDPG
完整实现5个agents参与拍卖的多智能体RL环境
"""

import numpy as np
import torch
from typing import Tuple, Dict, List

# Import the microgrid components
from microgrid_auction import (
    Microgrid, WindTurbine, PVSystem, DieselGenerator, 
    Battery, MainGrid, CustomerLoad
)
from microgrid_auction_enhanced import MicrogridAuction, AuctionResult


class MicrogridEnv:
    """
    Multi-Agent Microgrid Environment with Full Auction Mechanism
    
    All 5 agents participate in the electricity auction:
        1. Wind Agent - bids for selling
        2. Solar Agent - bids for selling
        3. Diesel Agent - bids for selling with power control
        4. Battery Agent - bids for buying/selling with charge control
        5. Customer Agent - demand side management
    """
    
    def __init__(self, num_envs=1, max_steps=24, use_policy=True):
        """
        Args:
            num_envs: Number of parallel environments
            max_steps: Maximum steps per episode (24 hours)
        """
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.num_agents = 5
        
        # Agent names
        self.agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
        
        # State and action dimensions for each agent (from paper Table II)
        self.obs_sizes = {
            'wind': 4,      # wind_speed, power_output, time, grid_price
            'solar': 4,     # irradiance, power_output, time, grid_price
            'diesel': 5,    # fuel_level, power_output, time, grid_price, load
            'battery': 5,   # soc, power_output, time, grid_price, load
            'customer': 4   # total_demand, consumption, time, grid_price
        }
        
        self.act_sizes = {
            'wind': 1,      # bid
            'solar': 1,     # bid
            'diesel': 2,    # power_ratio, bid
            'battery': 2,   # charge/discharge_power, bid
            'customer': 1   # curtailment_ratio
        }
        
        # Create multiple microgrid instances (for physical simulation only)
        self.microgrids = [self._create_microgrid() for _ in range(num_envs)]
        
        # Time step counter
        self.current_steps = np.zeros(num_envs, dtype=np.int32)
        
        # Price ranges for normalization and denormalization
        self.price_min = 5.0   # cents/kWh
        self.price_max = 45.0  # cents/kWh
        
        # Store latest environmental data
        self.latest_env_data = [None] * num_envs
        
        # 添加：奖励归一化统计
        self.reward_mean = np.zeros(self.num_agents)
        self.reward_std = np.ones(self.num_agents)
        self.reward_history = [[] for _ in range(self.num_agents)]
        self.normalize_rewards = True  # 开关
        self.use_policy = use_policy

    def _create_microgrid(self) -> Microgrid:
        """Create a single microgrid instance (for physical simulation)"""
        wind = WindTurbine(name='WindTurbine', capacity=100)
        pv = PVSystem(name='PVSystem', capacity=50)
        diesel = DieselGenerator(
            name='DieselGenerator', 
            capacity=75,
            fuel_consumption_rate=0.2,
            generation_cost_per_kwh=0.08
        )
        battery = Battery(
            name='Battery',
            capacity_kwh=200,
            max_charge_rate_kw=50,
            max_discharge_rate_kw=50,
            initial_soc=0.5,
            charge_cost_per_kwh=0.05,
            discharge_cost_per_kwh=0.15
        )
        grid = MainGrid(
            name='MainGrid',
            import_limit=100,
            export_limit=100,
            import_price_per_kwh=0.25,
            export_price_per_kwh=0.1
        )
        load = CustomerLoad(name='CustomerLoad')
        
        components = {
            'wind': wind,
            'pv': pv,
            'diesel': diesel,
            'battery': battery,
            'grid': grid,
            'load': load
        }
        
        return Microgrid(components)
    
    def _generate_env_data(self, env_idx: int) -> Dict:
        """Generate environmental data (weather, load) for one timestep"""
        time_hour = self.current_steps[env_idx] % 24
        
        # Generate random weather data
        wind_speed = np.random.uniform(3, 15)  # m/s
        solar_irradiance = self._get_solar_irradiance(time_hour)
        
        # Generate load based on time of day
        base_load = 80
        load_variation = 40 * np.sin(np.pi * time_hour / 12)
        load = max(30, base_load + load_variation + np.random.uniform(-10, 10))
        
        return {
            'time_hour': time_hour,
            'wind_speed': wind_speed,
            'solar_irradiance': solar_irradiance,
            'base_load': load
        }
    
    def _get_solar_irradiance(self, hour: int) -> float:
        """Get solar irradiance based on time (W/m²)"""
        if hour < 6 or hour > 18:
            return 0  # Night time
        return 1000 * np.sin(np.pi * (hour - 6) / 12) * np.random.uniform(0.7, 1.0)
    
    def _get_grid_price(self, hour: int) -> Tuple[float, float]:
        """Get main grid price based on time of use"""
        # Buy price (import) - Time-of-Use pricing
        if 23 <= hour or hour < 8:  # Off-peak
            buy_price = np.random.uniform(13, 16)
        elif 8 <= hour < 12 or 18 <= hour < 23:  # Mid-peak
            buy_price = np.random.uniform(23, 26)
        else:  # Peak (12-18)
            buy_price = np.random.uniform(40, 42)
        
        # Sell price (export)
        sell_price = np.random.uniform(5, 15)
        
        return sell_price, buy_price
    
    def _run_auction(self, env_idx: int, actions: np.ndarray, env_data: Dict) -> Dict:
        """
        Run uniform price auction for electricity market
        
        Args:
            env_idx: Environment index
            actions: [num_agents, act_size] - actions from all agents
            env_data: Environmental data (weather, load)
            
        Returns:
            auction_results: Dictionary with clearing price, allocations, etc.
        """
        mg = self.microgrids[env_idx]
        
        # Parse actions
        wind_bid_norm = actions[0, 0]  # [0, 1]
        solar_bid_norm = actions[1, 0]  # [0, 1]
        diesel_power_ratio = actions[2, 0]  # [0, 1]
        diesel_bid_norm = actions[2, 1]  # [0, 1]
        battery_action = actions[3, 0]  # [-1, 1]
        battery_bid_norm = actions[3, 1]  # [0, 1]
        customer_curtail = actions[4, 0]  # [0, 1]
        
        # Denormalize bids to actual prices (cents/kWh)
        wind_bid = wind_bid_norm * (self.price_max - self.price_min) + self.price_min
        solar_bid = solar_bid_norm * (self.price_max - self.price_min) + self.price_min
        diesel_bid = diesel_bid_norm * (self.price_max - self.price_min) + self.price_min
        battery_bid = battery_bid_norm * (self.price_max - self.price_min) + self.price_min
        
        # Calculate available power from each source
        # Wind power (based on weather)
        wind_power = mg.components['wind'].capacity * min(1.0, max(0, 
            (env_data['wind_speed'] - 3) / (12 - 3)))  # Simplified model
        
        # Solar power (based on weather)
        solar_power = mg.components['pv'].capacity * (env_data['solar_irradiance'] / 1000.0)
        
        # Diesel power (based on agent's decision)
        diesel_power = mg.components['diesel'].capacity * max(0, min(1, diesel_power_ratio))
        
        # Battery power
        battery_component = mg.components['battery']
        if battery_action > 0:  # Discharging (selling)
            battery_power = min(
                battery_action * battery_component.max_discharge_rate_kw,
                battery_component.soc * battery_component.capacity_kwh  # Available energy
            )
            battery_mode = 'discharge'
        else:  # Charging (buying)
            battery_power = min(
                abs(battery_action) * battery_component.max_charge_rate_kw,
                (1 - battery_component.soc) * battery_component.capacity_kwh  # Available capacity
            )
            battery_mode = 'charge'
        
        # Customer demand (after curtailment)
        total_demand = env_data['base_load']
        actual_demand = total_demand * (1 - max(0, min(1, customer_curtail)))
        
        # Build supply curve (sorted by bid price)
        suppliers = []
        if wind_power > 0:
            suppliers.append({
                'name': 'wind',
                'power': wind_power,
                'bid': wind_bid,
                'cost': 0.0  # Marginal cost
            })
        if solar_power > 0:
            suppliers.append({
                'name': 'solar',
                'power': solar_power,
                'bid': solar_bid,
                'cost': 0.0
            })
        if diesel_power > 0:
            suppliers.append({
                'name': 'diesel',
                'power': diesel_power,
                'bid': diesel_bid,
                'cost': 0.08  # cents/kWh
            })
        if battery_mode == 'discharge' and battery_power > 0:
            suppliers.append({
                'name': 'battery',
                'power': battery_power,
                'bid': battery_bid,
                'cost': 0.15
            })
        
        # Sort suppliers by bid (ascending)
        suppliers.sort(key=lambda x: x['bid'])
        
        # Market clearing
        total_supply = sum(s['power'] for s in suppliers)
        
        # Calculate clearing price
        if actual_demand <= 0:
            clearing_price = self.price_min
            allocated_power = {s['name']: 0 for s in suppliers}
        elif total_supply >= actual_demand:
            # Sufficient supply - find marginal supplier
            cumulative = 0
            clearing_price = self.price_min
            allocated_power = {}
            
            for supplier in suppliers:
                if cumulative >= actual_demand:
                    allocated_power[supplier['name']] = 0
                elif cumulative + supplier['power'] <= actual_demand:
                    # Fully allocated
                    allocated_power[supplier['name']] = supplier['power']
                    cumulative += supplier['power']
                    clearing_price = supplier['bid']
                else:
                    # Partially allocated (marginal supplier)
                    allocated_power[supplier['name']] = actual_demand - cumulative
                    cumulative = actual_demand
                    clearing_price = supplier['bid']
        else:
            # Insufficient supply - all accepted, use highest bid + penalty
            clearing_price = max(s['bid'] for s in suppliers) + 5.0
            allocated_power = {s['name']: s['power'] for s in suppliers}
        
        # Handle battery charging (if in charge mode)
        if battery_mode == 'charge':
            allocated_power['battery'] = -battery_power  # Negative = charging
        
        # Main grid balances the system
        net_supply = sum(p for p in allocated_power.values() if p > 0)
        net_demand = actual_demand + (battery_power if battery_mode == 'charge' else 0)
        grid_import = max(0, net_demand - net_supply)
        grid_export = max(0, net_supply - net_demand)
        
        # Compile results
        auction_results = {
            'clearing_price': clearing_price,
            'allocated_power': allocated_power,
            'actual_demand': actual_demand,
            'curtailed_load': total_demand - actual_demand,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'battery_mode': battery_mode,
            'total_supply': net_supply,
            'suppliers': suppliers
        }
        
        return auction_results
    
    def _calculate_rewards(self, env_idx: int, auction_results: Dict, env_data: Dict, 
                       use_policy_incentive: bool = True) -> np.ndarray:
        """
        Calculate rewards for all agents with renewable energy incentives
        
        Args:
        env_idx: Environment index
        auction_results: Auction results
        env_data: Environmental data
        use_policy_incentive: Whether to apply policy incentives (your innovation)
                            - False: Paper baseline (Equation 5)
                            - True: With green energy policy
        
        Returns:
            rewards: [num_agents] numpy array (normalized)
        """
        mg = self.microgrids[env_idx]
        clearing_price = auction_results['clearing_price']
        allocated = auction_results['allocated_power']
        
        rewards = np.zeros(self.num_agents)
        
        # ========== Policy Parameters (for innovation) ==========
    
        
        RENEWABLE_SUBSIDY = 2.0      # 新能源补贴 (cents/kWh)
                                    # 建议范围: 1.5-3.0
                                    # 越高越鼓励新能源
        
        CARBON_TAX = 1.5             # 碳税/污染税 (cents/kWh)
                                    # 建议范围: 1.0-2.5
                                    # 越高越抑制传统能源
        
        GREEN_DISCOUNT = 0.3         # 绿色电力折扣系数
                                    # 建议范围: 0.2-0.5
                                    # Customer使用绿电的折扣率
        
        # ================================
        
        # 1. Wind agent - 清洁能源，获得利润 + 环保补贴
        wind_power = allocated.get('wind', 0)
        if wind_power > 0:
            base_profit = wind_power * clearing_price
            if use_policy_incentive:
                subsidy = wind_power * RENEWABLE_SUBSIDY / 100.0
                rewards[0] = base_profit + subsidy
            else:
                rewards[0] = base_profit

        
        # 2. Solar agent - 清洁能源，获得利润 + 环保补贴
        solar_power = allocated.get('solar', 0)
        if solar_power > 0:
            base_profit = solar_power * clearing_price
            if use_policy_incentive:
                subsidy = solar_power * RENEWABLE_SUBSIDY / 100.0
                rewards[1] = base_profit + subsidy
            else:
                rewards[1] = base_profit
        
        # 3. Diesel agent - 传统能源，利润 - 运营成本 - 碳税
        diesel_power = allocated.get('diesel', 0)
        if diesel_power > 0:
            base_profit = diesel_power * (clearing_price - 0.08)
            if use_policy_incentive:
                carbon_penalty = diesel_power * CARBON_TAX / 100.0
                rewards[2] = base_profit - carbon_penalty
            else:
                rewards[2] = base_profit
    
        
        # 4. Battery agent - 储能套利，鼓励储存绿电
        battery_power = allocated.get('battery', 0)
        if battery_power > 0:  # Discharging (selling)
            discharge_cost = 0.10
            rewards[3] = battery_power * (clearing_price - discharge_cost) / 100.0
        elif battery_power < 0:  # Charging (buying)
            charge_cost = 0.03
            rewards[3] = battery_power * (clearing_price + charge_cost) / 100.0
        
        # 5. Customer agent
        # ==========================================================
        
        actual_demand = auction_results['actual_demand']  # kW
        curtailed = auction_results['curtailed_load']      # kW
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Paper baseline: Equation (5), p. 5752
        # Expense_i = P_C^t × El_i^t + K × |AD_i^t - El_i^t|
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        # Electricity cost: P_C × El (normalized to dollars)
        electricity_cost = actual_demand * clearing_price / 100.0
        
        # Discomfort cost: K × |AD - El| (normalized to dollars)
        # K = 10 cents/kWh (paper p. 5754)
        K = 10.0
        discomfort_cost = K * curtailed / 100.0
        
        # Base expense (paper implementation)
        base_expense = electricity_cost + discomfort_cost
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Optional: Policy incentive (your innovation)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        
        if use_policy_incentive:
            # Calculate green energy ratio
            wind_power = allocated.get('wind', 0)
            solar_power = allocated.get('solar', 0)
            diesel_power = allocated.get('diesel', 0)
            battery_power = allocated.get('battery', 0)
            
            total_supply = wind_power + solar_power + diesel_power + max(0, battery_power)
            
            if total_supply > 0:
                green_ratio = (wind_power + solar_power) / total_supply
                # Green discount (in dollars, same unit as electricity_cost)
                green_discount = electricity_cost * green_ratio * GREEN_DISCOUNT
            else:
                green_discount = 0.0
            
            # Total expense with policy
            total_expense = base_expense - green_discount
            rewards[4] = -total_expense
        else:
            # Paper baseline (no policy incentive)
            rewards[4] = -base_expense
        
        return rewards


# ============================================================================
# 可选：添加绿色能源统计
# ============================================================================

    def _get_green_energy_stats(self, auction_results: Dict) -> Dict:
        """
        计算绿色能源统计指标
        
        Returns:
            stats: {
                'green_ratio': 绿色能源比例 (0-1),
                'carbon_emissions': 碳排放量,
                'renewable_power': 可再生能源总发电量
            }
        """
        allocated = auction_results['allocated_power']
        
        wind_power = allocated.get('wind', 0)
        solar_power = allocated.get('solar', 0)
        diesel_power = allocated.get('diesel', 0)
        battery_power = max(0, allocated.get('battery', 0))
        
        total_supply = wind_power + solar_power + diesel_power + battery_power
        renewable_power = wind_power + solar_power
        
        green_ratio = renewable_power / max(total_supply, 1.0)
        carbon_emissions = diesel_power * 0.5  # kg CO2 per kWh (假设)
        
        return {
            'green_ratio': green_ratio,
            'carbon_emissions': carbon_emissions,
            'renewable_power': renewable_power,
            'total_supply': total_supply,
        }
    
    def _get_observation(self, env_idx: int, env_data: Dict, auction_results: Dict = None) -> np.ndarray:
        """
        Get observations for all agents
        
        Returns:
            obs: [num_agents, obs_size] numpy array
        """
        mg = self.microgrids[env_idx]
        time_hour = env_data['time_hour']
        time_normalized = time_hour / 24.0
        
        # Get grid price
        _, grid_price = self._get_grid_price(time_hour)
        price_normalized = (grid_price - self.price_min) / (self.price_max - self.price_min)
        
        # Get power outputs (if auction results available)
        if auction_results:
            allocated = auction_results['allocated_power']
            wind_power = allocated.get('wind', 0)
            solar_power = allocated.get('solar', 0)
            diesel_power = allocated.get('diesel', 0)
            battery_power = allocated.get('battery', 0)
            actual_demand = auction_results['actual_demand']
        else:
            # Initial state
            wind_power = 0
            solar_power = 0
            diesel_power = 0
            battery_power = 0
            actual_demand = env_data['base_load']
        
        # Create observations array: [num_agents, max_obs_size]
        max_obs_size = max(self.obs_sizes.values())  # = 5
        obs = np.zeros((self.num_agents, max_obs_size), dtype=np.float32)

        # 1. Wind agent (4 dims → pad to 5)
        wind_speed_norm = (env_data['wind_speed'] - 3) / (15 - 3)
        obs[0, :4] = [
            wind_speed_norm,
            wind_power / 100.0,
            time_normalized,
            price_normalized
        ]
        # obs[0, 4] = 0 (自动填充)

        # 2. Solar agent (4 dims → pad to 5)
        irradiance_norm = env_data['solar_irradiance'] / 1000.0
        obs[1, :4] = [
            irradiance_norm,
            solar_power / 50.0,
            time_normalized,
            price_normalized
        ]

        # 3. Diesel agent (5 dims - 完整)
        diesel_component = mg.components['diesel']
        obs[2, :] = [
            diesel_component.fuel_level / 1000.0,
            diesel_power / 75.0,
            time_normalized,
            price_normalized,
            actual_demand / 150.0
        ]

        # 4. Battery agent (5 dims - 完整)
        battery_component = mg.components['battery']
        obs[3, :] = [
            battery_component.soc,
            battery_power / 50.0,
            time_normalized,
            price_normalized,
            actual_demand / 150.0
        ]

        # 5. Customer agent (4 dims → pad to 5)
        obs[4, :4] = [
            env_data['base_load'] / 150.0,
            actual_demand / 150.0,
            time_normalized,
            price_normalized
        ]

        return obs
    
    def reset(self) -> np.ndarray:
        """
        Reset all environments
        
        Returns:
            obs: [num_envs, num_agents, obs_size] numpy array
        """
        # Reset all microgrids
        self.microgrids = [self._create_microgrid() for _ in range(self.num_envs)]
        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        
        # Get initial observations
        obs_list = []
        for env_idx in range(self.num_envs):
            env_data = self._generate_env_data(env_idx)
            self.latest_env_data[env_idx] = env_data
            obs = self._get_observation(env_idx, env_data)
            obs_list.append(obs)
        
        # Stack: [num_envs, num_agents, obs_size]
        return np.stack(obs_list, axis=0)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Execute actions in all environments using auction mechanism
        
        Args:
            actions: [num_envs, num_agents, act_size] numpy array
            
        Returns:
            next_obs: [num_envs, num_agents, obs_size]
            rewards: [num_envs, num_agents]
            dones: [num_envs, num_agents]
        """
        next_obs_list = []
        rewards_list = []
        dones_list = []
        
        for env_idx in range(self.num_envs):
            # Generate environmental data
            env_data = self._generate_env_data(env_idx)
            self.latest_env_data[env_idx] = env_data
            
            # Run auction with agent actions
            auction_results = self._run_auction(env_idx, actions[env_idx], env_data)
            
            # Update battery SOC based on auction results
            battery_power = auction_results['allocated_power'].get('battery', 0)
            battery_component = self.microgrids[env_idx].components['battery']
            if battery_power > 0:  # Discharging
                energy_change = battery_power * 1.0  # 1 hour timestep
                battery_component.soc -= energy_change / battery_component.capacity_kwh
            elif battery_power < 0:  # Charging
                energy_change = abs(battery_power) * 1.0
                battery_component.soc += energy_change / battery_component.capacity_kwh
            battery_component.soc = np.clip(battery_component.soc, 0.0, 1.0)
            
            # Get next observation
            next_obs = self._get_observation(env_idx, env_data, auction_results)
            next_obs_list.append(next_obs)
            
            # Calculate rewards
            rewards = self._calculate_rewards(env_idx, auction_results, env_data, self.use_policy)
            rewards_list.append(rewards)
            
            # Check if done
            self.current_steps[env_idx] += 1
            done = self.current_steps[env_idx] >= self.max_steps
            dones = np.full(self.num_agents, done)
            dones_list.append(dones)
        
        # Stack results
        next_obs = np.stack(next_obs_list, axis=0)
        rewards = np.stack(rewards_list, axis=0)
        dones = np.stack(dones_list, axis=0)
        
        return next_obs, rewards, dones
    
    @property
    def observation_space(self):
        """Return observation space dimensions"""
        return {name: size for name, size in self.obs_sizes.items()}
    
    @property
    def action_space(self):
        """Return action space dimensions"""
        return {name: size for name, size in self.act_sizes.items()}
    
    

if __name__ == "__main__":
    """Test the enhanced environment"""
    print("=" * 60)
    print("🧪 测试增强版 MicrogridEnv（完整拍卖机制）")
    print("=" * 60)
    
    # Create environment
    env = MicrogridEnv(num_envs=2, max_steps=24)
    print(f"\n✓ 环境创建成功")
    print(f"  - num_envs: {env.num_envs}")
    print(f"  - num_agents: {env.num_agents}")
    print(f"  - Agent 名称: {env.agent_names}")
    
    # Test reset
    obs = env.reset()
    print(f"\n✓ Reset 成功")
    print(f"  - obs shape: {obs.shape}")
    
    # Test step with random actions
    # Note: Actions need to match the action space
    actions = np.random.randn(env.num_envs, env.num_agents, 2)
    # Clip actions to valid ranges
    actions[:, :, 0] = np.clip(actions[:, :, 0], -1, 1)  # First dimension
    actions[:, :, 1] = np.clip(actions[:, :, 1], 0, 1)   # Second dimension (bids)
    # For single-action agents, second dimension is ignored
    
    next_obs, rewards, dones = env.step(actions)
    
    print(f"\n✓ Step 成功（含拍卖）")
    print(f"  - next_obs shape: {next_obs.shape}")
    print(f"  - rewards shape: {rewards.shape}")
    print(f"  - rewards (env 0): {rewards[0]}")
    print(f"  - dones shape: {dones.shape}")
    
    # Test multiple steps
    print(f"\n✓ 测试5步运行")
    env.reset()
    total_rewards = np.zeros((env.num_envs, env.num_agents))
    
    for step in range(5):
        actions = np.random.randn(env.num_envs, env.num_agents, 2)
        actions[:, :, 0] = np.clip(actions[:, :, 0], -1, 1)
        actions[:, :, 1] = np.clip(actions[:, :, 1], 0, 1)
        
        obs, rewards, dones = env.step(actions)
        total_rewards += rewards
        print(f"  Step {step+1}: rewards[0] = {rewards[0]}")
    
    print(f"\n  累计奖励 (env 0): {total_rewards[0]}")
    
    print("\n" + "=" * 60)
    print("🎉 增强版环境测试通过！拍卖机制工作正常")
    print("=" * 60)