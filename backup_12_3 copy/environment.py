"""
Enhanced Environment wrapper for Microgrid MADDPG
完整实现5个agents参与拍卖的多智能体RL环境
"""

import numpy as np
import torch
from typing import Tuple, Dict, List

from microgrid_auction_enhanced import (
    MicrogridComponent, WindTurbine, PVSystem, DieselGenerator, 
    Battery, MainGrid, CustomerLoad
)


class Microgrid:
    """Simple microgrid container matching microgrid_auction.py structure"""
    def __init__(self, components: Dict[str, MicrogridComponent]):
        self.components = components


class MicrogridEnv:
    """
    Multi-Agent Microgrid Environment with Full Auction Mechanism
    
    5 learning agents participate:
    1. Wind - sells with bid
    2. Solar - sells with bid  
    3. Diesel - sells with power control + bid
    4. Battery - buys/sells with charge control + bid
    5. Customer - demand response (curtailment)
    
    Main Grid balances as rule-based participant
    """
    
    def __init__(self, num_envs=1, max_steps=24, use_policy=True):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.num_agents = 5
        
        self.agent_names = ['wind', 'solar', 'diesel', 'battery', 'customer']
        
        # Unified observation size (5D for all agents after padding)
        self.obs_sizes = {
            'wind': 4, 'solar': 4, 'diesel': 5, 'battery': 5, 'customer': 4
        }
        
        self.act_sizes = {
            'wind': 1, 'solar': 1, 'diesel': 2, 'battery': 2, 'customer': 1
        }
        
        self.microgrids = [self._create_microgrid() for _ in range(num_envs)]
        self.current_steps = np.zeros(num_envs, dtype=np.int32)
        
        self.price_min = 5.0
        self.price_max = 45.0
        
        self.latest_env_data = [None] * num_envs
        self.use_policy = use_policy

    def _create_microgrid(self) -> Microgrid:
        """Create microgrid matching microgrid_auction.py components"""
        components = {
            'wind': WindTurbine(name='WindTurbine', capacity=30),
            'pv': PVSystem(name='PVSystem', capacity=15),
            'diesel': DieselGenerator(
                name='DieselGenerator', 
                capacity=75,
                fuel_consumption_rate=0.2,
                generation_cost_per_kwh=0.08
            ),
            'battery': Battery(
                name='Battery',
                capacity_kwh=200,
                max_charge_rate_kw=50,
                max_discharge_rate_kw=50,
                initial_soc=0.5,
                charge_cost_per_kwh=0.05,
                discharge_cost_per_kwh=0.15
            ),
            'grid': MainGrid(
                name='MainGrid',
                import_limit=100,
                export_limit=100,
                import_price_per_kwh=0.25,
                export_price_per_kwh=0.1
            ),
            'load': CustomerLoad(name='CustomerLoad')
        }
        
        return Microgrid(components)
    
    def _generate_env_data(self, env_idx: int) -> Dict:
        time_hour = self.current_steps[env_idx] % 24
        wind_speed = np.random.uniform(3, 15)
        solar_irradiance = self._get_solar_irradiance(time_hour)
        
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
        if hour < 6 or hour > 18:
            return 0
        return 1000 * np.sin(np.pi * (hour - 6) / 12) * np.random.uniform(0.7, 1.0)
    
    def _get_grid_price(self, hour: int) -> Tuple[float, float]:
        """Returns (export_price, import_price)"""
        # Import price (MG buying from grid)
        if 23 <= hour or hour < 8:
            import_price = np.random.uniform(13, 16)
        elif 8 <= hour < 12 or 18 <= hour < 23:
            import_price = np.random.uniform(23, 26)
        else:
            import_price = np.random.uniform(40, 42)
        
        # Export price (MG selling to grid)
        export_price = np.random.uniform(5, 15)
        
        return export_price, import_price
    
    def _run_auction(self, env_idx: int, actions: np.ndarray, env_data: Dict) -> Dict:
        """
        Run uniform price auction with Main Grid participation
        """
        mg = self.microgrids[env_idx]
        
        # Parse agent actions
        wind_bid_norm = actions[0, 0]
        solar_bid_norm = actions[1, 0]
        diesel_power_ratio = actions[2, 0]
        diesel_bid_norm = actions[2, 1]
        battery_action = actions[3, 0]
        battery_bid_norm = actions[3, 1]
        customer_curtail = actions[4, 0]
        
        # Denormalize bids
        wind_bid = (wind_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        solar_bid = (solar_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        diesel_bid = (diesel_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        battery_bid = (battery_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        
        # Update component states with environmental data
        mg.components['wind'].update(1.0, {
            'wind_speed': env_data['wind_speed'],
            'bid': wind_bid
        })
        
        mg.components['pv'].update(1.0, {
            'solar_irradiance': env_data['solar_irradiance'],
            'bid': solar_bid
        })
        
        mg.components['diesel'].update(1.0, {
            'target_power': diesel_power_ratio * mg.components['diesel'].capacity,
            'bid': diesel_bid
        })
        
        mg.components['battery'].update(1.0, {
            'target_action': battery_action,
            'bid': battery_bid
        })
        
        mg.components['load'].update(1.0, {
            'base_load': env_data['base_load'],
            'curtailment_ratio': max(0, min(1, customer_curtail))
        })
        
        # Get available power from components
        wind_power = mg.components['wind'].available_power
        solar_power = mg.components['pv'].available_power
        diesel_power = mg.components['diesel'].target_power
        
        # Battery mode
        battery_component = mg.components['battery']
        if battery_action > 0:  # Discharge
            battery_power = min(
                battery_action * battery_component.max_discharge_rate_kw,
                battery_component.soc * battery_component.capacity_kwh
            )
            battery_mode = 'discharge'
        else:
            battery_power = min(
                abs(battery_action) * battery_component.max_charge_rate_kw,
                (1 - battery_component.soc) * battery_component.capacity_kwh
            )
            battery_mode = 'charge'
        
        # Customer demand
        _, _, actual_demand = mg.components['load'].get_demand()
        base_demand, curtailment_ratio, _ = mg.components['load'].get_demand()
        
        # Get Main Grid prices
        grid_export_price, grid_import_price = self._get_grid_price(env_data['time_hour'])
        
        # Build supply curve
        suppliers = []
        if wind_power > 0:
            suppliers.append({'name': 'wind', 'power': wind_power, 'bid': wind_bid, 'cost': 0.0})
        if solar_power > 0:
            suppliers.append({'name': 'solar', 'power': solar_power, 'bid': solar_bid, 'cost': 0.0})
        if diesel_power > 0:
            suppliers.append({'name': 'diesel', 'power': diesel_power, 'bid': diesel_bid, 'cost': 0.08})
        if battery_mode == 'discharge' and battery_power > 0:
            suppliers.append({'name': 'battery', 'power': battery_power, 'bid': battery_bid, 'cost': 0.15})
        
        # Add Main Grid as supplier with infinite capacity
        suppliers.append({
            'name': 'main_grid',
            'power': 999.0,
            'bid': grid_import_price,
            'cost': 0.0
        })
        
        suppliers.sort(key=lambda x: x['bid'])
        
        # Total demand
        battery_charge_demand = battery_power if battery_mode == 'charge' else 0
        total_demand = actual_demand + battery_charge_demand
        
        # Market clearing
        if total_demand <= 0:
            clearing_price = self.price_min
            allocated_power = {s['name']: 0 for s in suppliers}
        else:
            cumulative = 0
            clearing_price = self.price_min
            allocated_power = {}
            
            for supplier in suppliers:
                if cumulative >= total_demand:
                    allocated_power[supplier['name']] = 0
                elif cumulative + supplier['power'] <= total_demand:
                    allocated_power[supplier['name']] = supplier['power']
                    cumulative += supplier['power']
                    clearing_price = supplier['bid']
                else:
                    allocated_power[supplier['name']] = total_demand - cumulative
                    cumulative = total_demand
                    clearing_price = supplier['bid']
                    break
        
        # Handle unsold energy (export to Main Grid)
        grid_purchases = 0
        for supplier in suppliers:
            if supplier['name'] in ['wind', 'solar', 'diesel']:
                accepted = allocated_power.get(supplier['name'], 0)
                unsold = supplier['power'] - accepted
                if unsold > 0:
                    grid_purchases += unsold
        
        # Battery charging
        if battery_mode == 'charge':
            net_supply = sum(allocated_power.get(name, 0) 
                           for name in ['wind', 'solar', 'diesel', 'main_grid'])
            available_for_battery = net_supply - actual_demand
            allocated_power['battery'] = -min(battery_charge_demand, max(0, available_for_battery))
        
        # Main Grid net position
        main_grid_sales = allocated_power.get('main_grid', 0)
        main_grid_net = main_grid_sales - grid_purchases
        
        return {
            'clearing_price': clearing_price,
            'allocated_power': allocated_power,
            'actual_demand': actual_demand,
            'curtailed_load': base_demand - actual_demand,
            'grid_import': max(0, main_grid_net),
            'grid_export': max(0, -main_grid_net),
            'battery_mode': battery_mode,
            'total_supply': sum(allocated_power.get(name, 0) 
                              for name in ['wind', 'solar', 'diesel', 'main_grid']),
            'suppliers': suppliers,
            'main_grid_trade': main_grid_net,
            'bids': {
                'wind': wind_bid,
                'solar': solar_bid,
                'diesel': diesel_bid,
                'battery': battery_bid
            },
        }
    
    def _calculate_rewards(self, env_idx: int, auction_results: Dict, env_data: Dict,
                          use_policy_incentive: bool = True) -> np.ndarray:
        """Calculate rewards matching paper Equation 5"""
        clearing_price = auction_results['clearing_price']
        allocated = auction_results['allocated_power']
        
        rewards = np.zeros(self.num_agents)
        
        RENEWABLE_SUBSIDY = 2
        CARBON_TAX = 5
        GREEN_DISCOUNT = 0.5
        
        # Wind
        wind_power = allocated.get('wind', 0)
        if wind_power > 0:
            base_profit = wind_power * clearing_price / 100.0
            rewards[0] = base_profit + (wind_power * RENEWABLE_SUBSIDY / 100.0 if use_policy_incentive else 0)
        
        # Solar
        solar_power = allocated.get('solar', 0)
        if solar_power > 0:
            base_profit = solar_power * clearing_price / 100.0
            rewards[1] = base_profit + (solar_power * RENEWABLE_SUBSIDY / 100.0 if use_policy_incentive else 0)
        
        # Diesel
        diesel_power = allocated.get('diesel', 0)
        if diesel_power > 0:
            base_profit = diesel_power * (clearing_price - 8.0) / 100.0
            rewards[2] = base_profit - (diesel_power * CARBON_TAX / 100.0 if use_policy_incentive else 0)
        
        # Battery
        battery_power = allocated.get('battery', 0)
        if battery_power > 0:
            rewards[3] = battery_power * (clearing_price - 8.0) / 100.0
        elif battery_power < 0:
            rewards[3] = battery_power * (clearing_price + 2.0) / 100.0
        
        # Customer (Paper Equation 5)
        actual_demand = auction_results['actual_demand']
        curtailed = auction_results['curtailed_load']
        
        electricity_cost = actual_demand * clearing_price / 100.0
        K = 10.0
        discomfort_cost = K * curtailed / 100.0
        base_expense = electricity_cost + discomfort_cost
        
        if use_policy_incentive:
            wind_power = allocated.get('wind', 0)
            solar_power = allocated.get('solar', 0)
            total_supply = sum(allocated.get(name, 0) 
                             for name in ['wind', 'solar', 'diesel', 'main_grid'])
            
            if total_supply > 0:
                green_ratio = (wind_power + solar_power) / total_supply
                green_discount = electricity_cost * green_ratio * GREEN_DISCOUNT
                base_expense -= green_discount
        
        rewards[4] = -base_expense
        
        return rewards
    
    def _get_observation(self, env_idx: int, env_data: Dict, auction_results: Dict = None) -> np.ndarray:
        """Get observations (5D unified across agents)"""
        mg = self.microgrids[env_idx]
        time_hour = env_data['time_hour']
        time_normalized = time_hour / 24.0
        
        _, grid_price = self._get_grid_price(time_hour)
        price_normalized = (grid_price - self.price_min) / (self.price_max - self.price_min)
        
        if auction_results:
            allocated = auction_results['allocated_power']
            wind_power = allocated.get('wind', 0)
            solar_power = allocated.get('solar', 0)
            diesel_power = allocated.get('diesel', 0)
            battery_power = allocated.get('battery', 0)
            actual_demand = auction_results['actual_demand']
        else:
            wind_power = solar_power = diesel_power = battery_power = 0
            actual_demand = env_data['base_load']
        
        obs = np.zeros((self.num_agents, 5), dtype=np.float32)
        
        # Wind (4D → 5D)
        wind_speed_norm = (env_data['wind_speed'] - 3) / 12
        obs[0, :4] = [wind_speed_norm, wind_power / 100.0, time_normalized, price_normalized]
        
        # Solar (4D → 5D)
        irradiance_norm = env_data['solar_irradiance'] / 1000.0
        obs[1, :4] = [irradiance_norm, solar_power / 50.0, time_normalized, price_normalized]
        
        # Diesel (5D)
        obs[2, :] = [
            mg.components['diesel'].fuel_level / 1000.0,
            diesel_power / 75.0,
            time_normalized,
            price_normalized,
            actual_demand / 150.0
        ]
        
        # Battery (5D)
        obs[3, :] = [
            mg.components['battery'].soc,
            battery_power / 50.0,
            time_normalized,
            price_normalized,
            actual_demand / 150.0
        ]
        
        # Customer (4D → 5D)
        obs[4, :4] = [
            env_data['base_load'] / 150.0,
            actual_demand / 150.0,
            time_normalized,
            price_normalized
        ]
        
        return obs
    
    def reset(self) -> np.ndarray:
        self.microgrids = [self._create_microgrid() for _ in range(self.num_envs)]
        self.current_steps = np.zeros(self.num_envs, dtype=np.int32)
        
        obs_list = []
        for env_idx in range(self.num_envs):
            env_data = self._generate_env_data(env_idx)
            self.latest_env_data[env_idx] = env_data
            obs = self._get_observation(env_idx, env_data)
            obs_list.append(obs)
        
        return np.stack(obs_list, axis=0)
    
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        next_obs_list = []
        rewards_list = []
        dones_list = []
        infos_list = []
        for env_idx in range(self.num_envs):
            env_data = self._generate_env_data(env_idx)
            self.latest_env_data[env_idx] = env_data
            
            auction_results = self._run_auction(env_idx, actions[env_idx], env_data)
            
            # Update battery SOC
            battery_power = auction_results['allocated_power'].get('battery', 0)
            battery = self.microgrids[env_idx].components['battery']
            if battery_power != 0:
                energy_change = abs(battery_power) * 1.0
                if battery_power > 0:
                    battery.soc -= energy_change / battery.capacity_kwh
                else:
                    battery.soc += energy_change / battery.capacity_kwh
                battery.soc = np.clip(battery.soc, 0.0, 1.0)
            
            next_obs = self._get_observation(env_idx, env_data, auction_results)
            next_obs_list.append(next_obs)
            
            rewards = self._calculate_rewards(env_idx, auction_results, env_data, self.use_policy)
            rewards_list.append(rewards)
            info = {
                'bids': auction_results['bids'],
                'generation': {
                    'wind': auction_results['allocated_power'].get('wind', 0),
                    'solar': auction_results['allocated_power'].get('solar', 0),
                    'diesel': auction_results['allocated_power'].get('diesel', 0),
                    'battery': auction_results['allocated_power'].get('battery', 0),
                    'main_grid_import': auction_results.get('grid_import', 0),
                    'main_grid_export': auction_results.get('grid_export', 0)
                }
            }
            infos_list.append(info)

            self.current_steps[env_idx] += 1
            done = self.current_steps[env_idx] >= self.max_steps
            dones = np.full(self.num_agents, done)
            dones_list.append(dones)
        
        return np.stack(next_obs_list), np.stack(rewards_list), np.stack(dones_list), infos_list
    
    @property
    def observation_space(self):
        return {name: size for name, size in self.obs_sizes.items()}
    
    @property
    def action_space(self):
        return {name: size for name, size in self.act_sizes.items()}