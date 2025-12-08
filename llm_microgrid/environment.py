import numpy as np
from typing import Dict
import json
from microgrid_components import (
    WindTurbine, PVSystem, DieselGenerator,
    Battery, CustomerLoad,
    get_solar_irradiance, get_grid_price
)


class MicrogridEnv:
    def __init__(self, config_path, num_envs=1, max_steps=24, use_policy=True):
        self.num_envs = num_envs
        self.max_steps = max_steps

        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.supplier_configs = self.config['components']['suppliers']
        self.agent_names = [s['name'] for s in self.supplier_configs] + ['battery', 'customer']
        self.agent_types = [s['type'] for s in self.supplier_configs] + ['battery', 'customer']
        self.num_agents = len(self.agent_names)

        self.obs_sizes = {
            'wind': 4, 'solar': 4, 'diesel': 5, 'battery': 5, 'customer': 4
        }
        self.act_sizes = {
            'wind': 1, 'solar': 1, 'diesel': 2, 'battery': 2, 'customer': 1
        }
        
        self.microgrids = [self._create_microgrid() for _ in range(num_envs)]
        self.current_steps = np.zeros(num_envs, dtype=np.int32)

        self.price_min = self.config['market']['price_min']
        self.price_max = self.config['market']['price_max']
        self.max_customer_curtail = self.config['market']['max_customer_curtail']
        
        self.latest_env_data = [None] * num_envs
        self.use_policy = use_policy
        self.reward_function = self._default_reward_function  # 新增

    def set_reward_function(self, reward_fn):
        """Allow to use LLM generated reward function"""
        self.reward_function = reward_fn

    def _create_microgrid(self) -> dict:
        components = {}

        # Creating suppliers
        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']

            if stype == 'wind':
                components[name] = WindTurbine(name=name, config=supplier_cfg)
            elif stype == 'solar':
                components[name] = PVSystem(name=name, config=supplier_cfg)
            elif stype == 'diesel':
                components[name] = DieselGenerator(name=name, config=supplier_cfg)

        # Creating battery and customer
        components['battery'] = Battery(name='Battery', config=self.config['components']['battery'])
        components['customer'] = CustomerLoad(name='CustomerLoad', config=self.config['components']['customer'])
        
        return components

    def _generate_env_data(self, env_idx: int) -> Dict:
        time_hour = self.current_steps[env_idx] % 24
        env_data = {'time_hour': time_hour}

        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']

            if stype == 'wind':
                env_data[f'{name}_wind_speed'] = np.random.uniform(3, 15)
            elif stype == 'solar':
                env_data[f'{name}_solar_irradiance'] = get_solar_irradiance(time_hour, name, self.config)

        # Customer load
        base_load = 80
        load_variation = 40 * np.sin(np.pi * time_hour / 12)
        env_data['base_load'] = max(30, base_load + load_variation + np.random.uniform(-10, 10))

        return env_data

    def _run_auction(self, env_idx: int, actions: np.ndarray, env_data: Dict) -> Dict:
        mg = self.microgrids[env_idx]
        hour = env_data['time_hour']

        # Parse actions dynamically
        agent_idx = 0
        supplier_actions = {}

        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            act_size = self.act_sizes[supplier_cfg['type']]
            supplier_actions[name] = actions[agent_idx, :act_size]
            agent_idx += 1

        battery_action = actions[agent_idx, :2]
        agent_idx += 1

        customer_action = (actions[agent_idx, 0] + 1) / 2 * self.max_customer_curtail

        # Update supplier components and collect available power
        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']

            if stype == 'wind':
                wind_speed = env_data.get(f'{name}_wind_speed', 0)
                mg[name].update(1.0, {'wind_speed': wind_speed})
            elif stype == 'solar':
                solar_irradiance = env_data.get(f'{name}_solar_irradiance', 0)
                mg[name].update(1.0, {'solar_irradiance': solar_irradiance})
            elif stype == 'diesel':
                # Diesel has 2 actions: [target_power, bid]
                target_power = (supplier_actions[name][0] + 1) / 2 * mg[name].capacity
                mg[name].update(1.0, {'target_power': target_power})

        # Update battery
        battery_action_normalized = battery_action[0]
        mg['battery'].update(1.0, {'target_action': battery_action_normalized})
        battery_power = mg['battery'].power_output
        battery_mode = 'discharge' if battery_power > 0 else 'charge' if battery_power < 0 else 'idle'

        # Update customer
        mg['customer'].update(1.0, {
            'base_load': env_data['base_load'],
            'curtailment_ratio': customer_action
        })
        actual_demand = mg['customer'].actual_consumption
        curtailed_load = env_data['base_load'] - actual_demand

        # Get grid prices
        grid_export_price, grid_import_price = get_grid_price(hour, self.config)

        # Build sell orders dynamically
        sell_orders = []

        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']
            power = mg[name].available_power if hasattr(mg[name], 'available_power') else mg[name].power_output

            if stype == 'diesel':
                # Diesel: action[1] is bid
                bid = (supplier_actions[name][1] + 1) / 2 * (self.price_max - self.price_min) + self.price_min
            else:
                # Wind/Solar: action[0] is bid
                bid = (supplier_actions[name][0] + 1) / 2 * (self.price_max - self.price_min) + self.price_min

            if power > 0:
                sell_orders.append({
                    'agent': name,
                    'quantity': power,
                    'ask': bid,
                    'type': stype,
                    'remaining': power
                })

        # Battery as seller (if discharging)
        if battery_mode == 'discharge' and battery_power > 0:
            battery_bid = (battery_action[1] + 1) / 2 * (self.price_max - self.price_min) + self.price_min
            sell_orders.append({
                'agent': 'battery',
                'quantity': abs(battery_power),
                'ask': battery_bid,
                'type': 'storage',
                'remaining': abs(battery_power)
            })

        # Main Grid as backstop seller
        sell_orders.append({
            'agent': 'main_grid',
            'quantity': 999.0,
            'ask': grid_import_price,
            'type': 'grid',
            'remaining': 999.0
        })

        sell_orders.sort(key=lambda x: x['ask'])

        # Build buy orders
        buy_orders = []

        # Customer as primary buyer
        if actual_demand > 0:
            customer_bid = self.price_max  # Customer willing to pay max
            buy_orders.append({
                'agent': 'customer',
                'quantity': actual_demand,
                'bid': customer_bid,
                'type': 'load',
                'remaining': actual_demand
            })

        # Battery as buyer (if charging)
        if battery_mode == 'charge' and battery_power < 0:
            battery_bid = (battery_action[1] + 1) / 2 * (self.price_max - self.price_min) + self.price_min
            buy_orders.append({
                'agent': 'battery',
                'quantity': abs(battery_power),
                'bid': battery_bid,
                'type': 'storage',
                'remaining': abs(battery_power)
            })

        # Main Grid as backstop buyer
        buy_orders.append({
            'agent': 'main_grid_buyer',
            'quantity': 999.0,
            'bid': grid_export_price,
            'type': 'grid',
            'remaining': 999.0
        })

        buy_orders.sort(key=lambda x: x['bid'], reverse=True)

        # Auction clearing
        matched_trades = []

        for buy_order in buy_orders:
            if buy_order['remaining'] <= 0:
                continue

            for sell_order in sell_orders:
                if sell_order['remaining'] <= 0:
                    continue

                if buy_order['bid'] >= sell_order['ask']:
                    trade_quantity = min(buy_order['remaining'], sell_order['remaining'])

                    # Determine trade price
                    if buy_order['agent'] == 'main_grid_buyer':
                        trade_price = buy_order['bid']
                    elif sell_order['agent'] == 'main_grid':
                        trade_price = sell_order['ask']
                    else:
                        trade_price = (buy_order['bid'] + sell_order['ask']) / 2.0

                    matched_trades.append({
                        'buyer': buy_order['agent'],
                        'seller': sell_order['agent'],
                        'quantity': trade_quantity,
                        'price': trade_price
                    })

                    buy_order['remaining'] -= trade_quantity
                    sell_order['remaining'] -= trade_quantity

                    if buy_order['remaining'] <= 0:
                        break

        # Calculate clearing price (weighted average of trade prices)
        if matched_trades:
            total_quantity = sum(t['quantity'] for t in matched_trades)
            if total_quantity > 0:
                clearing_price = sum(t['quantity'] * t['price'] for t in matched_trades) / total_quantity
            else:
                clearing_price = self.price_min
        else:
            clearing_price = self.price_min

        # Aggregate allocations
        allocated_power = {name: 0.0 for name in self.agent_names}
        allocated_power['main_grid'] = 0.0

        for trade in matched_trades:
            seller = trade['seller']
            buyer = trade['buyer']
            qty = trade['quantity']

            # Seller gets positive allocation
            if seller in allocated_power:
                allocated_power[seller] += qty
            elif seller == 'main_grid':
                allocated_power['main_grid'] += qty

            # Buyer gets negative allocation (except customer)
            if buyer == 'battery':
                allocated_power['battery'] -= qty
            elif buyer == 'customer' or buyer == 'main_grid_buyer':
                pass

        # Calculate grid import/export
        grid_import = sum(t['quantity'] for t in matched_trades if t['seller'] == 'main_grid')
        grid_export = sum(t['quantity'] for t in matched_trades if t['buyer'] == 'main_grid_buyer')

        # Collect bids for info
        bids = {}
        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']
            if stype == 'diesel':
                bids[name] = (supplier_actions[name][1] + 1) / 2 * (
                            self.price_max - self.price_min) + self.price_min
            else:
                bids[name] = (supplier_actions[name][0] + 1) / 2 * (
                            self.price_max - self.price_min) + self.price_min

        bids['battery'] = (battery_action[1] + 1) / 2 * (self.price_max - self.price_min) + self.price_min

        # Return auction results
        return {
            'clearing_price': clearing_price,
            'allocated_power': allocated_power,
            'actual_demand': actual_demand,
            'curtailed_load': curtailed_load,
            'matched_trades': matched_trades,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'bids': bids
        }
    
    def _default_reward_function(self, env_idx: int, auction_results: Dict, env_data: Dict,
                          use_policy_incentive: bool = True) -> np.ndarray:
        params = self.config['reward_params']
        RENEWABLE_SUBSIDY = params['renewable_subsidy']
        CARBON_TAX = params['carbon_tax']
        GREEN_DISCOUNT = params['green_discount']
        K = params['discomfort_cost_coefficient']

        clearing_price = auction_results['clearing_price']
        allocated = auction_results['allocated_power']
        matched_trades = auction_results['matched_trades']

        rewards = np.zeros(self.num_agents)

        # Calculate supplier rewards
        idx = 0
        total_renewable_power = 0.0

        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']
            power = allocated.get(name, 0)

            if stype in ['wind', 'solar']:
                # Renewable energy suppliers
                if power > 0:
                    base_profit = power * clearing_price / 100.0
                    rewards[idx] = base_profit

                    if use_policy_incentive:
                        rewards[idx] += power * RENEWABLE_SUBSIDY / 100.0

                    total_renewable_power += power

            elif stype == 'diesel':
                if power > 0:
                    base_profit = power * (clearing_price - 8.0) / 100.0
                    rewards[idx] = base_profit

                    if use_policy_incentive:
                        rewards[idx] -= power * CARBON_TAX / 100.0

            idx += 1

        # Battery reward
        battery_power = allocated.get('battery', 0)
        if battery_power > 0:  # Discharging
            rewards[idx] = battery_power * (clearing_price - 8.0) / 100.0
        elif battery_power < 0:  # Charging
            rewards[idx] = battery_power * (clearing_price + 2.0) / 100.0

        idx += 1

        # Customer reward
        electricity_cost = 0.0
        for trade in matched_trades:
            if trade['buyer'] == 'customer':
                electricity_cost += trade['quantity'] * trade['price'] / 100.0

        discomfort_cost = K * auction_results['curtailed_load'] / 100.0
        base_expense = electricity_cost + discomfort_cost

        # Green energy discount
        if use_policy_incentive:
            total_customer_purchase = sum(
                trade['quantity'] for trade in matched_trades
                if trade['buyer'] == 'customer'
            )

            if total_customer_purchase > 0:
                green_ratio = total_renewable_power / total_customer_purchase
                green_discount = electricity_cost * green_ratio * GREEN_DISCOUNT
                base_expense -= green_discount

        rewards[idx] = -base_expense

        return rewards

    def _get_observation(self, env_idx, env_data, auction_results=None):
        mg = self.microgrids[env_idx]
        time_hour = env_data['time_hour']
        time_normalized = time_hour / 24.0

        _, grid_price = get_grid_price(time_hour, self.config)
        price_normalized = (grid_price - self.price_min) / (self.price_max - self.price_min)

        # Get allocated power if available
        if auction_results:
            allocated = auction_results['allocated_power']
            actual_demand = auction_results['actual_demand']
        else:
            allocated = {name: 0.0 for name in self.agent_names}
            actual_demand = env_data.get('base_load', 0)

        obs = np.zeros((self.num_agents, 5), dtype=np.float32)

        # Build observations for suppliers
        idx = 0
        for supplier_cfg in self.supplier_configs:
            name = supplier_cfg['name']
            stype = supplier_cfg['type']
            power = allocated.get(name, 0)

            if stype == 'wind':
                wind_speed = env_data.get(f'{name}_wind_speed', 0)
                wind_speed_norm = (wind_speed - 3) / 12
                obs[idx, :4] = [wind_speed_norm, power / 100.0, time_normalized, price_normalized]

            elif stype == 'solar':
                irradiance = env_data.get(f'{name}_solar_irradiance', 0)
                irradiance_norm = irradiance / 1000.0
                obs[idx, :4] = [irradiance_norm, power / 50.0, time_normalized, price_normalized]

            elif stype == 'diesel':
                fuel_level_norm = mg[name].fuel_level / 1000.0
                obs[idx, :] = [
                    fuel_level_norm,
                    power / 75.0,
                    time_normalized,
                    price_normalized,
                    actual_demand / 150.0
                ]

            idx += 1

        # Battery observation
        battery_power = allocated.get('battery', 0)
        obs[idx, :] = [
            mg['battery'].soc,
            battery_power / 50.0,
            time_normalized,
            price_normalized,
            actual_demand / 150.0
        ]
        idx += 1

        # Customer observation
        base_load = env_data.get('base_load', 0)
        obs[idx, :4] = [
            base_load / 150.0,
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
    
    def step(self, actions: np.ndarray):
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
            battery = self.microgrids[env_idx]['battery']
            if battery_power != 0:
                energy_change = abs(battery_power) * 1.0
                if battery_power > 0:
                    battery.soc -= energy_change / battery.capacity_kwh
                else:
                    battery.soc += energy_change / battery.capacity_kwh
                battery.soc = np.clip(battery.soc, 0.0, 1.0)
            
            next_obs = self._get_observation(env_idx, env_data, auction_results)
            next_obs_list.append(next_obs)
            
            rewards = self.reward_function(env_idx, auction_results, env_data, self.use_policy)
            rewards_list.append(rewards)
            infos_list.append(auction_results)

            self.current_steps[env_idx] += 1
            done = self.current_steps[env_idx] >= self.max_steps
            dones = np.full(self.num_agents, done)
            dones_list.append(dones)
        
        return np.stack(next_obs_list), np.stack(rewards_list), np.stack(dones_list), infos_list