"""
Enhanced Environment wrapper for Microgrid MADDPG
完整实现5个agents参与拍卖的多智能体RL环境
"""

import numpy as np
from typing import Tuple, Dict, Any

from microgrid_components import (
    WindTurbine, PVSystem, DieselGenerator,
    Battery, MainGrid, CustomerLoad
)
from numpy import ndarray, dtype


def _get_solar_irradiance(hour: int) -> float:
    if hour < 6 or hour > 18:
        return 0
    return 1000 * np.sin(np.pi * (hour - 6) / 12) * np.random.uniform(0.7, 1.0)


def _get_grid_price(hour: int) -> Tuple[float, float]:
    """Returns (export_price, import_price)"""
    # Import price (MG buying from grid)
    if 23 <= hour or hour < 8:
        import_price = np.random.uniform(15, 18)
    elif 8 <= hour < 12 or 18 <= hour < 23:
        import_price = np.random.uniform(23, 26)
    else:
        import_price = np.random.uniform(40, 42)

    # Export price (MG selling to grid)
    export_price = np.random.uniform(5, 10)

    return export_price, import_price


def _build_sell_orders(env_idx: int, wind_power: float, solar_power: float,
                       diesel_power: float, battery_power: float, battery_mode: str,
                       wind_ask: float, solar_ask: float, diesel_ask: float,
                       battery_ask: float, grid_import_price: float) -> list:
    """
    Build sell orders (asks) from all potential sellers

    Returns:
        sell_orders: List of dicts with structure:
            {'agent': str, 'quantity': float, 'ask': float,
             'type': str, 'remaining': float}
    """
    sell_orders = []

    # Wind generator
    if wind_power > 0:
        sell_orders.append({
            'agent': 'wind',
            'quantity': wind_power,
            'ask': wind_ask,
            'type': 'renewable',
            'remaining': wind_power
        })

    # Solar generator
    if solar_power > 0:
        sell_orders.append({
            'agent': 'solar',
            'quantity': solar_power,
            'ask': solar_ask,
            'type': 'renewable',
            'remaining': solar_power
        })

    # Diesel generator
    if diesel_power > 0:
        sell_orders.append({
            'agent': 'diesel',
            'quantity': diesel_power,
            'ask': diesel_ask,
            'type': 'fossil',
            'remaining': diesel_power
        })

    # Battery (if discharging)
    if battery_mode == 'discharge' and battery_power > 0:
        sell_orders.append({
            'agent': 'battery',
            'quantity': battery_power,
            'ask': battery_ask,
            'type': 'storage',
            'remaining': battery_power
        })

    # Main Grid as backstop seller (infinite capacity at high price)
    sell_orders.append({
        'agent': 'main_grid',
        'quantity': 999.0,
        'ask': grid_import_price,
        'type': 'grid',
        'remaining': 999.0
    })

    # Sort by ask price (ascending) - merit order dispatch
    sell_orders.sort(key=lambda x: x['ask'])

    return sell_orders


def _build_buy_orders(env_idx: int, actual_demand: float,
                      battery_power: float, battery_mode: str, battery_bid: float,
                      customer_bid: float, grid_export_price: float) -> list:
    """
    Build buy orders (bids) from all potential buyers

    Returns:
        buy_orders: List of dicts with structure:
            {'agent': str, 'quantity': float, 'bid': float,
             'type': str, 'remaining': float}
    """
    buy_orders = []

    # Customer as primary buyer
    if actual_demand > 0:
        buy_orders.append({
            'agent': 'customer',
            'quantity': actual_demand,
            'bid': customer_bid,
            'type': 'load',
            'remaining': actual_demand
        })

    # Battery (if charging)
    if battery_mode == 'charge' and battery_power > 0:
        buy_orders.append({
            'agent': 'battery',
            'quantity': battery_power,
            'bid': battery_bid,
            'type': 'storage',
            'remaining': battery_power
        })

    # Main Grid as backstop buyer (infinite capacity at low price)
    # This buys surplus energy (MG exports to grid)
    buy_orders.append({
        'agent': 'main_grid_buyer',
        'quantity': 999.0,
        'bid': grid_export_price,
        'type': 'grid',
        'remaining': 999.0
    })

    # Sort by bid price (descending) - highest willingness to pay first
    buy_orders.sort(key=lambda x: x['bid'], reverse=True)

    return buy_orders


def _aggregate_allocations(matched_trades: list, agent_names: list) -> dict:
    """
    Aggregate matched trades into net allocations per agent

    Args:
        matched_trades: List of executed trades
        agent_names: List of agent names ['wind', 'solar', 'diesel', 'battery', 'customer']

    Returns:
        allocated_power: Dict mapping agent name to net power (positive=sell, negative=buy)
    """
    allocated_power = {name: 0.0 for name in agent_names}
    allocated_power['main_grid'] = 0.0

    for trade in matched_trades:
        seller = trade['seller']
        buyer = trade['buyer']
        qty = trade['quantity']

        # Seller gets positive allocation (sold energy)
        if seller in allocated_power:
            allocated_power[seller] += qty
        elif seller == 'main_grid':
            allocated_power['main_grid'] += qty

        # Buyer gets negative allocation (bought energy)
        # Exception: customer's consumption is recorded separately
        if buyer == 'battery':
            allocated_power['battery'] -= qty  # Negative for charging
        elif buyer == 'customer' or buyer == 'main_grid_buyer':
            # Customer consumption recorded in actual_demand, not in allocated_power
            pass

    return allocated_power


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
        self.max_customer_curtail = 0.2
        
        self.latest_env_data = [None] * num_envs
        self.use_policy = use_policy
        self.reward_function = self._default_reward_function  # 新增

    def set_reward_function(self, reward_fn):
        """允许外部设置自定义reward函数"""
        self.reward_function = reward_fn

    def _create_microgrid(self) -> dict:
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
        
        return components
    
    def _generate_env_data(self, env_idx: int) -> Dict:
        time_hour = self.current_steps[env_idx] % 24
        wind_speed = np.random.uniform(3, 15)
        solar_irradiance = _get_solar_irradiance(time_hour)
        
        base_load = 80
        load_variation = 40 * np.sin(np.pi * time_hour / 12)
        load = max(30, base_load + load_variation + np.random.uniform(-10, 10))
        
        return {
            'time_hour': time_hour,
            'wind_speed': wind_speed,
            'solar_irradiance': solar_irradiance,
            'base_load': load
        }

    def _get_customer_bid(self, hour: int, demand: float) -> float:
        """
        Calculate customer's willingness to pay for electricity

        Args:
            hour: Current hour (0-23)
            demand: Customer demand in kW

        Returns:
            customer_bid: Willingness to pay in cents/kWh
        """
        _, grid_import_price = _get_grid_price(hour)

        customer_bid = self.price_max

        # Optional: Add demand elasticity
        # High demand → willing to pay more
        # Can be enhanced with utility function

        return customer_bid

    def _match_orders(self, sell_orders: list, buy_orders: list) -> tuple:
        """
        Execute double auction matching: buyers and sellers trade where bid >= ask

        Args:
            sell_orders: List of sell orders (sorted by ask ascending)
            buy_orders: List of buy orders (sorted by bid descending)

        Returns:
            matched_trades: List of executed trades
            clearing_price: Uniform clearing price (last matched price)
        """
        matched_trades = []
        clearing_price = self.price_min

        # 【新增】跟踪内部交易用于计算加权平均清算价格
        internal_trade_value = 0.0  # 累计：交易价格 × 交易量
        internal_trade_quantity = 0.0  # 累计：交易量

        sell_idx = 0
        buy_idx = 0

        while sell_idx < len(sell_orders) and buy_idx < len(buy_orders):
            seller = sell_orders[sell_idx]
            buyer = buy_orders[buy_idx]

            # Check if trade is economically feasible
            if buyer['bid'] < seller['ask']:
                # No more profitable trades possible
                break

            # Trade quantity is minimum of remaining quantities
            trade_qty = min(seller['remaining'], buyer['remaining'])

            # Determine trade price based on participants
            # Main Grid trades use fixed prices, others use seller's ask
            if buyer['agent'] == 'main_grid_buyer':
                # Main Grid buying (MG exporting): use grid export price
                trade_price = buyer['bid']  # Already is grid_export_price
                # Don't update clearing_price for grid trades
            else:
                # Internal MG trade: use seller's ask
                trade_price = seller['ask']
                internal_trade_value += trade_price * trade_qty
                internal_trade_quantity += trade_qty

            # Record the trade
            matched_trades.append({
                'seller': seller['agent'],
                'buyer': buyer['agent'],
                'quantity': trade_qty,
                'price': trade_price
            })

            # Update remaining quantities
            seller['remaining'] -= trade_qty
            buyer['remaining'] -= trade_qty

            # Move to next order if current one is fully matched
            if seller['remaining'] <= 1e-6:  # Use small epsilon for float comparison
                sell_idx += 1
            if buyer['remaining'] <= 1e-6:
                buy_idx += 1

        # 新增：计算加权平均清算价格
        if internal_trade_quantity > 0:
            # 情况1：有内部交易 - 使用加权平均价格
            clearing_price = internal_trade_value / internal_trade_quantity
        else:
            # 情况2：无内部交易 - 使用回退逻辑确定合理的市场价格
            # 优先级：Customer出价 > 最低非Grid卖家报价 > 最低价格下限
            if len(buy_orders) > 0 and buy_orders[0]['agent'] == 'customer':
                # 使用Customer的支付意愿作为市场价格参考
                clearing_price = buy_orders[0]['bid']
            elif len(sell_orders) > 0:
                # 找到第一个非Main Grid的卖家
                for sell_order in sell_orders:
                    if sell_order['agent'] != 'main_grid':
                        clearing_price = sell_order['ask']
                        break
            # 如果以上都不满足，clearing_price保持初始值self.price_min

        # 确保清算价格在合理范围内
        clearing_price = np.clip(clearing_price, self.price_min, self.price_max)

        return matched_trades, clearing_price

    def _run_auction(self, env_idx: int, actions: np.ndarray, env_data: Dict) -> Dict:
        """
        Run DOUBLE AUCTION with uniform price clearing

        Double Auction Process:
        1. Parse agent actions and update component states
        2. Build sell orders (asks) from suppliers
        3. Build buy orders (bids) from demanders
        4. Match orders: trade where buyer_bid >= seller_ask
        5. Aggregate allocations per agent
        6. Calculate grid import/export and market metrics

        Args:
            env_idx: Environment index
            actions: Agent actions [5, max_act_size]
            env_data: Environmental data dict

        Returns:
            auction_results: Dict with keys:
                - clearing_price
                - allocated_power
                - actual_demand
                - curtailed_load
                - grid_import
                - grid_export
                - battery_mode
                - total_supply
                - matched_trades
                - main_grid_trade
                - market_efficiency
        """
        mg = self.microgrids[env_idx]

        # ============================================================
        # Step 1: Parse agent actions
        # ============================================================
        wind_bid_norm = actions[0, 0]
        solar_bid_norm = actions[1, 0]
        diesel_power_ratio = actions[2, 0]
        diesel_bid_norm = actions[2, 1]
        battery_action = actions[3, 0]
        battery_bid_norm = actions[3, 1]
        customer_curtail = actions[4, 0]

        # Denormalize bids to cents/kWh
        wind_ask = (wind_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        solar_ask = (solar_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        diesel_ask = (diesel_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min
        battery_bid_or_ask = (battery_bid_norm + 1) / 2 * (self.price_max - self.price_min) + self.price_min

        # Update component states with environmental data
        mg['wind'].update(1.0, {
            'wind_speed': env_data['wind_speed'],
            'bid': wind_ask
        })

        mg['pv'].update(1.0, {
            'solar_irradiance': env_data['solar_irradiance'],
            'bid': solar_ask
        })

        mg['diesel'].update(1.0, {
            'target_power': diesel_power_ratio * mg['diesel'].capacity,
            'bid': diesel_ask
        })

        mg['battery'].update(1.0, {
            'target_action': battery_action,
            'bid': battery_bid_or_ask
        })

        mg['load'].update(1.0, {
            'base_load': env_data['base_load'],
            'curtailment_ratio': (customer_curtail + 1) / 2 * self.max_customer_curtail
        })

        # Get available quantities from components
        wind_power = mg['wind'].available_power
        solar_power = mg['pv'].available_power
        diesel_power = mg['diesel'].target_power

        # Battery mode and power
        battery_component = mg['battery']
        if battery_action > 0:  # Discharge (sell)
            battery_power = min(
                battery_action * battery_component.max_discharge_rate_kw,
                battery_component.soc * battery_component.capacity_kwh
            )
            battery_mode = 'discharge'
        else:  # Charge (buy)
            battery_power = min(
                abs(battery_action) * battery_component.max_charge_rate_kw,
                (1 - battery_component.soc) * battery_component.capacity_kwh
            )
            battery_mode = 'charge'

        # Customer demand
        base_demand, curtailment_ratio, actual_demand = mg['load'].get_demand()

        # Get Main Grid prices
        grid_export_price, grid_import_price = _get_grid_price(env_data['time_hour'])

        # Get customer bid
        customer_bid = self._get_customer_bid(env_data['time_hour'], actual_demand)

        sell_orders = _build_sell_orders(
            env_idx, wind_power, solar_power, diesel_power, battery_power,
            battery_mode, wind_ask, solar_ask, diesel_ask, battery_bid_or_ask,
            grid_import_price
        )
        buy_orders = _build_buy_orders(
            env_idx, actual_demand, battery_power, battery_mode,
            battery_bid_or_ask, customer_bid, grid_export_price
        )

        matched_trades, clearing_price = self._match_orders(sell_orders, buy_orders)
        allocated_power = _aggregate_allocations(matched_trades, self.agent_names)

        # ============================================================
        # Step 6: Calculate grid import/export
        # ============================================================
        grid_import = 0.0
        grid_export = 0.0

        for trade in matched_trades:
            if trade['seller'] == 'main_grid':
                grid_import += trade['quantity']
            if trade['buyer'] == 'main_grid_buyer':
                grid_export += trade['quantity']

        main_grid_net = grid_import - grid_export

        # ============================================================
        # Step 7: Calculate market metrics
        # ============================================================
        total_supply = sum(allocated_power.get(name, 0)
                           for name in ['wind', 'solar', 'diesel', 'main_grid'])
        if battery_mode == 'discharge':
            total_supply += allocated_power.get('battery', 0)

        curtailed_load = base_demand - actual_demand

        market_efficiency = len(matched_trades) / max(len(sell_orders) + len(buy_orders), 1)

        # ============================================================
        # Step 8: Return auction results
        # ============================================================
        return {
            'clearing_price': clearing_price,
            'allocated_power': allocated_power,
            'actual_demand': actual_demand,
            'curtailed_load': curtailed_load,
            'grid_import': grid_import,
            'grid_export': grid_export,
            'battery_mode': battery_mode,
            'total_supply': total_supply,
            'matched_trades': matched_trades,
            'main_grid_trade': main_grid_net,
            'market_efficiency': market_efficiency,
            'num_trades': len(matched_trades),
            'price_spread': grid_import_price - grid_export_price,
            'bids': {
                'wind': wind_ask,
                'solar': solar_ask,
                'diesel': diesel_ask,
                'battery': battery_bid_or_ask,
                'customer': customer_bid
            }
        }
    
    def _default_reward_function(self, env_idx: int, auction_results: Dict, env_data: Dict,
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

        # Customer (Paper Equation 5) - 修复：使用实际交易价格计算成本
        matched_trades = auction_results['matched_trades']  # 新增：获取交易记录
        actual_demand = auction_results['actual_demand']
        curtailed = auction_results['curtailed_load']

        # 修改：计算Customer实际支付的电费
        # 遍历所有交易，累加Customer作为买方的实际支付
        electricity_cost = 0.0
        for trade in matched_trades:
            if trade['buyer'] == 'customer':
                # Customer支付的是每笔交易的实际价格
                # 从Wind/Solar/Diesel购买：使用卖家报价
                # 从Main Grid购买：使用grid_import_price
                electricity_cost += trade['quantity'] * trade['price'] / 100.0

        K = 10.0
        discomfort_cost = K * curtailed / 100.0
        base_expense = electricity_cost + discomfort_cost

        # 绿色能源折扣计算（修改：基于实际购买量）
        if use_policy_incentive:
            wind_power = allocated.get('wind', 0)
            solar_power = allocated.get('solar', 0)

            # 计算Customer实际购买的总电量
            total_customer_purchase = sum(
                trade['quantity'] for trade in matched_trades
                if trade['buyer'] == 'customer'
            )

            if total_customer_purchase > 0:
                # 计算可再生能源占Customer购买量的比例
                green_ratio = (wind_power + solar_power) / total_customer_purchase
                green_discount = electricity_cost * green_ratio * GREEN_DISCOUNT
                base_expense -= green_discount

        rewards[4] = -base_expense
        
        return rewards
    
    def _get_observation(self, env_idx: int, env_data: Dict, auction_results: Dict = None) -> np.ndarray:
        """Get observations (5D unified across agents)"""
        mg = self.microgrids[env_idx]
        time_hour = env_data['time_hour']
        time_normalized = time_hour / 24.0
        
        _, grid_price = _get_grid_price(time_hour)
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
            mg['diesel'].fuel_level / 1000.0,
            diesel_power / 75.0,
            time_normalized,
            price_normalized,
            actual_demand / 150.0
        ]
        
        # Battery (5D)
        obs[3, :] = [
            mg['battery'].soc,
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
    
    def step(self, actions: np.ndarray) -> tuple[
        ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], list[Any]]:
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
            info = {
                'bids': auction_results['bids'],
                'generation': {
                    'wind': auction_results['allocated_power'].get('wind', 0),
                    'solar': auction_results['allocated_power'].get('solar', 0),
                    'diesel': auction_results['allocated_power'].get('diesel', 0),
                    'battery': auction_results['allocated_power'].get('battery', 0),
                    'main_grid_import': auction_results.get('grid_import', 0),
                    'main_grid_export': auction_results.get('grid_export', 0)
                },
                'clearing_price': auction_results['clearing_price']
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