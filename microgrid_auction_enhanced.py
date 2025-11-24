# -*- coding: utf-8 -*-
"""
Enhanced Microgrid with Full 5-Agent Auction Mechanism

This version extends the original microgrid_auction.py to support:
- Wind and Solar agents submitting bids
- Diesel agent controlling power output and bidding
- Battery agent bidding for charging/discharging
- Customer agent with demand response (curtailment)
- Uniform price auction market clearing
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class MicrogridComponent:
    """Base class for all microgrid components."""
    def __init__(self, name):
        self.name = name
        self.power_output = 0  # Represents power output (positive) or consumption (negative)

    def update(self, timestep, data=None):
        """
        Update the component's state and power output based on the current timestep and data.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the update method")

    def get_power_output(self):
        """Get the current power output of the component."""
        return self.power_output

    def __str__(self):
        return f"{self.name}: {self.power_output:.2f} kW"


class WindTurbine(MicrogridComponent):
    """Represents a wind turbine with bidding capability."""
    def __init__(self, name, capacity):
        super().__init__(name)
        self.capacity = capacity  # Maximum power output in kW
        self.current_bid = 0.0  # Current bid price in cents/kWh
        self.available_power = 0.0  # Available power based on wind conditions

    def update(self, timestep, data=None):
        """
        Update the wind turbine's power output based on wind speed data.

        Args:
            timestep: The current simulation timestep.
            data: A dictionary containing relevant data, e.g., 
                  {'wind_speed': float, 'bid': float, 'accepted_power': float}.
        """
        if data and 'wind_speed' in data:
            wind_speed = data['wind_speed']
            # Simple wind power model
            cut_in_speed = 3  # m/s
            rated_speed = 12  # m/s
            cut_out_speed = 25  # m/s

            if wind_speed < cut_in_speed or wind_speed > cut_out_speed:
                self.available_power = 0
            elif wind_speed >= rated_speed:
                self.available_power = self.capacity
            else:
                # Linear interpolation between cut-in and rated speed
                self.available_power = self.capacity * (wind_speed - cut_in_speed) / (rated_speed - cut_in_speed)
        else:
            self.available_power = 0

        # Update bid if provided
        if data and 'bid' in data:
            self.current_bid = data['bid']

        # Update actual power output based on what was accepted in auction
        if data and 'accepted_power' in data:
            self.power_output = min(data['accepted_power'], self.available_power)
        else:
            self.power_output = self.available_power  # Default: offer all available

    def get_bid_offer(self) -> Tuple[float, float]:
        """Get the current bid and available power for auction."""
        return self.current_bid, self.available_power


class PVSystem(MicrogridComponent):
    """Represents a PV (solar) system with bidding capability."""
    def __init__(self, name, capacity):
        super().__init__(name)
        self.capacity = capacity  # Maximum power output in kW
        self.current_bid = 0.0  # Current bid price in cents/kWh
        self.available_power = 0.0  # Available power based on solar irradiance

    def update(self, timestep, data=None):
        """
        Update the PV system's power output based on solar irradiance data.

        Args:
            timestep: The current simulation timestep.
            data: A dictionary containing relevant data, e.g., 
                  {'solar_irradiance': float, 'bid': float, 'accepted_power': float}.
        """
        if data and 'solar_irradiance' in data:
            solar_irradiance = data['solar_irradiance']  # W/m^2
            # Simple PV power model
            stc_irradiance = 1000
            self.available_power = self.capacity * (solar_irradiance / stc_irradiance)
            # Ensure output does not exceed capacity and is not negative
            self.available_power = max(0, min(self.capacity, self.available_power))
        else:
            self.available_power = 0

        # Update bid if provided
        if data and 'bid' in data:
            self.current_bid = data['bid']

        # Update actual power output based on what was accepted in auction
        if data and 'accepted_power' in data:
            self.power_output = min(data['accepted_power'], self.available_power)
        else:
            self.power_output = self.available_power

    def get_bid_offer(self) -> Tuple[float, float]:
        """Get the current bid and available power for auction."""
        return self.current_bid, self.available_power


class DieselGenerator(MicrogridComponent):
    """Represents a diesel generator with power control and bidding."""
    def __init__(self, name, capacity, fuel_consumption_rate=0.2, generation_cost_per_kwh=0.08):
        super().__init__(name)
        self.capacity = capacity  # Maximum power output in kW
        self.fuel_consumption_rate = fuel_consumption_rate  # Fuel consumed per kWh generated
        self.generation_cost_per_kwh = generation_cost_per_kwh  # Cost to generate 1 kWh
        self.fuel_level = 1000  # Initial fuel level
        self.is_running = False
        self.current_bid = 0.0  # Current bid price in cents/kWh
        self.target_power = 0.0  # Target power output (from agent's decision)

    def update(self, timestep, data=None):
        """
        Update the diesel generator's power output and fuel level.

        Args:
            timestep: The current simulation timestep (in hours).
            data: A dictionary containing relevant data, e.g., 
                  {'target_power': float, 'bid': float, 'accepted_power': float}.
        """
        # Update target power if provided (from agent)
        if data and 'target_power' in data:
            self.target_power = data['target_power']

        # Update bid if provided
        if data and 'bid' in data:
            self.current_bid = data['bid']

        # Determine actual power output
        if data and 'accepted_power' in data:
            # Auction determined how much we can generate
            required_power = data['accepted_power']
        else:
            # Use target power
            required_power = self.target_power

        # Ensure diesel only runs if fuel is available and required_power is positive
        if required_power > 0 and self.fuel_level > 0:
            self.is_running = True
            # Generate power up to capacity or required power, whichever is less
            power_to_generate = min(self.capacity, required_power)

            # Check if enough fuel to generate power_to_generate for the timestep
            max_power_from_fuel_kwh = (self.fuel_level / self.fuel_consumption_rate) if self.fuel_consumption_rate > 0 else float('inf')
            max_power_from_fuel_for_timestep = max_power_from_fuel_kwh / timestep if timestep > 0 else float('inf')

            self.power_output = min(power_to_generate, max_power_from_fuel_for_timestep)

            # Calculate fuel consumption based on actual generated power
            fuel_consumed = self.power_output * self.fuel_consumption_rate * timestep
            self.fuel_level -= fuel_consumed
            if self.fuel_level < 0:
                self.fuel_level = 0
                self.power_output = 0
        else:
            self.is_running = False
            self.power_output = 0

    def get_bid_offer(self) -> Tuple[float, float]:
        """Get the current bid and available power for auction."""
        # Available power is limited by capacity and fuel
        max_available = min(self.capacity, self.target_power)
        return self.current_bid, max_available

    def get_fuel_level(self):
        """Get the current fuel level."""
        return self.fuel_level

    def __str__(self):
        return f"{self.name}: {self.power_output:.2f} kW, Fuel: {self.fuel_level:.2f}"


class Battery(MicrogridComponent):
    """Represents an energy storage system (battery) with bidding for charge/discharge."""
    def __init__(self, name, capacity_kwh, max_charge_rate_kw, max_discharge_rate_kw, 
                 initial_soc=0.5, charge_cost_per_kwh=0.05, discharge_cost_per_kwh=0.15):
        super().__init__(name)
        self.capacity_kwh = capacity_kwh  # Total energy capacity in kWh
        self.max_charge_rate_kw = max_charge_rate_kw  # Maximum charging power in kW
        self.max_discharge_rate_kw = max_discharge_rate_kw  # Maximum discharging power in kW
        self.soc = initial_soc  # State of Charge (0.0 to 1.0)
        self.charge_cost_per_kwh = charge_cost_per_kwh  # Cost to charge 1 kWh
        self.discharge_cost_per_kwh = discharge_cost_per_kwh  # Cost of discharging 1 kWh
        self.power_output = 0  # Power output (discharge is positive, charge is negative)
        self.current_bid = 0.0  # Current bid price in cents/kWh
        self.target_action = 0.0  # Target charge/discharge (-1 to 1, from agent)

    def update(self, timestep, data=None):
        """
        Update the battery's state of charge and power output.

        Args:
            timestep: The current simulation timestep (in hours).
            data: A dictionary containing relevant data, e.g., 
                  {'target_action': float, 'bid': float, 'accepted_power': float}.
                  target_action: -1 to 1 (negative = charge, positive = discharge)
        """
        # Update target action if provided (from agent)
        if data and 'target_action' in data:
            self.target_action = data['target_action']

        # Update bid if provided
        if data and 'bid' in data:
            self.current_bid = data['bid']

        # Determine actual power based on auction or target
        if data and 'accepted_power' in data:
            requested_power = data['accepted_power']  # Can be positive (discharge) or negative (charge)
        else:
            # Use target action to determine requested power
            if self.target_action > 0:  # Discharge
                requested_power = self.target_action * self.max_discharge_rate_kw
            else:  # Charge
                requested_power = self.target_action * self.max_charge_rate_kw  # Will be negative

        if requested_power > 0:  # Discharge
            # Limit requested discharge by max rate and available energy
            max_available_discharge_energy_kwh = self.soc * self.capacity_kwh
            max_discharge_power_possible = max_available_discharge_energy_kwh / timestep if timestep > 0 else float('inf')

            actual_power = min(requested_power, self.max_discharge_rate_kw, max_discharge_power_possible)

            energy_change_kwh = actual_power * timestep
            self.soc -= (energy_change_kwh / self.capacity_kwh)
            self.power_output = actual_power

        elif requested_power < 0:  # Charge
            requested_charge_power = -requested_power  # Convert to positive value for calculation
            # Limit requested charge by max rate and available capacity
            max_available_charge_energy_kwh = (1.0 - self.soc) * self.capacity_kwh
            max_charge_power_possible = max_available_charge_energy_kwh / timestep if timestep > 0 else float('inf')

            actual_power = min(requested_charge_power, self.max_charge_rate_kw, max_charge_power_possible)

            energy_change_kwh = actual_power * timestep
            self.soc += (energy_change_kwh / self.capacity_kwh)
            self.power_output = -actual_power  # Charging is negative power output

        else:
            self.power_output = 0  # No charge/discharge command

        # Ensure SOC is within bounds
        self.soc = max(0.0, min(1.0, self.soc))

    def get_bid_offer(self, mode: str) -> Tuple[float, float]:
        """
        Get the current bid and available power for auction.
        
        Args:
            mode: 'discharge' or 'charge'
            
        Returns:
            (bid_price, available_power)
        """
        if mode == 'discharge':
            max_available = min(
                self.max_discharge_rate_kw,
                self.soc * self.capacity_kwh  # Available energy
            )
            return self.current_bid, max_available
        elif mode == 'charge':
            max_available = min(
                self.max_charge_rate_kw,
                (1.0 - self.soc) * self.capacity_kwh  # Available capacity
            )
            return self.current_bid, max_available
        else:
            return 0.0, 0.0

    def get_soc(self):
        """Get the current state of charge (SOC)."""
        return self.soc

    def __str__(self):
        return f"{self.name}: {self.power_output:.2f} kW, SOC: {self.soc:.2f}"


class MainGrid(MicrogridComponent):
    """Represents the connection to the external power grid."""
    def __init__(self, name, import_limit=float('inf'), export_limit=float('inf'),
                 import_price_per_kwh=0.25, export_price_per_kwh=0.1):
        super().__init__(name)
        self.import_limit = import_limit  # Maximum power that can be imported (kW)
        self.export_limit = export_limit  # Maximum power that can be exported (kW)
        self.import_price_per_kwh = import_price_per_kwh  # Cost to import 1 kWh
        self.export_price_per_kwh = export_price_per_kwh  # Revenue from exporting 1 kWh
        self.power_output = 0  # Power flow (positive for import, negative for export)

    def update(self, timestep, data=None):
        """
        Update the main grid's power flow.

        Args:
            timestep: The current simulation timestep.
            data: A dictionary containing relevant data, e.g., {'requested_power': float}.
                  Positive values for importing from the grid, negative for exporting.
        """
        requested_power = data.get('requested_power', 0) if data else 0

        if requested_power > 0:  # Requesting power from the grid (import)
            self.power_output = min(requested_power, self.import_limit)
        elif requested_power < 0:  # Sending power to the grid (export)
            self.power_output = max(requested_power, -self.export_limit)
        else:
            self.power_output = 0

    def __str__(self):
        flow_direction = "importing" if self.power_output > 0 else ("exporting" if self.power_output < 0 else "not exchanging")
        return f"{self.name}: {abs(self.power_output):.2f} kW ({flow_direction})"


class CustomerLoad(MicrogridComponent):
    """Represents the aggregate energy consumption of customers with demand response."""
    def __init__(self, name):
        super().__init__(name)
        self.base_demand = 0  # Base load demand in kW
        self.curtailment_ratio = 0.0  # Curtailment ratio (0 to 1)
        self.actual_consumption = 0  # Actual consumption after curtailment
        self.power_output = 0  # Load is power consumption (negative in our convention)

    def update(self, timestep, data=None):
        """
        Update the customer load based on load data and curtailment.

        Args:
            timestep: The current simulation timestep.
            data: A dictionary containing relevant data, e.g., 
                  {'base_load': float, 'curtailment_ratio': float}.
        """
        if data and 'base_load' in data:
            self.base_demand = data['base_load']
        
        if data and 'curtailment_ratio' in data:
            self.curtailment_ratio = max(0.0, min(1.0, data['curtailment_ratio']))
        
        # Calculate actual consumption after curtailment
        self.actual_consumption = self.base_demand * (1 - self.curtailment_ratio)
        
        # Load is consumption, so it's negative power output
        self.power_output = -self.actual_consumption

    def get_demand(self) -> Tuple[float, float, float]:
        """
        Get demand information.
        
        Returns:
            (base_demand, curtailment_ratio, actual_consumption)
        """
        return self.base_demand, self.curtailment_ratio, self.actual_consumption

    def __str__(self):
        return f"{self.name}: {self.actual_consumption:.2f} kW (base: {self.base_demand:.2f} kW, curtailed: {self.curtailment_ratio*100:.1f}%)"


class AuctionResult:
    """Data class to store auction results."""
    def __init__(self):
        self.clearing_price = 0.0
        self.allocated_power = {}  # Dict[agent_name, power]
        self.total_supply = 0.0
        self.total_demand = 0.0
        self.grid_import = 0.0
        self.grid_export = 0.0
        self.suppliers = []  # List of supplier bids
        self.unmet_demand = 0.0


class MicrogridAuction:
    """
    Enhanced Microgrid with Full Auction Mechanism
    
    Manages the uniform price auction for electricity trading among:
    - Wind, Solar, Diesel (sellers)
    - Battery (buyer or seller)
    - Customer (buyer with demand response)
    - Main Grid (balances the system)
    """
    
    def __init__(self, components: Dict[str, MicrogridComponent]):
        """
        Args:
            components: Dictionary of MicrogridComponent objects
        """
        self.components = components
        self.time = 0
        self.history = []
        
    def run_auction(self, timestep: float, agent_actions: Dict[str, Dict]) -> AuctionResult:
        """
        Run the uniform price auction for the current timestep.
        
        Args:
            timestep: Duration of timestep in hours
            agent_actions: Dictionary of actions for each agent
                Example: {
                    'wind': {'bid': 15.0},
                    'solar': {'bid': 18.0},
                    'diesel': {'target_power': 50.0, 'bid': 25.0},
                    'battery': {'target_action': 0.5, 'bid': 20.0},  # Positive = discharge
                    'customer': {'curtailment_ratio': 0.1}
                }
                
        Returns:
            AuctionResult object with clearing price and allocations
        """
        result = AuctionResult()
        
        # Build list of suppliers (sellers)
        suppliers = []
        
        # Wind
        if 'wind' in self.components:
            wind = self.components['wind']
            bid, available = wind.get_bid_offer()
            if available > 0:
                suppliers.append({
                    'name': 'wind',
                    'component': wind,
                    'bid': bid,
                    'power': available,
                    'cost': 0.0  # Marginal cost
                })
        
        # Solar
        if 'pv' in self.components:
            pv = self.components['pv']
            bid, available = pv.get_bid_offer()
            if available > 0:
                suppliers.append({
                    'name': 'pv',
                    'component': pv,
                    'bid': bid,
                    'power': available,
                    'cost': 0.0
                })
        
        # Diesel
        if 'diesel' in self.components:
            diesel = self.components['diesel']
            bid, available = diesel.get_bid_offer()
            if available > 0:
                suppliers.append({
                    'name': 'diesel',
                    'component': diesel,
                    'bid': bid,
                    'power': available,
                    'cost': diesel.generation_cost_per_kwh
                })
        
        # Battery (if discharging)
        battery_mode = None
        if 'battery' in self.components:
            battery = self.components['battery']
            target_action = agent_actions.get('battery', {}).get('target_action', 0)
            if target_action > 0:  # Discharging (selling)
                battery_mode = 'discharge'
                bid, available = battery.get_bid_offer('discharge')
                if available > 0:
                    suppliers.append({
                        'name': 'battery',
                        'component': battery,
                        'bid': bid,
                        'power': available,
                        'cost': battery.discharge_cost_per_kwh
                    })
            elif target_action < 0:  # Charging (buying)
                battery_mode = 'charge'
        
        # Sort suppliers by bid price (ascending)
        suppliers.sort(key=lambda x: x['bid'])
        result.suppliers = suppliers
        
        # Calculate total demand
        customer = self.components.get('load')
        if customer:
            base_demand, curtailment_ratio, actual_demand = customer.get_demand()
            total_demand = actual_demand
        else:
            total_demand = 0
        
        # Add battery charging demand if applicable
        battery_charge_demand = 0
        if battery_mode == 'charge' and 'battery' in self.components:
            battery = self.components['battery']
            _, max_charge = battery.get_bid_offer('charge')
            battery_charge_demand = max_charge
            total_demand += battery_charge_demand
        
        result.total_demand = total_demand
        
        # Market clearing
        total_supply = sum(s['power'] for s in suppliers)
        result.total_supply = total_supply
        
        if total_demand <= 0:
            # No demand
            result.clearing_price = 5.0  # Minimum price
            result.allocated_power = {s['name']: 0 for s in suppliers}
            result.grid_export = total_supply  # Export all supply
        elif total_supply >= total_demand:
            # Sufficient supply - find marginal supplier
            cumulative = 0
            result.clearing_price = 5.0  # Minimum price
            result.allocated_power = {}
            
            for supplier in suppliers:
                if cumulative >= total_demand:
                    result.allocated_power[supplier['name']] = 0
                elif cumulative + supplier['power'] <= total_demand:
                    # Fully allocated
                    result.allocated_power[supplier['name']] = supplier['power']
                    cumulative += supplier['power']
                    result.clearing_price = supplier['bid']
                else:
                    # Partially allocated (marginal supplier)
                    result.allocated_power[supplier['name']] = total_demand - cumulative
                    cumulative = total_demand
                    result.clearing_price = supplier['bid']
            
            # No grid import needed
            result.grid_import = 0
        else:
            # Insufficient supply - all accepted, need grid import
            result.clearing_price = max(s['bid'] for s in suppliers) + 5.0  # Penalty
            result.allocated_power = {s['name']: s['power'] for s in suppliers}
            result.grid_import = total_demand - total_supply
            result.unmet_demand = 0
        
        # Handle battery charging allocation
        if battery_mode == 'charge':
            net_supply = sum(result.allocated_power.values())
            if net_supply + result.grid_import >= battery_charge_demand + actual_demand:
                # Enough for both customer and battery
                result.allocated_power['battery'] = -battery_charge_demand  # Negative = charging
            else:
                # Not enough - prioritize customer load
                available_for_battery = net_supply + result.grid_import - actual_demand
                result.allocated_power['battery'] = -max(0, available_for_battery)
        
        return result
    
    def update(self, timestep: float, agent_actions: Dict[str, Dict], env_data: Dict):
        """
        Update the microgrid state using auction mechanism.
        
        Args:
            timestep: Duration of timestep in hours
            agent_actions: Actions from all agents
            env_data: Environmental data (weather, base load, etc.)
        """
        current_state = {'time': self.time}
        
        # Update components with environmental data and agent actions
        for name, component in self.components.items():
            if name == 'wind':
                component.update(timestep, {
                    'wind_speed': env_data.get('wind_speed', 0),
                    'bid': agent_actions.get('wind', {}).get('bid', 15.0)
                })
            elif name == 'pv':
                component.update(timestep, {
                    'solar_irradiance': env_data.get('solar_irradiance', 0),
                    'bid': agent_actions.get('solar', {}).get('bid', 18.0)
                })
            elif name == 'diesel':
                diesel_actions = agent_actions.get('diesel', {})
                component.update(timestep, {
                    'target_power': diesel_actions.get('target_power', 0),
                    'bid': diesel_actions.get('bid', 25.0)
                })
            elif name == 'battery':
                battery_actions = agent_actions.get('battery', {})
                component.update(timestep, {
                    'target_action': battery_actions.get('target_action', 0),
                    'bid': battery_actions.get('bid', 20.0)
                })
            elif name == 'load':
                customer_actions = agent_actions.get('customer', {})
                component.update(timestep, {
                    'base_load': env_data.get('base_load', 100),
                    'curtailment_ratio': customer_actions.get('curtailment_ratio', 0)
                })
        
        # Run auction
        auction_result = self.run_auction(timestep, agent_actions)
        
        # Update components with auction results
        for name, allocated_power in auction_result.allocated_power.items():
            if name in self.components:
                self.components[name].update(timestep, {
                    **agent_actions.get(name, {}),
                    'accepted_power': allocated_power
                })
        
        # Update main grid
        if 'grid' in self.components:
            grid_power = auction_result.grid_import - auction_result.grid_export
            self.components['grid'].update(timestep, {'requested_power': grid_power})
        
        # Store state
        current_state['clearing_price'] = auction_result.clearing_price
        current_state['total_supply'] = auction_result.total_supply
        current_state['total_demand'] = auction_result.total_demand
        current_state['grid_import'] = auction_result.grid_import
        current_state['grid_export'] = auction_result.grid_export
        
        for name, component in self.components.items():
            current_state[f'{name}_power'] = component.get_power_output()
            if name == 'battery':
                current_state['battery_soc'] = component.get_soc()
        
        self.history.append(current_state)
        self.time += timestep
    
    def get_history(self):
        """Get simulation history."""
        return self.history


if __name__ == "__main__":
    """Test the enhanced auction-based microgrid"""
    print("=" * 60)
    print("🧪 测试增强版 Microgrid Auction")
    print("=" * 60)
    
    # Create components
    wind = WindTurbine(name='WindTurbine', capacity=100)
    pv = PVSystem(name='PVSystem', capacity=50)
    diesel = DieselGenerator(name='DieselGenerator', capacity=75, 
                            fuel_consumption_rate=0.2, generation_cost_per_kwh=0.08)
    battery = Battery(name='Battery', capacity_kwh=200, 
                     max_charge_rate_kw=50, max_discharge_rate_kw=50,
                     initial_soc=0.5, charge_cost_per_kwh=0.05, 
                     discharge_cost_per_kwh=0.15)
    grid = MainGrid(name='MainGrid', import_limit=100, export_limit=100,
                   import_price_per_kwh=0.25, export_price_per_kwh=0.1)
    load = CustomerLoad(name='CustomerLoad')
    
    components = {
        'wind': wind,
        'pv': pv,
        'diesel': diesel,
        'battery': battery,
        'grid': grid,
        'load': load
    }
    
    # Create microgrid auction system
    mg_auction = MicrogridAuction(components)
    
    # Test scenario 1: High renewable generation, low load
    print("\n✅ Scenario 1: High renewables, low load")
    env_data = {
        'wind_speed': 10,
        'solar_irradiance': 800,
        'base_load': 60
    }
    agent_actions = {
        'wind': {'bid': 12.0},
        'solar': {'bid': 15.0},
        'diesel': {'target_power': 30, 'bid': 25.0},
        'battery': {'target_action': -0.5, 'bid': 18.0},  # Charging
        'customer': {'curtailment_ratio': 0.0}
    }
    
    mg_auction.update(1.0, agent_actions, env_data)
    state = mg_auction.get_history()[-1]
    
    print(f"   Clearing Price: {state['clearing_price']:.2f} cents/kWh")
    print(f"   Wind Power: {state['wind_power']:.2f} kW")
    print(f"   Solar Power: {state['pv_power']:.2f} kW")
    print(f"   Diesel Power: {state['diesel_power']:.2f} kW")
    print(f"   Battery Power: {state['battery_power']:.2f} kW")
    print(f"   Customer Consumption: {-state['load_power']:.2f} kW")
    print(f"   Grid Import/Export: {state['grid_power']:.2f} kW")
    
    # Test scenario 2: Low renewable generation, high load
    print("\n✅ Scenario 2: Low renewables, high load")
    env_data = {
        'wind_speed': 3,
        'solar_irradiance': 100,
        'base_load': 120
    }
    agent_actions = {
        'wind': {'bid': 12.0},
        'solar': {'bid': 15.0},
        'diesel': {'target_power': 60, 'bid': 22.0},
        'battery': {'target_action': 0.8, 'bid': 20.0},  # Discharging
        'customer': {'curtailment_ratio': 0.1}  # 10% curtailment
    }
    
    mg_auction.update(1.0, agent_actions, env_data)
    state = mg_auction.get_history()[-1]
    
    print(f"   Clearing Price: {state['clearing_price']:.2f} cents/kWh")
    print(f"   Wind Power: {state['wind_power']:.2f} kW")
    print(f"   Solar Power: {state['pv_power']:.2f} kW")
    print(f"   Diesel Power: {state['diesel_power']:.2f} kW")
    print(f"   Battery Power: {state['battery_power']:.2f} kW")
    print(f"   Customer Consumption: {-state['load_power']:.2f} kW")
    print(f"   Grid Import/Export: {state['grid_power']:.2f} kW")
    print(f"   Battery SOC: {state['battery_soc']:.2f}")
    
    print("\n" + "=" * 60)
    print("🎉 增强版 Microgrid Auction 测试完成！")
    print("=" * 60)
