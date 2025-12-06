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

    def __str__(self):
        return f"{self.name}: {self.power_output:.2f} kW"


class WindTurbine(MicrogridComponent):
    """Represents a wind turbine with bidding capability."""
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity = config['capacity']
        self.cut_in_speed = config['cut_in_speed']
        self.rated_speed = config['rated_speed']
        self.cut_out_speed = config['cut_out_speed']
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

            if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
                self.available_power = 0
            elif wind_speed >= self.rated_speed:
                self.available_power = self.capacity
            else:
                # Linear interpolation between cut-in and rated speed
                self.available_power = self.capacity * (wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)
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


class PVSystem(MicrogridComponent):
    """Represents a PV (solar) system with bidding capability."""
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity = config['capacity']
        self.stc_irradiance = config['stc_irradiance']
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
            self.available_power = self.capacity * (solar_irradiance / self.stc_irradiance)
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


class DieselGenerator(MicrogridComponent):
    """Represents a diesel generator with power control and bidding."""
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity = config['capacity']
        self.fuel_consumption_rate = config['fuel_consumption_rate']
        self.generation_cost_per_kwh = config['generation_cost_per_kwh']
        self.fuel_level = config['initial_fuel_level']
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

    def __str__(self):
        return f"{self.name}: {self.power_output:.2f} kW, Fuel: {self.fuel_level:.2f}"


class Battery(MicrogridComponent):
    """Represents an energy storage system (battery) with bidding for charge/discharge."""
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity_kwh = config['capacity_kwh']
        self.max_charge_rate_kw = config['max_charge_rate_kw']
        self.max_discharge_rate_kw = config['max_discharge_rate_kw']
        self.soc = config['initial_soc']
        self.charge_cost_per_kwh = config['charge_cost_per_kwh']
        self.discharge_cost_per_kwh = config['discharge_cost_per_kwh']
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

    def __str__(self):
        return f"{self.name}: {self.power_output:.2f} kW, SOC: {self.soc:.2f}"


class CustomerLoad(MicrogridComponent):
    """Represents the aggregate energy consumption of customers with demand response."""
    def __init__(self, name, config):
        super().__init__(name)
        self.base_demand = 0  # Base load demand in kW
        self.max_curtailment = config['max_curtailment']
        self.curtailment_ratio = 0  # Curtailment ratio (0 to 1)
        self.actual_consumption = 0  # Actual consumption after curtailment

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
            self.curtailment_ratio = max(0.0, min(self.max_curtailment, data['curtailment_ratio']))
        
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