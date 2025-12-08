import numpy as np
from typing import Tuple


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
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity = config['capacity']
        self.cut_in_speed = config['cut_in_speed']
        self.rated_speed = config['rated_speed']
        self.cut_out_speed = config['cut_out_speed']

    def update(self, timestep, data=None):
        """
        Update the wind turbine's power output based on wind speed data.
        """
        if data and 'wind_speed' in data:
            wind_speed = data['wind_speed']

            if wind_speed < self.cut_in_speed or wind_speed > self.cut_out_speed:
                self.power_output = 0
            elif wind_speed >= self.rated_speed:
                self.power_output = self.capacity
            else:
                # Linear interpolation between cut-in and rated speed
                self.power_output = self.capacity * (wind_speed - self.cut_in_speed) / (self.rated_speed - self.cut_in_speed)
        else:
            self.power_output = 0


class PVSystem(MicrogridComponent):
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity = config['capacity']
        self.stc_irradiance = config['stc_irradiance']

    def update(self, timestep, data=None):
        """
        Update the PV system's power output based on solar irradiance data.
        """
        if data and 'solar_irradiance' in data:
            solar_irradiance = data['solar_irradiance']  # W/m^2
            self.power_output = self.capacity * (solar_irradiance / self.stc_irradiance)
            # Ensure output does not exceed capacity and is not negative
            self.power_output = max(0, min(self.capacity, self.power_output))
        else:
            self.power_output = 0


class DieselGenerator(MicrogridComponent):
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity = config['capacity']
        self.fuel_consumption_rate = config['fuel_consumption_rate']
        self.generation_cost_per_kwh = config['generation_cost_per_kwh']
        self.fuel_level = config['initial_fuel_level']
        self.is_running = False
        self.target_power = 0.0  # Target power output (from agent's decision)

    def update(self, timestep, data=None):
        """
        Update the diesel generator's power output and fuel level.
        """
        # Update target power if provided (from agent)
        if data and 'target_power' in data:
            self.target_power = data['target_power']

        # Ensure diesel only runs if fuel is available and target_power is positive
        if self.target_power > 0 and self.fuel_level > 0:
            self.is_running = True
            # Generate power up to capacity or required power, whichever is less
            power_to_generate = min(self.capacity, self.target_power)

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
    def __init__(self, name, config):
        super().__init__(name)
        self.capacity_kwh = config['capacity_kwh']
        self.max_charge_rate_kw = config['max_charge_rate_kw']
        self.max_discharge_rate_kw = config['max_discharge_rate_kw']
        self.soc = config['initial_soc']
        self.charge_cost_per_kwh = config['charge_cost_per_kwh']
        self.discharge_cost_per_kwh = config['discharge_cost_per_kwh']
        self.power_output = 0  # Power output (discharge is positive, charge is negative)
        self.target_action = 0.0  # Target charge/discharge (-1 to 1, from agent)

    def update(self, timestep, data=None):
        """
        Update the battery's state of charge and power output.
        """
        # Update target action if provided (from agent)
        if data and 'target_action' in data:
            self.target_action = data['target_action']

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
    def __init__(self, name, config):
        super().__init__(name)
        self.base_demand = 0  # Base load demand in kW
        self.max_curtailment = config['max_curtailment']
        self.curtailment_ratio = 0  # Curtailment ratio (0 to 1)
        self.actual_consumption = 0  # Actual consumption after curtailment

    def update(self, timestep, data=None):
        """
        Update the customer load based on load data and curtailment.
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
        return self.base_demand, self.curtailment_ratio, self.actual_consumption


def get_solar_irradiance(hour: int, name: str, config: dict) -> float:
    cfg = config['environment']['solar_generation']
    if hour < cfg['start_hour'] or hour > cfg['end_hour']:
        return 0
    stc_irradiance = 0
    for supplier in config['components']['suppliers']:
        if supplier['name'] == name:
            stc_irradiance = supplier['stc_irradiance']
    return stc_irradiance * np.sin(
        np.pi * (hour - cfg['start_hour']) / (cfg['end_hour'] - cfg['start_hour'])) * np.random.uniform(
        cfg['noise_min'], cfg['noise_max'])


def get_grid_price(hour: int, config: dict) -> Tuple[float, float]:
    pricing = config['environment']['grid_pricing']
    import_price = 0

    # Different import price for different hours
    found = False
    for hours_range in pricing['off_peak']['hours']:
        if hours_range[0] <= hour < hours_range[1]:
            import_price = np.random.uniform(*pricing['off_peak']['import_price_range'])
            found = True
            break

    if not found:
        for hours_range in pricing['mid_peak']['hours']:
            if hours_range[0] <= hour < hours_range[1]:
                import_price = np.random.uniform(*pricing['mid_peak']['import_price_range'])
                found = True
                break

    if not found:
        for hours_range in pricing['peak']['hours']:
            if hours_range[0] <= hour < hours_range[1]:
                import_price = np.random.uniform(*pricing['peak']['import_price_range'])
                break

    export_price = np.random.uniform(*pricing['export_price_range'])
    return export_price, import_price