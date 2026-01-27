"""
Energy system simulation module using oemof.solph.
Models the hydrogen/electricity flows between nodes.
"""

import pandas as pd
from oemof import solph
from oemof.solph import views
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


@dataclass
class SimulationConfig:
    """Configuration for the energy system simulation."""
    # Time parameters
    periods: int = 24  # Number of time periods (e.g., hours)
    freq: str = 'H'    # Frequency (hourly)

    # Alpha: Electricity production capacity (kW)
    alpha_capacity: float = 100.0

    # Bravo: Electrolyzer parameters
    bravo_capacity: float = 50.0       # Max input power (kW)
    bravo_efficiency: float = 0.7      # Conversion efficiency (70%)

    # Charlie: Hydrogen demand (kg/period)
    charlie_demand: float = 10.0

    # Delta: Electricity demand (kW)
    delta_demand: float = 30.0


class HydrogenSystemSimulation:
    """
    Simulation of the hydrogen/electricity system using oemof.

    System layout:
        Alpha (Source) --> Electricity Bus --> Bravo (Transformer) --> Hydrogen Bus --> Charlie (Sink)
                                |
                                +--> Delta (Sink)
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        self.energy_system = None
        self.model = None
        self.results = None
        self._components = {}

    def build_system(self) -> None:
        """Build the oemof energy system model."""
        # Create date range for simulation
        date_range = pd.date_range(
            start='2024-01-01',
            periods=self.config.periods,
            freq=self.config.freq
        )

        # Create energy system
        self.energy_system = solph.EnergySystem(
            timeindex=date_range,
            infer_last_interval=False
        )

        # Create buses (energy carriers)
        bus_electricity = solph.Bus(label='electricity_bus')
        bus_hydrogen = solph.Bus(label='hydrogen_bus')

        # Alpha: Electricity producer (e.g., renewable source)
        # Using variable output to simulate production pattern
        alpha_output_profile = self._generate_production_profile()

        alpha = solph.components.Source(
            label='Alpha',
            outputs={
                bus_electricity: solph.Flow(
                    nominal_value=self.config.alpha_capacity,
                    fix=alpha_output_profile,
                )
            }
        )

        # Bravo: Electrolyzer (converts electricity to hydrogen)
        bravo = solph.components.Converter(
            label='Bravo',
            inputs={bus_electricity: solph.Flow()},
            outputs={
                bus_hydrogen: solph.Flow(
                    nominal_value=self.config.bravo_capacity
                )
            },
            conversion_factors={
                bus_hydrogen: self.config.bravo_efficiency
            }
        )

        # Charlie: Hydrogen consumer
        charlie_demand_profile = self._generate_hydrogen_demand_profile()

        charlie = solph.components.Sink(
            label='Charlie',
            inputs={
                bus_hydrogen: solph.Flow(
                    nominal_value=self.config.charlie_demand,
                    fix=charlie_demand_profile,
                )
            }
        )

        # Delta: Electricity consumer
        delta_demand_profile = self._generate_electricity_demand_profile()

        delta = solph.components.Sink(
            label='Delta',
            inputs={
                bus_electricity: solph.Flow(
                    nominal_value=self.config.delta_demand,
                    fix=delta_demand_profile,
                )
            }
        )

        # Excess sink for unused electricity
        excess = solph.components.Sink(
            label='Excess',
            inputs={
                bus_electricity: solph.Flow()
            }
        )

        # Store components for reference
        self._components = {
            'bus_electricity': bus_electricity,
            'bus_hydrogen': bus_hydrogen,
            'Alpha': alpha,
            'Bravo': bravo,
            'Charlie': charlie,
            'Delta': delta,
            'Excess': excess,
        }

        # Add all components to energy system
        self.energy_system.add(
            bus_electricity, bus_hydrogen,
            alpha, bravo, charlie, delta, excess
        )

        print("Energy system built successfully.")

    def _generate_production_profile(self) -> list:
        """Generate electricity production profile (e.g., solar pattern)."""
        # Simple sinusoidal pattern simulating solar production
        import math
        profile = []
        for t in range(self.config.periods):
            # Peak at midday (t=12), zero at night
            value = max(0, math.sin(math.pi * t / self.config.periods))
            profile.append(value)
        return profile

    def _generate_hydrogen_demand_profile(self) -> list:
        """Generate hydrogen demand profile."""
        # Constant demand with slight variation
        import random
        random.seed(42)
        return [0.8 + 0.2 * random.random() for _ in range(self.config.periods)]

    def _generate_electricity_demand_profile(self) -> list:
        """Generate electricity demand profile."""
        # Higher demand during working hours
        profile = []
        for t in range(self.config.periods):
            if 8 <= t <= 18:  # Working hours
                profile.append(0.9)
            else:
                profile.append(0.4)
        return profile

    def solve(self) -> bool:
        """Solve the optimization model."""
        if self.energy_system is None:
            raise RuntimeError("Energy system not built. Call build_system() first.")

        # Create model
        self.model = solph.Model(self.energy_system)

        # Solve (using HiGHS solver - pip install highspy)
        print("Solving optimization model...")
        self.model.solve(solver='highs', solve_kwargs={'tee': False})

        # Get results
        self.results = solph.processing.results(self.model)

        print("Model solved successfully.")
        return True

    def get_flow_results(self) -> Dict[str, pd.DataFrame]:
        """Extract flow results from the solved model."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        flows = {}

        # Electricity flows
        bus_el = self._components['bus_electricity']

        # Alpha -> Electricity Bus
        alpha_flow = views.node(self.results, 'Alpha')
        if alpha_flow is not None:
            flows['Alpha_production'] = alpha_flow['sequences']

        # Electricity Bus -> Bravo
        bravo_flow = views.node(self.results, 'Bravo')
        if bravo_flow is not None:
            flows['Bravo_conversion'] = bravo_flow['sequences']

        # Electricity Bus -> Delta
        delta_flow = views.node(self.results, 'Delta')
        if delta_flow is not None:
            flows['Delta_consumption'] = delta_flow['sequences']

        # Hydrogen Bus -> Charlie
        charlie_flow = views.node(self.results, 'Charlie')
        if charlie_flow is not None:
            flows['Charlie_consumption'] = charlie_flow['sequences']

        return flows

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of the simulation."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        summary = {}

        # Get total flows
        try:
            alpha_data = views.node(self.results, 'Alpha')
            if alpha_data is not None and 'sequences' in alpha_data:
                summary['total_electricity_produced'] = float(
                    alpha_data['sequences'].sum().sum()
                )

            bravo_data = views.node(self.results, 'Bravo')
            if bravo_data is not None and 'sequences' in bravo_data:
                summary['total_electricity_to_hydrogen'] = float(
                    bravo_data['sequences'].sum().sum()
                )

            charlie_data = views.node(self.results, 'Charlie')
            if charlie_data is not None and 'sequences' in charlie_data:
                summary['total_hydrogen_consumed'] = float(
                    charlie_data['sequences'].sum().sum()
                )

            delta_data = views.node(self.results, 'Delta')
            if delta_data is not None and 'sequences' in delta_data:
                summary['total_electricity_consumed'] = float(
                    delta_data['sequences'].sum().sum()
                )
        except Exception as e:
            print(f"Warning: Could not extract all summary data: {e}")

        summary['electrolyzer_efficiency'] = self.config.bravo_efficiency

        return summary

    def export_results_to_json(self, filepath: str) -> None:
        """Export simulation results to JSON for Blender visualization."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        # Get flow data
        flows = self.get_flow_results()

        # Convert to serializable format
        export_data = {
            'config': {
                'periods': self.config.periods,
                'alpha_capacity': self.config.alpha_capacity,
                'bravo_efficiency': self.config.bravo_efficiency,
                'charlie_demand': self.config.charlie_demand,
                'delta_demand': self.config.delta_demand,
            },
            'summary': self.get_summary(),
            'flows': {}
        }

        for name, df in flows.items():
            export_data['flows'][name] = df.values.tolist()

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Results exported to {filepath}")

    def print_results(self) -> None:
        """Print simulation results summary."""
        summary = self.get_summary()

        print("\n=== Simulation Results ===")
        print(f"Simulation periods: {self.config.periods}")
        print(f"\nEnergy Flows:")

        for key, value in summary.items():
            if key != 'electrolyzer_efficiency':
                print(f"  {key}: {value:.2f} kWh/kg")

        print(f"\nElectrolyzer efficiency: {summary.get('electrolyzer_efficiency', 0) * 100:.1f}%")


def run_simulation(config: Optional[SimulationConfig] = None) -> HydrogenSystemSimulation:
    """Run a complete simulation with the given configuration."""
    sim = HydrogenSystemSimulation(config)
    sim.build_system()
    sim.solve()
    return sim


if __name__ == "__main__":
    # Run test simulation
    config = SimulationConfig(
        periods=24,
        alpha_capacity=100,
        bravo_efficiency=0.7,
        charlie_demand=10,
        delta_demand=30,
    )

    sim = run_simulation(config)
    sim.print_results()
    sim.export_results_to_json("simulation_results.json")
