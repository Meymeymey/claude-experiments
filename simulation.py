"""
Energy system simulation module using oemof.solph.
Models the hydrogen/electricity flows between nodes with price-based economics.

Economics:
- Producers have production costs (ct/kWh)
- Transformers have conversion costs (ct/kWh)
- Consumers have utility functions with diminishing marginal utility
  (modeled as demand tranches with decreasing willingness to pay)
"""

import pandas as pd
from oemof import solph
from oemof.solph import views
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import json


@dataclass
class DemandTranche:
    """A demand tranche with quantity and willingness to pay."""
    quantity: float  # kWh or kg per period
    price: float     # ct/unit (willingness to pay)


@dataclass
class SimulationConfig:
    """Configuration for the energy system simulation."""
    # Time parameters
    periods: int = 24  # Number of time periods (e.g., hours)
    freq: str = 'h'    # Frequency (hourly)

    # Alpha: Electricity production
    alpha_capacity: float = 100.0      # Max capacity (kW)
    alpha_cost: float = 8.0            # Production cost (ct/kWh)

    # Bravo: Electrolyzer parameters
    bravo_capacity: float = 50.0       # Max output power (kW hydrogen equivalent)
    bravo_efficiency: float = 0.7      # Conversion efficiency (70%)
    bravo_cost: float = 1.5            # Transformation cost (ct/kWh input)

    # Grid backup (expensive fallback)
    grid_cost: float = 15.0            # Grid electricity cost (ct/kWh)

    # Charlie: Hydrogen consumer - demand tranches (diminishing marginal utility)
    # Each tuple: (quantity per period, willingness to pay in ct/kg)
    charlie_tranches: List[Tuple[float, float]] = field(default_factory=lambda: [
        (5.0, 25.0),   # First 5 kg/period: high value, willing to pay 25 ct/kg
        (5.0, 18.0),   # Next 5 kg/period: medium value, 18 ct/kg
        (5.0, 12.0),   # Next 5 kg/period: lower value, 12 ct/kg
        (10.0, 6.0),   # Beyond that: low priority, 6 ct/kg
    ])

    # Delta: Electricity consumer - demand tranches (diminishing marginal utility)
    # Each tuple: (quantity per period, willingness to pay in ct/kWh)
    delta_tranches: List[Tuple[float, float]] = field(default_factory=lambda: [
        (10.0, 20.0),  # First 10 kWh/period: essential use, 20 ct/kWh
        (10.0, 14.0),  # Next 10 kWh/period: important use, 14 ct/kWh
        (15.0, 10.0),  # Next 15 kWh/period: comfort use, 10 ct/kWh
        (20.0, 5.0),   # Beyond that: discretionary, 5 ct/kWh
    ])


class HydrogenSystemSimulation:
    """
    Simulation of the hydrogen/electricity system using oemof.

    Economic model:
    - Alpha produces electricity at a cost
    - Bravo converts electricity to hydrogen (with efficiency and cost)
    - Charlie and Delta consume with diminishing marginal utility
    - Optimizer maximizes total welfare (consumer utility - costs)

    System layout:
        Alpha (Source) --> Electricity Bus --> Bravo (Converter) --> Hydrogen Bus --> Charlie (Sink)
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
        """Build the oemof energy system model with price-based economics."""
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

        # Alpha: Electricity producer with production cost
        # Variable production based on availability (e.g., solar pattern)
        alpha_output_profile = self._generate_production_profile()

        alpha = solph.components.Source(
            label='Alpha',
            outputs={
                bus_electricity: solph.Flow(
                    nominal_value=self.config.alpha_capacity,
                    max=alpha_output_profile,  # max availability per period
                    variable_costs=self.config.alpha_cost,  # 8 ct/kWh production cost
                )
            }
        )

        # Grid: Backup electricity source (more expensive)
        grid = solph.components.Source(
            label='Grid',
            outputs={
                bus_electricity: solph.Flow(
                    variable_costs=self.config.grid_cost,  # 15 ct/kWh
                )
            }
        )

        # Bravo: Electrolyzer (converts electricity to hydrogen)
        # Cost is on the input side (electricity consumed)
        bravo = solph.components.Converter(
            label='Bravo',
            inputs={
                bus_electricity: solph.Flow(
                    variable_costs=self.config.bravo_cost,  # 1.5 ct/kWh transformation cost
                )
            },
            outputs={
                bus_hydrogen: solph.Flow(
                    nominal_value=self.config.bravo_capacity
                )
            },
            conversion_factors={
                bus_hydrogen: self.config.bravo_efficiency  # 70% efficiency
            }
        )

        # Charlie: Hydrogen consumer with diminishing marginal utility
        # Model as multiple sinks (tranches) with different utility values
        # Negative cost = positive utility (benefit from consumption)
        charlie_sinks = []
        for i, (qty, wtp) in enumerate(self.config.charlie_tranches):
            sink = solph.components.Sink(
                label=f'Charlie_T{i+1}',
                inputs={
                    bus_hydrogen: solph.Flow(
                        nominal_value=qty,
                        variable_costs=-wtp,  # Negative = utility/benefit
                    )
                }
            )
            charlie_sinks.append(sink)

        # Delta: Electricity consumer with diminishing marginal utility
        delta_sinks = []
        for i, (qty, wtp) in enumerate(self.config.delta_tranches):
            sink = solph.components.Sink(
                label=f'Delta_T{i+1}',
                inputs={
                    bus_electricity: solph.Flow(
                        nominal_value=qty,
                        variable_costs=-wtp,  # Negative = utility/benefit
                    )
                }
            )
            delta_sinks.append(sink)

        # Excess sinks (curtailment) - no cost, just allows excess
        excess_electricity = solph.components.Sink(
            label='Excess_Electricity',
            inputs={bus_electricity: solph.Flow()}
        )

        excess_hydrogen = solph.components.Sink(
            label='Excess_Hydrogen',
            inputs={bus_hydrogen: solph.Flow()}
        )

        # Store components for reference
        self._components = {
            'bus_electricity': bus_electricity,
            'bus_hydrogen': bus_hydrogen,
            'Alpha': alpha,
            'Grid': grid,
            'Bravo': bravo,
            'Charlie_sinks': charlie_sinks,
            'Delta_sinks': delta_sinks,
            'Excess_Electricity': excess_electricity,
            'Excess_Hydrogen': excess_hydrogen,
        }

        # Add all components to energy system
        components_to_add = [
            bus_electricity, bus_hydrogen,
            alpha, grid, bravo,
            excess_electricity, excess_hydrogen
        ]
        components_to_add.extend(charlie_sinks)
        components_to_add.extend(delta_sinks)

        self.energy_system.add(*components_to_add)

        print("Energy system built successfully (price-based model).")

    def _generate_production_profile(self) -> list:
        """Generate electricity production availability profile (e.g., solar pattern)."""
        import math
        profile = []
        for t in range(self.config.periods):
            # Peak at midday (t=12), zero at night
            value = max(0, math.sin(math.pi * t / self.config.periods))
            profile.append(value)
        return profile

    def solve(self) -> bool:
        """Solve the optimization model (welfare maximization)."""
        if self.energy_system is None:
            raise RuntimeError("Energy system not built. Call build_system() first.")

        # Create model
        self.model = solph.Model(self.energy_system)

        # Solve using APPSI HiGHS interface directly
        print("Solving optimization model (maximizing welfare)...")

        try:
            # Use APPSI interface directly - bypasses SolverFactory issues
            from pyomo.contrib.appsi.solvers import Highs
            opt = Highs()
            results = opt.solve(self.model)

            if results.termination_condition.name not in ('optimal', 'feasible'):
                raise RuntimeError(f"Solver did not find optimal solution: {results.termination_condition}")

        except ImportError:
            # Fallback to SolverFactory
            import pyomo.environ  # Ensures solvers are registered
            from pyomo.opt import SolverFactory
            opt = SolverFactory('highs')
            if not opt.available():
                raise RuntimeError("No working solver found. Install HiGHS: pip install highspy")
            results = opt.solve(self.model, tee=False)

        # Get results
        self.results = solph.processing.results(self.model)

        print("Model solved successfully.")
        return True

    def get_flow_results(self) -> Dict[str, pd.DataFrame]:
        """Extract flow results from the solved model."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        flows = {}

        # Alpha production
        alpha_flow = views.node(self.results, 'Alpha')
        if alpha_flow is not None:
            flows['Alpha_production'] = alpha_flow['sequences']

        # Grid usage
        grid_flow = views.node(self.results, 'Grid')
        if grid_flow is not None:
            flows['Grid_supply'] = grid_flow['sequences']

        # Bravo conversion
        bravo_flow = views.node(self.results, 'Bravo')
        if bravo_flow is not None:
            flows['Bravo_conversion'] = bravo_flow['sequences']

        # Charlie consumption (sum all tranches)
        charlie_total = None
        for i in range(len(self.config.charlie_tranches)):
            sink_flow = views.node(self.results, f'Charlie_T{i+1}')
            if sink_flow is not None and 'sequences' in sink_flow:
                if charlie_total is None:
                    charlie_total = sink_flow['sequences'].copy()
                else:
                    charlie_total += sink_flow['sequences']
        if charlie_total is not None:
            flows['Charlie_consumption'] = charlie_total

        # Delta consumption (sum all tranches)
        delta_total = None
        for i in range(len(self.config.delta_tranches)):
            sink_flow = views.node(self.results, f'Delta_T{i+1}')
            if sink_flow is not None and 'sequences' in sink_flow:
                if delta_total is None:
                    delta_total = sink_flow['sequences'].copy()
                else:
                    delta_total += sink_flow['sequences']
        if delta_total is not None:
            flows['Delta_consumption'] = delta_total

        return flows

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of the simulation including economics."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        summary = {}

        try:
            # Production totals
            alpha_data = views.node(self.results, 'Alpha')
            if alpha_data is not None and 'sequences' in alpha_data:
                alpha_total = float(alpha_data['sequences'].sum().sum())
                summary['total_alpha_production'] = alpha_total
                summary['alpha_cost_total'] = alpha_total * self.config.alpha_cost

            grid_data = views.node(self.results, 'Grid')
            if grid_data is not None and 'sequences' in grid_data:
                grid_total = float(grid_data['sequences'].sum().sum())
                summary['total_grid_supply'] = grid_total
                summary['grid_cost_total'] = grid_total * self.config.grid_cost

            # Bravo conversion
            bravo_data = views.node(self.results, 'Bravo')
            if bravo_data is not None and 'sequences' in bravo_data:
                # Input electricity to Bravo
                bravo_input = float(bravo_data['sequences'].sum().sum()) / 2  # Approximate
                summary['bravo_conversion_cost'] = bravo_input * self.config.bravo_cost

            # Charlie consumption and utility
            charlie_utility = 0.0
            charlie_total = 0.0
            for i, (qty, wtp) in enumerate(self.config.charlie_tranches):
                sink_data = views.node(self.results, f'Charlie_T{i+1}')
                if sink_data is not None and 'sequences' in sink_data:
                    consumption = float(sink_data['sequences'].sum().sum())
                    charlie_total += consumption
                    charlie_utility += consumption * wtp
            summary['total_hydrogen_consumed'] = charlie_total
            summary['charlie_utility_total'] = charlie_utility

            # Delta consumption and utility
            delta_utility = 0.0
            delta_total = 0.0
            for i, (qty, wtp) in enumerate(self.config.delta_tranches):
                sink_data = views.node(self.results, f'Delta_T{i+1}')
                if sink_data is not None and 'sequences' in sink_data:
                    consumption = float(sink_data['sequences'].sum().sum())
                    delta_total += consumption
                    delta_utility += consumption * wtp
            summary['total_electricity_consumed'] = delta_total
            summary['delta_utility_total'] = delta_utility

            # Total welfare = utility - costs
            total_costs = summary.get('alpha_cost_total', 0) + \
                         summary.get('grid_cost_total', 0) + \
                         summary.get('bravo_conversion_cost', 0)
            total_utility = summary.get('charlie_utility_total', 0) + \
                           summary.get('delta_utility_total', 0)
            summary['total_costs'] = total_costs
            summary['total_utility'] = total_utility
            summary['total_welfare'] = total_utility - total_costs

        except Exception as e:
            print(f"Warning: Could not extract all summary data: {e}")

        summary['electrolyzer_efficiency'] = self.config.bravo_efficiency
        summary['alpha_cost_per_kwh'] = self.config.alpha_cost
        summary['bravo_cost_per_kwh'] = self.config.bravo_cost

        return summary

    def get_price_analysis(self) -> Dict[str, Any]:
        """Get detailed price and marginal utility analysis."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        analysis = {
            'electricity': {
                'production_cost': self.config.alpha_cost,
                'grid_cost': self.config.grid_cost,
                'consumer_tranches': []
            },
            'hydrogen': {
                'marginal_cost': self._calculate_hydrogen_marginal_cost(),
                'consumer_tranches': []
            }
        }

        # Analyze which tranches were used
        for i, (qty, wtp) in enumerate(self.config.delta_tranches):
            sink_data = views.node(self.results, f'Delta_T{i+1}')
            if sink_data is not None and 'sequences' in sink_data:
                consumption = float(sink_data['sequences'].sum().sum())
                analysis['electricity']['consumer_tranches'].append({
                    'tranche': i + 1,
                    'max_quantity': qty * self.config.periods,
                    'consumed': consumption,
                    'willingness_to_pay': wtp,
                    'utilization': consumption / (qty * self.config.periods) * 100
                })

        for i, (qty, wtp) in enumerate(self.config.charlie_tranches):
            sink_data = views.node(self.results, f'Charlie_T{i+1}')
            if sink_data is not None and 'sequences' in sink_data:
                consumption = float(sink_data['sequences'].sum().sum())
                analysis['hydrogen']['consumer_tranches'].append({
                    'tranche': i + 1,
                    'max_quantity': qty * self.config.periods,
                    'consumed': consumption,
                    'willingness_to_pay': wtp,
                    'utilization': consumption / (qty * self.config.periods) * 100
                })

        return analysis

    def _calculate_hydrogen_marginal_cost(self) -> float:
        """Calculate the marginal cost of hydrogen production."""
        # Cost = (electricity cost / efficiency) + transformation cost / efficiency
        elec_cost = self.config.alpha_cost
        transform_cost = self.config.bravo_cost
        efficiency = self.config.bravo_efficiency

        # Total cost per kWh of hydrogen = electricity needed * elec_cost + transform_cost
        # 1 kWh hydrogen requires 1/efficiency kWh electricity
        return (elec_cost + transform_cost) / efficiency

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
                'alpha_cost': self.config.alpha_cost,
                'bravo_efficiency': self.config.bravo_efficiency,
                'bravo_cost': self.config.bravo_cost,
                'charlie_tranches': self.config.charlie_tranches,
                'delta_tranches': self.config.delta_tranches,
            },
            'summary': self.get_summary(),
            'price_analysis': self.get_price_analysis(),
            'flows': {}
        }

        for name, df in flows.items():
            export_data['flows'][name] = df.values.tolist()

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        print(f"Results exported to {filepath}")

    def print_results(self) -> None:
        """Print simulation results summary with economic analysis."""
        summary = self.get_summary()
        analysis = self.get_price_analysis()

        print("\n" + "=" * 60)
        print("  SIMULATION RESULTS (Price-Based Model)")
        print("=" * 60)

        print(f"\nSimulation periods: {self.config.periods}")

        print("\n--- COSTS ---")
        print(f"  Alpha production cost: {self.config.alpha_cost} ct/kWh")
        print(f"  Bravo transformation cost: {self.config.bravo_cost} ct/kWh")
        print(f"  Bravo efficiency: {self.config.bravo_efficiency * 100:.0f}%")
        print(f"  Hydrogen marginal cost: {analysis['hydrogen']['marginal_cost']:.2f} ct/kWh")

        print("\n--- PRODUCTION ---")
        print(f"  Alpha electricity produced: {summary.get('total_alpha_production', 0):.2f} kWh")
        print(f"  Grid electricity used: {summary.get('total_grid_supply', 0):.2f} kWh")

        print("\n--- CONSUMPTION ---")
        print(f"  Delta electricity consumed: {summary.get('total_electricity_consumed', 0):.2f} kWh")
        print(f"  Charlie hydrogen consumed: {summary.get('total_hydrogen_consumed', 0):.2f} kg")

        print("\n--- ECONOMIC SUMMARY ---")
        print(f"  Total production costs: {summary.get('total_costs', 0):.2f} ct")
        print(f"  Total consumer utility: {summary.get('total_utility', 0):.2f} ct")
        print(f"  Total welfare (utility - costs): {summary.get('total_welfare', 0):.2f} ct")

        print("\n--- DEMAND TRANCHE UTILIZATION ---")
        print("  Electricity (Delta):")
        for t in analysis['electricity']['consumer_tranches']:
            print(f"    Tranche {t['tranche']}: {t['consumed']:.1f}/{t['max_quantity']:.1f} kWh "
                  f"({t['utilization']:.1f}%) @ {t['willingness_to_pay']} ct/kWh")

        print("  Hydrogen (Charlie):")
        for t in analysis['hydrogen']['consumer_tranches']:
            print(f"    Tranche {t['tranche']}: {t['consumed']:.1f}/{t['max_quantity']:.1f} kg "
                  f"({t['utilization']:.1f}%) @ {t['willingness_to_pay']} ct/kg")

        print("=" * 60)


def run_simulation(config: Optional[SimulationConfig] = None) -> HydrogenSystemSimulation:
    """Run a complete simulation with the given configuration."""
    sim = HydrogenSystemSimulation(config)
    sim.build_system()
    sim.solve()
    return sim


if __name__ == "__main__":
    # Run test simulation with price-based economics
    config = SimulationConfig(
        periods=24,
        alpha_capacity=100,
        alpha_cost=8.0,           # 8 ct/kWh production
        bravo_capacity=50,
        bravo_efficiency=0.7,     # 70% efficiency
        bravo_cost=1.5,           # 1.5 ct/kWh transformation
        grid_cost=15.0,           # 15 ct/kWh grid backup
    )

    sim = run_simulation(config)
    sim.print_results()
    sim.export_results_to_json("simulation_results.json")
