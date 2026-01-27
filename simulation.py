"""
Energy system simulation module using oemof.solph.
Models the hydrogen/electricity flows between nodes with price-based economics.

Economics:
- Producers have production costs (ct/kWh)
- Transformers have conversion costs (ct/kWh)
- Consumers have utility functions with diminishing marginal utility:
  - Charlie & Delta: Logarithmic utility U(x) = a * ln(1 + x/b)
  - Echo: Cobb-Douglas utility U(h,e) = A * h^α * e^β
"""

import pandas as pd
import math
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
class LogarithmicUtility:
    """
    Logarithmic utility function: U(x) = scale * ln(1 + x/shape)

    Properties:
    - Marginal utility: MU(x) = scale / (shape + x)
    - Diminishing marginal utility as x increases
    - scale controls overall willingness to pay
    - shape controls how fast marginal utility decreases
    """
    scale: float = 30.0    # 'a' parameter - controls max willingness to pay
    shape: float = 5.0     # 'b' parameter - controls diminishing rate
    max_quantity: float = 25.0  # Maximum consumption per period
    num_tranches: int = 5  # Number of tranches for LP approximation

    def utility(self, x: float) -> float:
        """Calculate utility at consumption level x."""
        return self.scale * math.log(1 + x / self.shape)

    def marginal_utility(self, x: float) -> float:
        """Calculate marginal utility (willingness to pay) at consumption level x."""
        return self.scale / (self.shape + x)

    def generate_tranches(self) -> List[Tuple[float, float]]:
        """Generate piecewise linear tranches to approximate the utility function."""
        tranches = []
        tranche_size = self.max_quantity / self.num_tranches

        for i in range(self.num_tranches):
            # Use midpoint of tranche for marginal utility calculation
            midpoint = (i + 0.5) * tranche_size
            wtp = self.marginal_utility(midpoint)
            tranches.append((tranche_size, wtp))

        return tranches


@dataclass
class CobbDouglasUtility:
    """
    Cobb-Douglas utility function: U(h, e) = A * h^α * e^β

    For LP approximation, we assume Echo consumes hydrogen and electricity
    in a fixed ratio determined by α and β, then optimize over "bundles".

    Properties:
    - A: scale parameter (overall utility multiplier)
    - alpha: exponent for hydrogen (h)
    - beta: exponent for electricity (e)
    - The optimal ratio h:e = α:β at any price ratio
    """
    A: float = 10.0        # Scale parameter
    alpha: float = 0.4     # Hydrogen exponent (share of budget)
    beta: float = 0.6      # Electricity exponent (share of budget)
    max_bundles: float = 20.0  # Max number of "bundles" per period
    h_per_bundle: float = 1.0  # kg hydrogen per bundle
    e_per_bundle: float = 3.0  # kWh electricity per bundle
    num_tranches: int = 5  # Number of tranches for LP approximation

    def utility(self, h: float, e: float) -> float:
        """Calculate utility at consumption levels (h, e)."""
        if h <= 0 or e <= 0:
            return 0
        return self.A * (h ** self.alpha) * (e ** self.beta)

    def bundle_utility(self, n_bundles: float) -> float:
        """Calculate utility from n bundles of (h_per_bundle, e_per_bundle)."""
        h = n_bundles * self.h_per_bundle
        e = n_bundles * self.e_per_bundle
        return self.utility(h, e)

    def marginal_bundle_utility(self, n_bundles: float) -> float:
        """Calculate marginal utility of an additional bundle."""
        if n_bundles <= 0:
            n_bundles = 0.1  # Small value to avoid division issues
        # Derivative of U with respect to n_bundles
        h = n_bundles * self.h_per_bundle
        e = n_bundles * self.e_per_bundle
        # dU/dn = dU/dh * dh/dn + dU/de * de/dn
        dU_dh = self.A * self.alpha * (h ** (self.alpha - 1)) * (e ** self.beta)
        dU_de = self.A * self.beta * (h ** self.alpha) * (e ** (self.beta - 1))
        return dU_dh * self.h_per_bundle + dU_de * self.e_per_bundle

    def generate_tranches(self) -> List[Tuple[float, float]]:
        """Generate tranches for bundle consumption (each bundle uses h and e)."""
        tranches = []
        tranche_size = self.max_bundles / self.num_tranches

        for i in range(self.num_tranches):
            midpoint = (i + 0.5) * tranche_size
            wtp = self.marginal_bundle_utility(midpoint)
            tranches.append((tranche_size, wtp))

        return tranches


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

    # Charlie: Hydrogen consumer with logarithmic utility
    charlie_utility: LogarithmicUtility = field(default_factory=lambda: LogarithmicUtility(
        scale=30.0,       # High willingness to pay for first units
        shape=5.0,        # Moderate diminishing rate
        max_quantity=25.0,
        num_tranches=5
    ))

    # Delta: Electricity consumer with logarithmic utility
    delta_utility: LogarithmicUtility = field(default_factory=lambda: LogarithmicUtility(
        scale=25.0,       # Slightly lower max WTP than Charlie
        shape=10.0,       # Slower diminishing rate (electricity more essential)
        max_quantity=50.0,
        num_tranches=5
    ))

    # Echo: Combined consumer with Cobb-Douglas utility (uses both H2 and electricity)
    echo_utility: CobbDouglasUtility = field(default_factory=lambda: CobbDouglasUtility(
        A=15.0,           # Scale parameter
        alpha=0.4,        # Hydrogen share
        beta=0.6,         # Electricity share
        max_bundles=15.0,
        h_per_bundle=1.0, # 1 kg H2 per bundle
        e_per_bundle=3.0, # 3 kWh electricity per bundle
        num_tranches=5
    ))

    # Whether to include Echo in the simulation
    include_echo: bool = True


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
        # Generated tranches (populated by build_system)
        self._charlie_tranches = []
        self._delta_tranches = []
        self._echo_tranches = []

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

        # Charlie: Hydrogen consumer with logarithmic utility
        # Generate tranches from utility function
        charlie_tranches = self.config.charlie_utility.generate_tranches()
        charlie_sinks = []
        for i, (qty, wtp) in enumerate(charlie_tranches):
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

        # Delta: Electricity consumer with logarithmic utility
        delta_tranches = self.config.delta_utility.generate_tranches()
        delta_sinks = []
        for i, (qty, wtp) in enumerate(delta_tranches):
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

        # Echo: Combined consumer with Cobb-Douglas utility
        # Consumes fixed-ratio bundles of hydrogen + electricity
        echo_sinks = []
        echo_tranches = []
        if self.config.include_echo:
            echo_tranches = self.config.echo_utility.generate_tranches()
            for i, (qty, wtp) in enumerate(echo_tranches):
                # Each Echo tranche is a "bundle sink" that needs both H2 and electricity
                # We model this as a Converter that consumes both inputs
                # The "product" is abstract utility (no physical output needed)
                h_per_bundle = self.config.echo_utility.h_per_bundle
                e_per_bundle = self.config.echo_utility.e_per_bundle

                # Use a dummy bus for Echo's "satisfaction" output
                if i == 0:
                    bus_echo_satisfaction = solph.Bus(label='echo_satisfaction_bus')

                # Echo bundle converter: takes H2 and electricity, outputs "satisfaction"
                echo_converter = solph.components.Converter(
                    label=f'Echo_T{i+1}',
                    inputs={
                        bus_hydrogen: solph.Flow(),
                        bus_electricity: solph.Flow(),
                    },
                    outputs={
                        bus_echo_satisfaction: solph.Flow(
                            nominal_value=qty,
                            variable_costs=-wtp,  # Negative = utility/benefit
                        )
                    },
                    conversion_factors={
                        bus_hydrogen: 1 / h_per_bundle,      # H2 needed per bundle
                        bus_electricity: 1 / e_per_bundle,   # Electricity needed per bundle
                        bus_echo_satisfaction: 1.0,          # Output: 1 satisfaction unit per bundle
                    }
                )
                echo_sinks.append(echo_converter)

        # Excess sinks (curtailment) - no cost, just allows excess
        excess_electricity = solph.components.Sink(
            label='Excess_Electricity',
            inputs={bus_electricity: solph.Flow()}
        )

        excess_hydrogen = solph.components.Sink(
            label='Excess_Hydrogen',
            inputs={bus_hydrogen: solph.Flow()}
        )

        # Echo satisfaction sink (needed to complete the bus)
        echo_satisfaction_sink = None
        if self.config.include_echo:
            echo_satisfaction_sink = solph.components.Sink(
                label='Echo_Satisfaction',
                inputs={bus_echo_satisfaction: solph.Flow()}
            )

        # Store components and tranches for reference
        self._components = {
            'bus_electricity': bus_electricity,
            'bus_hydrogen': bus_hydrogen,
            'Alpha': alpha,
            'Grid': grid,
            'Bravo': bravo,
            'Charlie_sinks': charlie_sinks,
            'Delta_sinks': delta_sinks,
            'Echo_sinks': echo_sinks,
            'Excess_Electricity': excess_electricity,
            'Excess_Hydrogen': excess_hydrogen,
        }
        self._charlie_tranches = charlie_tranches
        self._delta_tranches = delta_tranches
        self._echo_tranches = echo_tranches

        # Add all components to energy system
        components_to_add = [
            bus_electricity, bus_hydrogen,
            alpha, grid, bravo,
            excess_electricity, excess_hydrogen
        ]
        components_to_add.extend(charlie_sinks)
        components_to_add.extend(delta_sinks)

        if self.config.include_echo:
            components_to_add.append(bus_echo_satisfaction)
            components_to_add.extend(echo_sinks)
            components_to_add.append(echo_satisfaction_sink)

        self.energy_system.add(*components_to_add)

        print("Energy system built successfully (logarithmic + Cobb-Douglas utilities).")

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
        for i in range(len(self._charlie_tranches)):
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
        for i in range(len(self._delta_tranches)):
            sink_flow = views.node(self.results, f'Delta_T{i+1}')
            if sink_flow is not None and 'sequences' in sink_flow:
                if delta_total is None:
                    delta_total = sink_flow['sequences'].copy()
                else:
                    delta_total += sink_flow['sequences']
        if delta_total is not None:
            flows['Delta_consumption'] = delta_total

        # Echo consumption (sum all tranches - bundles consumed)
        if self.config.include_echo and self._echo_tranches:
            echo_total = None
            echo_h2_total = None
            echo_elec_total = None
            for i in range(len(self._echo_tranches)):
                echo_flow = views.node(self.results, f'Echo_T{i+1}')
                if echo_flow is not None and 'sequences' in echo_flow:
                    if echo_total is None:
                        echo_total = echo_flow['sequences'].copy()
                    else:
                        echo_total += echo_flow['sequences']
            if echo_total is not None:
                flows['Echo_consumption'] = echo_total
                # Also calculate H2 and electricity used by Echo
                h_per_bundle = self.config.echo_utility.h_per_bundle
                e_per_bundle = self.config.echo_utility.e_per_bundle
                flows['Echo_hydrogen'] = echo_total * h_per_bundle
                flows['Echo_electricity'] = echo_total * e_per_bundle

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

            # Charlie consumption and utility (using generated tranches)
            charlie_utility = 0.0
            charlie_total = 0.0
            for i, (qty, wtp) in enumerate(self._charlie_tranches):
                sink_data = views.node(self.results, f'Charlie_T{i+1}')
                if sink_data is not None and 'sequences' in sink_data:
                    consumption = float(sink_data['sequences'].sum().sum())
                    charlie_total += consumption
                    charlie_utility += consumption * wtp
            summary['charlie_hydrogen_consumed'] = charlie_total
            summary['charlie_utility_total'] = charlie_utility

            # Delta consumption and utility (using generated tranches)
            delta_utility = 0.0
            delta_total = 0.0
            for i, (qty, wtp) in enumerate(self._delta_tranches):
                sink_data = views.node(self.results, f'Delta_T{i+1}')
                if sink_data is not None and 'sequences' in sink_data:
                    consumption = float(sink_data['sequences'].sum().sum())
                    delta_total += consumption
                    delta_utility += consumption * wtp
            summary['delta_electricity_consumed'] = delta_total
            summary['delta_utility_total'] = delta_utility

            # Echo consumption and utility (Cobb-Douglas)
            echo_utility = 0.0
            echo_bundles = 0.0
            echo_h2 = 0.0
            echo_elec = 0.0
            if self.config.include_echo and self._echo_tranches:
                for i, (qty, wtp) in enumerate(self._echo_tranches):
                    echo_data = views.node(self.results, f'Echo_T{i+1}')
                    if echo_data is not None and 'sequences' in echo_data:
                        bundles = float(echo_data['sequences'].sum().sum())
                        echo_bundles += bundles
                        echo_utility += bundles * wtp
                echo_h2 = echo_bundles * self.config.echo_utility.h_per_bundle
                echo_elec = echo_bundles * self.config.echo_utility.e_per_bundle
            summary['echo_bundles_consumed'] = echo_bundles
            summary['echo_hydrogen_consumed'] = echo_h2
            summary['echo_electricity_consumed'] = echo_elec
            summary['echo_utility_total'] = echo_utility

            # Total consumption
            summary['total_hydrogen_consumed'] = charlie_total + echo_h2
            summary['total_electricity_consumed'] = delta_total + echo_elec

            # Total welfare = utility - costs
            total_costs = summary.get('alpha_cost_total', 0) + \
                         summary.get('grid_cost_total', 0) + \
                         summary.get('bravo_conversion_cost', 0)
            total_utility = summary.get('charlie_utility_total', 0) + \
                           summary.get('delta_utility_total', 0) + \
                           summary.get('echo_utility_total', 0)
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
            },
            'echo': {
                'bundles': [],
                'h_per_bundle': self.config.echo_utility.h_per_bundle if self.config.include_echo else 0,
                'e_per_bundle': self.config.echo_utility.e_per_bundle if self.config.include_echo else 0,
            }
        }

        # Delta tranches (electricity - logarithmic utility)
        for i, (qty, wtp) in enumerate(self._delta_tranches):
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

        # Charlie tranches (hydrogen - logarithmic utility)
        for i, (qty, wtp) in enumerate(self._charlie_tranches):
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

        # Echo tranches (Cobb-Douglas bundles)
        if self.config.include_echo and self._echo_tranches:
            for i, (qty, wtp) in enumerate(self._echo_tranches):
                echo_data = views.node(self.results, f'Echo_T{i+1}')
                if echo_data is not None and 'sequences' in echo_data:
                    bundles = float(echo_data['sequences'].sum().sum())
                    analysis['echo']['bundles'].append({
                        'tranche': i + 1,
                        'max_bundles': qty * self.config.periods,
                        'consumed': bundles,
                        'willingness_to_pay': wtp,
                        'utilization': bundles / (qty * self.config.periods) * 100 if qty > 0 else 0
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
                'charlie_utility': {
                    'type': 'logarithmic',
                    'scale': self.config.charlie_utility.scale,
                    'shape': self.config.charlie_utility.shape,
                    'max_quantity': self.config.charlie_utility.max_quantity,
                },
                'delta_utility': {
                    'type': 'logarithmic',
                    'scale': self.config.delta_utility.scale,
                    'shape': self.config.delta_utility.shape,
                    'max_quantity': self.config.delta_utility.max_quantity,
                },
                'echo_utility': {
                    'type': 'cobb_douglas',
                    'A': self.config.echo_utility.A,
                    'alpha': self.config.echo_utility.alpha,
                    'beta': self.config.echo_utility.beta,
                    'h_per_bundle': self.config.echo_utility.h_per_bundle,
                    'e_per_bundle': self.config.echo_utility.e_per_bundle,
                } if self.config.include_echo else None,
                'include_echo': self.config.include_echo,
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

        print("\n" + "=" * 70)
        print("  SIMULATION RESULTS (Logarithmic + Cobb-Douglas Utilities)")
        print("=" * 70)

        print(f"\nSimulation periods: {self.config.periods}")

        print("\n--- UTILITY FUNCTIONS ---")
        cu = self.config.charlie_utility
        print(f"  Charlie (H2): U(x) = {cu.scale} * ln(1 + x/{cu.shape})")
        du = self.config.delta_utility
        print(f"  Delta (Elec): U(x) = {du.scale} * ln(1 + x/{du.shape})")
        if self.config.include_echo:
            eu = self.config.echo_utility
            print(f"  Echo (Both):  U(h,e) = {eu.A} * h^{eu.alpha} * e^{eu.beta}")
            print(f"                Bundle: {eu.h_per_bundle} kg H2 + {eu.e_per_bundle} kWh elec")

        print("\n--- COSTS ---")
        print(f"  Alpha production cost: {self.config.alpha_cost} ct/kWh")
        print(f"  Bravo transformation cost: {self.config.bravo_cost} ct/kWh")
        print(f"  Bravo efficiency: {self.config.bravo_efficiency * 100:.0f}%")
        print(f"  Hydrogen marginal cost: {analysis['hydrogen']['marginal_cost']:.2f} ct/kWh")

        print("\n--- PRODUCTION ---")
        print(f"  Alpha electricity produced: {summary.get('total_alpha_production', 0):.2f} kWh")
        print(f"  Grid electricity used: {summary.get('total_grid_supply', 0):.2f} kWh")

        print("\n--- CONSUMPTION ---")
        print(f"  Charlie hydrogen: {summary.get('charlie_hydrogen_consumed', 0):.2f} kg")
        print(f"  Delta electricity: {summary.get('delta_electricity_consumed', 0):.2f} kWh")
        if self.config.include_echo:
            print(f"  Echo bundles: {summary.get('echo_bundles_consumed', 0):.2f}")
            print(f"    -> H2: {summary.get('echo_hydrogen_consumed', 0):.2f} kg")
            print(f"    -> Elec: {summary.get('echo_electricity_consumed', 0):.2f} kWh")
        print(f"  TOTAL hydrogen: {summary.get('total_hydrogen_consumed', 0):.2f} kg")
        print(f"  TOTAL electricity: {summary.get('total_electricity_consumed', 0):.2f} kWh")

        print("\n--- ECONOMIC SUMMARY ---")
        print(f"  Total production costs: {summary.get('total_costs', 0):.2f} ct")
        print(f"  Charlie utility: {summary.get('charlie_utility_total', 0):.2f} ct")
        print(f"  Delta utility: {summary.get('delta_utility_total', 0):.2f} ct")
        if self.config.include_echo:
            print(f"  Echo utility: {summary.get('echo_utility_total', 0):.2f} ct")
        print(f"  Total consumer utility: {summary.get('total_utility', 0):.2f} ct")
        print(f"  Total welfare (utility - costs): {summary.get('total_welfare', 0):.2f} ct")

        print("\n--- DEMAND TRANCHE UTILIZATION ---")
        print("  Delta (Electricity - Logarithmic):")
        for t in analysis['electricity']['consumer_tranches']:
            print(f"    T{t['tranche']}: {t['consumed']:.1f}/{t['max_quantity']:.1f} kWh "
                  f"({t['utilization']:.1f}%) @ {t['willingness_to_pay']:.2f} ct/kWh")

        print("  Charlie (Hydrogen - Logarithmic):")
        for t in analysis['hydrogen']['consumer_tranches']:
            print(f"    T{t['tranche']}: {t['consumed']:.1f}/{t['max_quantity']:.1f} kg "
                  f"({t['utilization']:.1f}%) @ {t['willingness_to_pay']:.2f} ct/kg")

        if self.config.include_echo and analysis['echo']['bundles']:
            print("  Echo (Bundles - Cobb-Douglas):")
            for t in analysis['echo']['bundles']:
                print(f"    T{t['tranche']}: {t['consumed']:.1f}/{t['max_bundles']:.1f} bundles "
                      f"({t['utilization']:.1f}%) @ {t['willingness_to_pay']:.2f} ct/bundle")

        print("=" * 70)


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
