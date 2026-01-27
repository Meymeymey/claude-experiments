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


# =============================================================================
# Dynamic Node Configuration Classes
# =============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for an electricity generator node."""
    name: str
    capacity: float = 100.0       # Max capacity (kW)
    cost: float = 8.0             # Production cost (ct/kWh)
    profile_type: str = 'solar'   # 'solar', 'flat', 'wind'


@dataclass
class TransformerConfig:
    """Configuration for a transformer/converter node (e.g., electrolyzer)."""
    name: str
    capacity: float = 50.0        # Max output power
    efficiency: float = 0.7       # Conversion efficiency
    cost: float = 1.5             # Transformation cost (ct/kWh input)
    input_carrier: str = 'electricity'
    output_carrier: str = 'hydrogen'


@dataclass
class ConsumerConfig:
    """Configuration for a consumer node."""
    name: str
    consumer_type: str            # 'logarithmic_electricity', 'logarithmic_hydrogen', 'cobb_douglas'
    carrier: str                  # 'electricity', 'hydrogen', 'both'
    # Logarithmic utility params (for single-carrier consumers)
    log_scale: float = 30.0
    log_shape: float = 5.0
    log_max_quantity: float = 25.0
    log_num_tranches: int = 5
    # Cobb-Douglas params (for dual-carrier consumers)
    cd_A: float = 15.0
    cd_alpha: float = 0.4
    cd_beta: float = 0.6
    cd_max_bundles: float = 15.0
    cd_h_per_bundle: float = 1.0
    cd_e_per_bundle: float = 3.0
    cd_num_tranches: int = 5

    def get_logarithmic_utility(self) -> LogarithmicUtility:
        """Get LogarithmicUtility object from this config."""
        return LogarithmicUtility(
            scale=self.log_scale,
            shape=self.log_shape,
            max_quantity=self.log_max_quantity,
            num_tranches=self.log_num_tranches
        )

    def get_cobb_douglas_utility(self) -> CobbDouglasUtility:
        """Get CobbDouglasUtility object from this config."""
        return CobbDouglasUtility(
            A=self.cd_A,
            alpha=self.cd_alpha,
            beta=self.cd_beta,
            max_bundles=self.cd_max_bundles,
            h_per_bundle=self.cd_h_per_bundle,
            e_per_bundle=self.cd_e_per_bundle,
            num_tranches=self.cd_num_tranches
        )


@dataclass
class BatteryConfig:
    """Configuration for electricity storage (battery)."""
    name: str
    capacity: float = 100.0           # kWh storage capacity
    charge_rate: float = 50.0         # kW max charge rate
    discharge_rate: float = 50.0      # kW max discharge rate
    efficiency_in: float = 0.95       # Charging efficiency
    efficiency_out: float = 0.95      # Discharging efficiency
    loss_rate: float = 0.0002         # Hourly loss rate (0.02%)
    initial_level: float = 0.5        # Initial fill level (fraction)
    min_level: float = 0.1            # Minimum allowed level (fraction)
    cost: float = 1.0                 # Cost per kWh throughput (ct/kWh)


@dataclass
class H2StorageConfig:
    """Configuration for hydrogen storage (tank/cavern)."""
    name: str
    capacity: float = 50.0            # kg storage capacity
    injection_rate: float = 10.0      # kg/h max injection rate
    withdrawal_rate: float = 10.0     # kg/h max withdrawal rate
    efficiency_in: float = 0.95       # Compression/injection efficiency
    efficiency_out: float = 0.99      # Withdrawal efficiency
    loss_rate: float = 0.0001         # Hourly loss rate (0.01%)
    initial_level: float = 0.5        # Initial fill level (fraction)
    cost: float = 0.5                 # Cost per kg throughput (ct/kg)


@dataclass
class SimulationConfig:
    """Configuration for the energy system simulation."""
    # Time parameters
    periods: int = 24  # Number of time periods (e.g., hours)
    freq: str = 'h'    # Frequency (hourly)

    # Grid backup (expensive fallback)
    grid_cost: float = 15.0            # Grid electricity cost (ct/kWh)

    # Dynamic node lists (new approach)
    generators: List[GeneratorConfig] = field(default_factory=list)
    transformers: List[TransformerConfig] = field(default_factory=list)
    consumers: List[ConsumerConfig] = field(default_factory=list)
    batteries: List[BatteryConfig] = field(default_factory=list)
    h2_storage: List[H2StorageConfig] = field(default_factory=list)

    # Legacy fields for backwards compatibility
    alpha_capacity: float = 100.0      # Max capacity (kW)
    alpha_cost: float = 8.0            # Production cost (ct/kWh)
    bravo_capacity: float = 50.0       # Max output power (kW hydrogen equivalent)
    bravo_efficiency: float = 0.7      # Conversion efficiency (70%)
    bravo_cost: float = 1.5            # Transformation cost (ct/kWh input)
    charlie_utility: LogarithmicUtility = field(default_factory=lambda: LogarithmicUtility(
        scale=30.0, shape=5.0, max_quantity=25.0, num_tranches=5
    ))
    delta_utility: LogarithmicUtility = field(default_factory=lambda: LogarithmicUtility(
        scale=25.0, shape=10.0, max_quantity=50.0, num_tranches=5
    ))
    echo_utility: CobbDouglasUtility = field(default_factory=lambda: CobbDouglasUtility(
        A=15.0, alpha=0.4, beta=0.6, max_bundles=15.0,
        h_per_bundle=1.0, e_per_bundle=3.0, num_tranches=5
    ))
    include_echo: bool = True

    def __post_init__(self):
        """Convert legacy config to dynamic nodes if no dynamic nodes provided."""
        if not self.generators and not self.transformers and not self.consumers:
            self._convert_legacy_config()

    def _convert_legacy_config(self):
        """Convert legacy hardcoded config to dynamic node lists."""
        self.generators = [
            GeneratorConfig(name='Alpha', capacity=self.alpha_capacity, cost=self.alpha_cost, profile_type='solar')
        ]
        self.transformers = [
            TransformerConfig(name='Bravo', capacity=self.bravo_capacity,
                            efficiency=self.bravo_efficiency, cost=self.bravo_cost)
        ]
        self.consumers = [
            ConsumerConfig(
                name='Charlie', consumer_type='logarithmic_hydrogen', carrier='hydrogen',
                log_scale=self.charlie_utility.scale, log_shape=self.charlie_utility.shape,
                log_max_quantity=self.charlie_utility.max_quantity,
                log_num_tranches=self.charlie_utility.num_tranches
            ),
            ConsumerConfig(
                name='Delta', consumer_type='logarithmic_electricity', carrier='electricity',
                log_scale=self.delta_utility.scale, log_shape=self.delta_utility.shape,
                log_max_quantity=self.delta_utility.max_quantity,
                log_num_tranches=self.delta_utility.num_tranches
            ),
        ]
        if self.include_echo:
            self.consumers.append(ConsumerConfig(
                name='Echo', consumer_type='cobb_douglas', carrier='both',
                cd_A=self.echo_utility.A, cd_alpha=self.echo_utility.alpha,
                cd_beta=self.echo_utility.beta, cd_max_bundles=self.echo_utility.max_bundles,
                cd_h_per_bundle=self.echo_utility.h_per_bundle,
                cd_e_per_bundle=self.echo_utility.e_per_bundle,
                cd_num_tranches=self.echo_utility.num_tranches
            ))


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
        # Generated tranches per consumer (populated by build_system)
        self._consumer_tranches: Dict[str, List[Tuple[float, float]]] = {}
        # Legacy tranche references (for backwards compatibility)
        self._charlie_tranches = []
        self._delta_tranches = []
        self._echo_tranches = []

    def build_system(self) -> None:
        """Build the oemof energy system model with dynamic nodes."""
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

        components_to_add = [bus_electricity, bus_hydrogen]

        # Build generators dynamically
        for gen_config in self.config.generators:
            profile = self._generate_production_profile(gen_config.profile_type)
            generator = solph.components.Source(
                label=gen_config.name,
                outputs={
                    bus_electricity: solph.Flow(
                        nominal_value=gen_config.capacity,
                        max=profile,
                        variable_costs=gen_config.cost,
                    )
                }
            )
            components_to_add.append(generator)
            self._components[gen_config.name] = generator

        # Grid: Backup electricity source (more expensive)
        grid = solph.components.Source(
            label='Grid',
            outputs={
                bus_electricity: solph.Flow(
                    variable_costs=self.config.grid_cost,
                )
            }
        )
        components_to_add.append(grid)
        self._components['Grid'] = grid

        # Build transformers dynamically
        for trans_config in self.config.transformers:
            input_bus = bus_electricity if trans_config.input_carrier == 'electricity' else bus_hydrogen
            output_bus = bus_hydrogen if trans_config.output_carrier == 'hydrogen' else bus_electricity

            transformer = solph.components.Converter(
                label=trans_config.name,
                inputs={
                    input_bus: solph.Flow(
                        variable_costs=trans_config.cost,
                    )
                },
                outputs={
                    output_bus: solph.Flow(
                        nominal_value=trans_config.capacity
                    )
                },
                conversion_factors={
                    output_bus: trans_config.efficiency
                }
            )
            components_to_add.append(transformer)
            self._components[trans_config.name] = transformer

        # Build consumers dynamically
        satisfaction_buses = {}  # For Cobb-Douglas consumers

        for cons_config in self.config.consumers:
            if cons_config.consumer_type.startswith('logarithmic'):
                # Logarithmic utility consumer (single carrier)
                utility = cons_config.get_logarithmic_utility()
                bus = bus_hydrogen if cons_config.carrier == 'hydrogen' else bus_electricity
                tranches = utility.generate_tranches()

                self._consumer_tranches[cons_config.name] = tranches
                consumer_sinks = []

                for i, (qty, wtp) in enumerate(tranches):
                    sink = solph.components.Sink(
                        label=f'{cons_config.name}_T{i+1}',
                        inputs={
                            bus: solph.Flow(
                                nominal_value=qty,
                                variable_costs=-wtp,  # Negative = utility/benefit
                            )
                        }
                    )
                    consumer_sinks.append(sink)
                    components_to_add.append(sink)

                self._components[f'{cons_config.name}_sinks'] = consumer_sinks

                # Legacy compatibility
                if cons_config.name == 'Charlie':
                    self._charlie_tranches = tranches
                elif cons_config.name == 'Delta':
                    self._delta_tranches = tranches

            elif cons_config.consumer_type == 'cobb_douglas':
                # Cobb-Douglas utility consumer (dual carrier)
                utility = cons_config.get_cobb_douglas_utility()
                tranches = utility.generate_tranches()

                self._consumer_tranches[cons_config.name] = tranches

                # Create satisfaction bus for this consumer
                bus_satisfaction = solph.Bus(label=f'{cons_config.name}_satisfaction_bus')
                satisfaction_buses[cons_config.name] = bus_satisfaction
                components_to_add.append(bus_satisfaction)

                consumer_converters = []
                for i, (qty, wtp) in enumerate(tranches):
                    h_per_bundle = cons_config.cd_h_per_bundle
                    e_per_bundle = cons_config.cd_e_per_bundle

                    converter = solph.components.Converter(
                        label=f'{cons_config.name}_T{i+1}',
                        inputs={
                            bus_hydrogen: solph.Flow(),
                            bus_electricity: solph.Flow(),
                        },
                        outputs={
                            bus_satisfaction: solph.Flow(
                                nominal_value=qty,
                                variable_costs=-wtp,
                            )
                        },
                        conversion_factors={
                            bus_hydrogen: 1 / h_per_bundle,
                            bus_electricity: 1 / e_per_bundle,
                            bus_satisfaction: 1.0,
                        }
                    )
                    consumer_converters.append(converter)
                    components_to_add.append(converter)

                self._components[f'{cons_config.name}_sinks'] = consumer_converters

                # Satisfaction sink (to complete the bus)
                satisfaction_sink = solph.components.Sink(
                    label=f'{cons_config.name}_Satisfaction',
                    inputs={bus_satisfaction: solph.Flow()}
                )
                components_to_add.append(satisfaction_sink)

                # Legacy compatibility
                if cons_config.name == 'Echo':
                    self._echo_tranches = tranches

        # Build battery storage dynamically
        for batt_config in self.config.batteries:
            battery = solph.components.GenericStorage(
                label=batt_config.name,
                nominal_storage_capacity=batt_config.capacity,
                inputs={
                    bus_electricity: solph.Flow(
                        nominal_value=batt_config.charge_rate,
                        variable_costs=batt_config.cost,
                    )
                },
                outputs={
                    bus_electricity: solph.Flow(
                        nominal_value=batt_config.discharge_rate,
                    )
                },
                loss_rate=batt_config.loss_rate,
                initial_storage_level=batt_config.initial_level,
                min_storage_level=batt_config.min_level,
                max_storage_level=1.0,
                inflow_conversion_factor=batt_config.efficiency_in,
                outflow_conversion_factor=batt_config.efficiency_out,
                balanced=True,  # Final storage = initial storage
            )
            components_to_add.append(battery)
            self._components[batt_config.name] = battery

        # Build hydrogen storage dynamically
        for h2_config in self.config.h2_storage:
            h2_tank = solph.components.GenericStorage(
                label=h2_config.name,
                nominal_storage_capacity=h2_config.capacity,
                inputs={
                    bus_hydrogen: solph.Flow(
                        nominal_value=h2_config.injection_rate,
                        variable_costs=h2_config.cost,
                    )
                },
                outputs={
                    bus_hydrogen: solph.Flow(
                        nominal_value=h2_config.withdrawal_rate,
                    )
                },
                loss_rate=h2_config.loss_rate,
                initial_storage_level=h2_config.initial_level,
                min_storage_level=0.0,  # H2 tanks can be emptied
                max_storage_level=1.0,
                inflow_conversion_factor=h2_config.efficiency_in,
                outflow_conversion_factor=h2_config.efficiency_out,
                balanced=True,  # Final storage = initial storage
            )
            components_to_add.append(h2_tank)
            self._components[h2_config.name] = h2_tank

        # Excess sinks (curtailment) - no cost, just allows excess
        excess_electricity = solph.components.Sink(
            label='Excess_Electricity',
            inputs={bus_electricity: solph.Flow()}
        )
        excess_hydrogen = solph.components.Sink(
            label='Excess_Hydrogen',
            inputs={bus_hydrogen: solph.Flow()}
        )
        components_to_add.extend([excess_electricity, excess_hydrogen])

        # Store bus references
        self._components['bus_electricity'] = bus_electricity
        self._components['bus_hydrogen'] = bus_hydrogen
        self._components['Excess_Electricity'] = excess_electricity
        self._components['Excess_Hydrogen'] = excess_hydrogen

        self.energy_system.add(*components_to_add)

        print(f"Energy system built: {len(self.config.generators)} generators, "
              f"{len(self.config.transformers)} transformers, {len(self.config.consumers)} consumers, "
              f"{len(self.config.batteries)} batteries, {len(self.config.h2_storage)} H2 storage.")

    def _generate_production_profile(self, profile_type: str = 'solar') -> list:
        """Generate electricity production availability profile based on type."""
        import math
        profile = []

        for t in range(self.config.periods):
            if profile_type == 'solar':
                # Peak at midday (t=12), zero at night
                value = max(0, math.sin(math.pi * t / self.config.periods))
            elif profile_type == 'wind':
                # More variable pattern
                value = 0.3 + 0.5 * math.sin(2 * math.pi * t / self.config.periods) + \
                       0.2 * math.sin(4 * math.pi * t / self.config.periods)
                value = max(0.1, min(1.0, value))
            elif profile_type == 'flat':
                # Constant availability
                value = 1.0
            else:
                # Default to solar
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

        # Generator production (all generators)
        for gen_config in self.config.generators:
            gen_flow = views.node(self.results, gen_config.name)
            if gen_flow is not None:
                flows[f'{gen_config.name}_production'] = gen_flow['sequences']

        # Grid usage
        grid_flow = views.node(self.results, 'Grid')
        if grid_flow is not None:
            flows['Grid_supply'] = grid_flow['sequences']

        # Transformer conversion (all transformers)
        for trans_config in self.config.transformers:
            trans_flow = views.node(self.results, trans_config.name)
            if trans_flow is not None:
                flows[f'{trans_config.name}_conversion'] = trans_flow['sequences']

        # Consumer consumption (all consumers)
        for cons_config in self.config.consumers:
            tranches = self._consumer_tranches.get(cons_config.name, [])
            cons_total = None

            for i in range(len(tranches)):
                sink_flow = views.node(self.results, f'{cons_config.name}_T{i+1}')
                if sink_flow is not None and 'sequences' in sink_flow:
                    if cons_total is None:
                        cons_total = sink_flow['sequences'].copy()
                    else:
                        cons_total += sink_flow['sequences']

            if cons_total is not None:
                flows[f'{cons_config.name}_consumption'] = cons_total

                # For Cobb-Douglas consumers, also calculate H2 and electricity used
                if cons_config.consumer_type == 'cobb_douglas':
                    flows[f'{cons_config.name}_hydrogen'] = cons_total * cons_config.cd_h_per_bundle
                    flows[f'{cons_config.name}_electricity'] = cons_total * cons_config.cd_e_per_bundle

        # Battery storage state - extract storage content (state of charge)
        for batt_config in self.config.batteries:
            batt_data = views.node(self.results, batt_config.name)
            if batt_data is not None and 'sequences' in batt_data:
                batt_df = batt_data['sequences']
                # Find the storage content column
                storage_col = None
                for col in batt_df.columns:
                    if 'storage_content' in str(col) or 'capacity' in str(col):
                        storage_col = col
                        break
                if storage_col is not None:
                    flows[f'{batt_config.name}_storage'] = batt_df[storage_col].tolist()
                else:
                    # Fall back to full DataFrame if no storage_content found
                    flows[f'{batt_config.name}_storage'] = batt_df

        # H2 storage state - extract storage content
        for h2_config in self.config.h2_storage:
            h2_data = views.node(self.results, h2_config.name)
            if h2_data is not None and 'sequences' in h2_data:
                h2_df = h2_data['sequences']
                # Find the storage content column
                storage_col = None
                for col in h2_df.columns:
                    if 'storage_content' in str(col) or 'capacity' in str(col):
                        storage_col = col
                        break
                if storage_col is not None:
                    flows[f'{h2_config.name}_storage'] = h2_df[storage_col].tolist()
                else:
                    # Fall back to full DataFrame if no storage_content found
                    flows[f'{h2_config.name}_storage'] = h2_df

        return flows

    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics of the simulation including economics."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        summary = {}

        try:
            # Generator production totals
            total_production_cost = 0.0
            total_production = 0.0
            for gen_config in self.config.generators:
                gen_data = views.node(self.results, gen_config.name)
                if gen_data is not None and 'sequences' in gen_data:
                    gen_total = float(gen_data['sequences'].sum().sum())
                    summary[f'{gen_config.name}_production'] = gen_total
                    summary[f'{gen_config.name}_cost'] = gen_total * gen_config.cost
                    total_production += gen_total
                    total_production_cost += gen_total * gen_config.cost

            # Legacy compatibility
            if 'Alpha_production' in summary:
                summary['total_alpha_production'] = summary['Alpha_production']
                summary['alpha_cost_total'] = summary['Alpha_cost']

            summary['total_production'] = total_production

            # Grid supply
            grid_data = views.node(self.results, 'Grid')
            if grid_data is not None and 'sequences' in grid_data:
                grid_total = float(grid_data['sequences'].sum().sum())
                summary['total_grid_supply'] = grid_total
                summary['grid_cost_total'] = grid_total * self.config.grid_cost
                total_production_cost += grid_total * self.config.grid_cost

            # Transformer costs
            total_transformer_cost = 0.0
            for trans_config in self.config.transformers:
                trans_data = views.node(self.results, trans_config.name)
                if trans_data is not None and 'sequences' in trans_data:
                    trans_input = float(trans_data['sequences'].sum().sum()) / 2
                    trans_cost = trans_input * trans_config.cost
                    summary[f'{trans_config.name}_cost'] = trans_cost
                    total_transformer_cost += trans_cost

            # Legacy compatibility
            if self.config.transformers:
                summary['bravo_conversion_cost'] = total_transformer_cost

            # Consumer consumption and utility (all consumers)
            total_utility = 0.0
            total_h2_consumed = 0.0
            total_elec_consumed = 0.0

            for cons_config in self.config.consumers:
                tranches = self._consumer_tranches.get(cons_config.name, [])
                cons_utility = 0.0
                cons_total = 0.0

                for i, (qty, wtp) in enumerate(tranches):
                    sink_data = views.node(self.results, f'{cons_config.name}_T{i+1}')
                    if sink_data is not None and 'sequences' in sink_data:
                        consumption = float(sink_data['sequences'].sum().sum())
                        cons_total += consumption
                        cons_utility += consumption * wtp

                summary[f'{cons_config.name}_consumed'] = cons_total
                summary[f'{cons_config.name}_utility'] = cons_utility
                total_utility += cons_utility

                # Track carrier consumption
                if cons_config.carrier == 'hydrogen':
                    total_h2_consumed += cons_total
                    # Legacy compatibility
                    if cons_config.name == 'Charlie':
                        summary['charlie_hydrogen_consumed'] = cons_total
                        summary['charlie_utility_total'] = cons_utility
                elif cons_config.carrier == 'electricity':
                    total_elec_consumed += cons_total
                    # Legacy compatibility
                    if cons_config.name == 'Delta':
                        summary['delta_electricity_consumed'] = cons_total
                        summary['delta_utility_total'] = cons_utility
                elif cons_config.carrier == 'both':
                    # Cobb-Douglas: bundles consumed
                    h2_used = cons_total * cons_config.cd_h_per_bundle
                    elec_used = cons_total * cons_config.cd_e_per_bundle
                    total_h2_consumed += h2_used
                    total_elec_consumed += elec_used
                    summary[f'{cons_config.name}_hydrogen'] = h2_used
                    summary[f'{cons_config.name}_electricity'] = elec_used
                    # Legacy compatibility
                    if cons_config.name == 'Echo':
                        summary['echo_bundles_consumed'] = cons_total
                        summary['echo_hydrogen_consumed'] = h2_used
                        summary['echo_electricity_consumed'] = elec_used
                        summary['echo_utility_total'] = cons_utility

            summary['total_hydrogen_consumed'] = total_h2_consumed
            summary['total_electricity_consumed'] = total_elec_consumed

            # Storage statistics
            total_storage_cost = 0.0

            # Battery storage
            for batt_config in self.config.batteries:
                batt_data = views.node(self.results, batt_config.name)
                if batt_data is not None and 'sequences' in batt_data:
                    # Extract charge/discharge from sequences
                    batt_df = batt_data['sequences']
                    # Sum all flows (charge + discharge throughput)
                    batt_throughput = float(batt_df.sum().sum()) / 2  # Divide by 2 as flows are double-counted
                    batt_cost = batt_throughput * batt_config.cost
                    summary[f'{batt_config.name}_throughput'] = batt_throughput
                    summary[f'{batt_config.name}_cost'] = batt_cost
                    summary[f'{batt_config.name}_capacity'] = batt_config.capacity
                    total_storage_cost += batt_cost

            # H2 storage
            for h2_config in self.config.h2_storage:
                h2_data = views.node(self.results, h2_config.name)
                if h2_data is not None and 'sequences' in h2_data:
                    h2_df = h2_data['sequences']
                    h2_throughput = float(h2_df.sum().sum()) / 2
                    h2_cost = h2_throughput * h2_config.cost
                    summary[f'{h2_config.name}_throughput'] = h2_throughput
                    summary[f'{h2_config.name}_cost'] = h2_cost
                    summary[f'{h2_config.name}_capacity'] = h2_config.capacity
                    total_storage_cost += h2_cost

            summary['total_storage_cost'] = total_storage_cost

            # Total welfare = utility - costs
            total_costs = total_production_cost + total_transformer_cost + total_storage_cost
            summary['total_costs'] = total_costs
            summary['total_utility'] = total_utility
            summary['total_welfare'] = total_utility - total_costs

        except Exception as e:
            print(f"Warning: Could not extract all summary data: {e}")
            import traceback
            traceback.print_exc()

        # Legacy compatibility fields
        if self.config.transformers:
            summary['electrolyzer_efficiency'] = self.config.transformers[0].efficiency
            summary['bravo_cost_per_kwh'] = self.config.transformers[0].cost
        if self.config.generators:
            summary['alpha_cost_per_kwh'] = self.config.generators[0].cost

        return summary

    def get_price_analysis(self) -> Dict[str, Any]:
        """Get detailed price and marginal utility analysis."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        # Get production costs from first generator (for legacy compatibility)
        gen_cost = self.config.generators[0].cost if self.config.generators else 8.0

        analysis = {
            'electricity': {
                'production_cost': gen_cost,
                'grid_cost': self.config.grid_cost,
                'consumer_tranches': []
            },
            'hydrogen': {
                'marginal_cost': self._calculate_hydrogen_marginal_cost(),
                'consumer_tranches': []
            },
            'echo': {
                'bundles': [],
                'h_per_bundle': 0,
                'e_per_bundle': 0,
            },
            'consumers': {}  # New: per-consumer tranche data
        }

        # Process all consumers dynamically
        for cons_config in self.config.consumers:
            tranches = self._consumer_tranches.get(cons_config.name, [])
            consumer_data = {
                'consumer_type': cons_config.consumer_type,
                'carrier': cons_config.carrier,
                'tranches': []
            }

            for i, (qty, wtp) in enumerate(tranches):
                sink_data = views.node(self.results, f'{cons_config.name}_T{i+1}')
                if sink_data is not None and 'sequences' in sink_data:
                    consumption = float(sink_data['sequences'].sum().sum())
                    tranche_data = {
                        'tranche': i + 1,
                        'max_quantity': qty * self.config.periods,
                        'consumed': consumption,
                        'willingness_to_pay': wtp,
                        'utilization': consumption / (qty * self.config.periods) * 100 if qty > 0 else 0
                    }
                    consumer_data['tranches'].append(tranche_data)

                    # Also add to legacy structure
                    if cons_config.carrier == 'electricity':
                        analysis['electricity']['consumer_tranches'].append(tranche_data)
                    elif cons_config.carrier == 'hydrogen':
                        analysis['hydrogen']['consumer_tranches'].append(tranche_data)
                    elif cons_config.carrier == 'both':
                        # Cobb-Douglas bundle data
                        bundle_data = {
                            'tranche': i + 1,
                            'max_bundles': qty * self.config.periods,
                            'consumed': consumption,
                            'willingness_to_pay': wtp,
                            'utilization': consumption / (qty * self.config.periods) * 100 if qty > 0 else 0
                        }
                        analysis['echo']['bundles'].append(bundle_data)
                        analysis['echo']['h_per_bundle'] = cons_config.cd_h_per_bundle
                        analysis['echo']['e_per_bundle'] = cons_config.cd_e_per_bundle

            analysis['consumers'][cons_config.name] = consumer_data

        return analysis

    def _calculate_hydrogen_marginal_cost(self) -> float:
        """Calculate the marginal cost of hydrogen production."""
        # Get costs from first generator and transformer
        if self.config.generators:
            elec_cost = self.config.generators[0].cost
        else:
            elec_cost = 8.0

        if self.config.transformers:
            transform_cost = self.config.transformers[0].cost
            efficiency = self.config.transformers[0].efficiency
        else:
            transform_cost = 1.5
            efficiency = 0.7

        # Total cost per kWh of hydrogen = electricity needed * elec_cost + transform_cost
        return (elec_cost + transform_cost) / efficiency

    def get_detailed_log(self) -> List[Dict[str, Any]]:
        """Get detailed per-timestep log of all node activities."""
        if self.results is None:
            raise RuntimeError("No results available. Call solve() first.")

        log_entries = []

        for t in range(self.config.periods):
            entry = {
                'timestep': t,
                'hour': t,
                'generators': {},
                'transformers': {},
                'consumers': {},
                'storage': {},
                'buses': {},
                'excess': {}
            }

            # Generator data
            for gen_config in self.config.generators:
                gen_data = views.node(self.results, gen_config.name)
                if gen_data is not None and 'sequences' in gen_data:
                    gen_df = gen_data['sequences']
                    # Sum all outflows for this timestep
                    production = 0.0
                    for col in gen_df.columns:
                        if t < len(gen_df):
                            production += float(gen_df.iloc[t][col])
                    entry['generators'][gen_config.name] = {
                        'production_kW': round(production, 2),
                        'cost_ct': round(production * gen_config.cost, 2),
                        'profile_type': gen_config.profile_type,
                        'capacity_kW': gen_config.capacity
                    }

            # Grid backup
            grid_data = views.node(self.results, 'Grid')
            if grid_data is not None and 'sequences' in grid_data:
                grid_df = grid_data['sequences']
                grid_supply = 0.0
                for col in grid_df.columns:
                    if t < len(grid_df):
                        grid_supply += float(grid_df.iloc[t][col])
                entry['generators']['Grid'] = {
                    'supply_kW': round(grid_supply, 2),
                    'cost_ct': round(grid_supply * self.config.grid_cost, 2)
                }

            # Transformer data - extract from bus flows for reliability
            for trans_config in self.config.transformers:
                input_flow = 0.0
                output_flow = 0.0

                # Method 1: Try views.node on the transformer directly
                trans_data = views.node(self.results, trans_config.name)
                if trans_data is not None and 'sequences' in trans_data:
                    trans_df = trans_data['sequences']
                    for col in trans_df.columns:
                        if t < len(trans_df):
                            val = float(trans_df.iloc[t][col])
                            col_str = str(col)
                            # Check direction: (source, target) in column tuple
                            if isinstance(col, tuple) and len(col) >= 1:
                                flow_tuple = col[0] if isinstance(col[0], tuple) else col
                                if isinstance(flow_tuple, tuple) and len(flow_tuple) >= 2:
                                    source, target = flow_tuple[0], flow_tuple[1]
                                    if target == trans_config.name:  # Inflow
                                        input_flow += val
                                    elif source == trans_config.name:  # Outflow
                                        output_flow += val

                # Method 2: Fallback - query from electricity bus if no data found
                if input_flow == 0.0 and output_flow == 0.0:
                    elec_bus_data = views.node(self.results, 'bus_electricity')
                    if elec_bus_data is not None and 'sequences' in elec_bus_data:
                        elec_df = elec_bus_data['sequences']
                        for col in elec_df.columns:
                            if t < len(elec_df):
                                col_str = str(col)
                                if trans_config.name in col_str and 'flow' in col_str:
                                    val = float(elec_df.iloc[t][col])
                                    # Flow TO transformer = input
                                    if isinstance(col, tuple) and isinstance(col[0], tuple):
                                        if col[0][1] == trans_config.name:
                                            input_flow += val

                    h2_bus_data = views.node(self.results, 'bus_hydrogen')
                    if h2_bus_data is not None and 'sequences' in h2_bus_data:
                        h2_df = h2_bus_data['sequences']
                        for col in h2_df.columns:
                            if t < len(h2_df):
                                col_str = str(col)
                                if trans_config.name in col_str and 'flow' in col_str:
                                    val = float(h2_df.iloc[t][col])
                                    # Flow FROM transformer = output
                                    if isinstance(col, tuple) and isinstance(col[0], tuple):
                                        if col[0][0] == trans_config.name:
                                            output_flow += val

                entry['transformers'][trans_config.name] = {
                    'input_kW': round(input_flow, 2),
                    'output_kg': round(output_flow, 2),
                    'efficiency': trans_config.efficiency,
                    'cost_ct': round(input_flow * trans_config.cost, 2)
                }

            # Consumer data with tranche details
            for cons_config in self.config.consumers:
                tranches = self._consumer_tranches.get(cons_config.name, [])
                consumer_entry = {
                    'consumer_type': cons_config.consumer_type,
                    'carrier': cons_config.carrier,
                    'total_consumption': 0.0,
                    'total_utility': 0.0,
                    'tranches': []
                }

                for i, (qty, wtp) in enumerate(tranches):
                    sink_data = views.node(self.results, f'{cons_config.name}_T{i+1}')
                    if sink_data is not None and 'sequences' in sink_data:
                        sink_df = sink_data['sequences']
                        consumption = 0.0
                        for col in sink_df.columns:
                            if t < len(sink_df):
                                consumption += float(sink_df.iloc[t][col])
                        utility = consumption * wtp
                        consumer_entry['tranches'].append({
                            'tranche': i + 1,
                            'consumption': round(consumption, 3),
                            'max_qty': qty,
                            'wtp_ct': round(wtp, 2),
                            'utility_ct': round(utility, 2)
                        })
                        consumer_entry['total_consumption'] += consumption
                        consumer_entry['total_utility'] += utility

                consumer_entry['total_consumption'] = round(consumer_entry['total_consumption'], 3)
                consumer_entry['total_utility'] = round(consumer_entry['total_utility'], 2)

                # Add units based on carrier
                if cons_config.carrier == 'hydrogen':
                    consumer_entry['unit'] = 'kg'
                elif cons_config.carrier == 'electricity':
                    consumer_entry['unit'] = 'kWh'
                else:
                    consumer_entry['unit'] = 'bundles'

                entry['consumers'][cons_config.name] = consumer_entry

            # Battery storage
            for batt_config in self.config.batteries:
                batt_data = views.node(self.results, batt_config.name)
                if batt_data is not None and 'sequences' in batt_data:
                    batt_df = batt_data['sequences']
                    charge = 0.0
                    discharge = 0.0
                    storage_level = 0.0
                    for col in batt_df.columns:
                        if t < len(batt_df):
                            val = float(batt_df.iloc[t][col])
                            col_str = str(col)
                            if 'storage_content' in col_str or 'capacity' in col_str:
                                storage_level = val
                            elif 'bus_electricity' in col_str:
                                # Check direction from column tuple
                                if isinstance(col, tuple) and len(col) >= 2:
                                    if col[0] == batt_config.name:
                                        discharge += val
                                    else:
                                        charge += val
                    entry['storage'][batt_config.name] = {
                        'type': 'battery',
                        'charge_kW': round(charge, 2),
                        'discharge_kW': round(discharge, 2),
                        'level_kWh': round(storage_level, 2),
                        'capacity_kWh': batt_config.capacity,
                        'level_pct': round(storage_level / batt_config.capacity * 100, 1) if batt_config.capacity > 0 else 0
                    }

            # H2 storage
            for h2_config in self.config.h2_storage:
                h2_data = views.node(self.results, h2_config.name)
                if h2_data is not None and 'sequences' in h2_data:
                    h2_df = h2_data['sequences']
                    injection = 0.0
                    withdrawal = 0.0
                    storage_level = 0.0
                    for col in h2_df.columns:
                        if t < len(h2_df):
                            val = float(h2_df.iloc[t][col])
                            col_str = str(col)
                            if 'storage_content' in col_str or 'capacity' in col_str:
                                storage_level = val
                            elif 'bus_hydrogen' in col_str:
                                if isinstance(col, tuple) and len(col) >= 2:
                                    if col[0] == h2_config.name:
                                        withdrawal += val
                                    else:
                                        injection += val
                    entry['storage'][h2_config.name] = {
                        'type': 'h2_tank',
                        'injection_kg': round(injection, 2),
                        'withdrawal_kg': round(withdrawal, 2),
                        'level_kg': round(storage_level, 2),
                        'capacity_kg': h2_config.capacity,
                        'level_pct': round(storage_level / h2_config.capacity * 100, 1) if h2_config.capacity > 0 else 0
                    }

            # Excess/curtailment
            excess_elec_data = views.node(self.results, 'Excess_Electricity')
            if excess_elec_data is not None and 'sequences' in excess_elec_data:
                excess_df = excess_elec_data['sequences']
                excess = 0.0
                for col in excess_df.columns:
                    if t < len(excess_df):
                        excess += float(excess_df.iloc[t][col])
                entry['excess']['electricity_kW'] = round(excess, 2)

            excess_h2_data = views.node(self.results, 'Excess_Hydrogen')
            if excess_h2_data is not None and 'sequences' in excess_h2_data:
                excess_df = excess_h2_data['sequences']
                excess = 0.0
                for col in excess_df.columns:
                    if t < len(excess_df):
                        excess += float(excess_df.iloc[t][col])
                entry['excess']['hydrogen_kg'] = round(excess, 2)

            log_entries.append(entry)

        return log_entries

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
