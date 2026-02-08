"""
Flask-based controller for the hydrogen simulation.
Provides a web interface to control Blender sync, run simulations, and adjust parameters.

Note: Node types and connections are now also available in the modular structure:
- nodes/: Generator, Transformer, Consumer, Battery, H2Storage
- connections/: Connection, ElectricityConnection, HydrogenConnection
- utilities/: LogarithmicUtility, CobbDouglasUtility
"""

import json
import os
import subprocess
import sys
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from network import create_hydrogen_network, export_network_to_json, print_network_info
import math

# Import from new modular structure
MODULAR_IMPORTS_AVAILABLE = False
try:
    from nodes import Generator, Transformer, Consumer, Battery, H2Storage
    from nodes import NodeType, ConsumerType
    from nodes.base import Carrier
    from connections import Connection, ElectricityConnection, HydrogenConnection, ConnectionFactory
    from utilities import LogarithmicUtility as LogUtil, CobbDouglasUtility as CDUtil
    MODULAR_IMPORTS_AVAILABLE = True
except ImportError:
    pass  # Will use legacy classes from simulation.py


def sanitize_nan(obj):
    """Recursively replace NaN and Inf values with None for JSON serialization."""
    if isinstance(obj, dict):
        return {k: sanitize_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_nan(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


# Optional: simulation requires oemof which needs HiGHS solver
SIMULATION_AVAILABLE = False
try:
    from simulation import (
        SimulationConfig, HydrogenSystemSimulation,
        LogarithmicUtility, CobbDouglasUtility,
        GeneratorConfig, TransformerConfig, ConsumerConfig,
        BatteryConfig, H2StorageConfig
    )
    SIMULATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Simulation not available ({e})")
    print("Install with: pip install oemof.solph pandas highspy")

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# Store current configuration (utility-function-based model)
current_config = {
    'periods': 24,
    # Alpha: Electricity producer
    'alpha_capacity': 100.0,      # Max capacity (kW)
    'alpha_cost': 8.0,            # Production cost (ct/kWh)
    # Bravo: Electrolyzer
    'bravo_capacity': 50.0,       # Max output (kW hydrogen)
    'bravo_efficiency': 0.7,      # 70% efficiency
    'bravo_cost': 1.5,            # Transformation cost (ct/kWh)
    # Grid backup
    'grid_cost': 15.0,            # Grid electricity cost (ct/kWh)

    # Charlie: Hydrogen consumer - Logarithmic utility U(x) = scale * ln(1 + x/shape)
    'charlie_scale': 30.0,        # Scale parameter (max willingness to pay)
    'charlie_shape': 5.0,         # Shape parameter (diminishing rate)
    'charlie_max_quantity': 25.0, # Max consumption per period (kg)
    'charlie_num_tranches': 5,    # Number of LP approximation tranches

    # Delta: Electricity consumer - Logarithmic utility U(x) = scale * ln(1 + x/shape)
    'delta_scale': 25.0,          # Scale parameter
    'delta_shape': 10.0,          # Shape parameter (slower diminishing)
    'delta_max_quantity': 50.0,   # Max consumption per period (kWh)
    'delta_num_tranches': 5,      # Number of LP approximation tranches

    # Echo: Combined consumer - Cobb-Douglas utility U(h,e) = A * h^alpha * e^beta
    'include_echo': True,
    'echo_A': 15.0,               # Scale parameter
    'echo_alpha': 0.4,            # Hydrogen exponent
    'echo_beta': 0.6,             # Electricity exponent
    'echo_max_bundles': 15.0,     # Max bundles per period
    'echo_h_per_bundle': 1.0,     # kg H2 per bundle
    'echo_e_per_bundle': 3.0,     # kWh electricity per bundle
    'echo_num_tranches': 5,       # Number of LP approximation tranches
}

# Store last simulation results
last_results = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SCRIPT_DIR, 'network_data.json')
BLEND_PATH = os.path.join(SCRIPT_DIR, 'hydrogen_network.blend')
NODE_REGISTRY_PATH = os.path.join(SCRIPT_DIR, 'node_registry.json')

# =============================================================================
# Dynamic Node Registry
# =============================================================================

# Default node registry (backwards compatible with original nodes)
DEFAULT_NODE_REGISTRY = {
    'generators': [
        {'name': 'Alpha', 'capacity': 100.0, 'cost': 8.0, 'profile_type': 'solar', 'position': [0, 2, 0]}
    ],
    'transformers': [
        {'name': 'Bravo', 'capacity': 50.0, 'efficiency': 0.7, 'cost': 1.5,
         'input_carrier': 'electricity', 'output_carrier': 'hydrogen', 'position': [-2, 0, 0]}
    ],
    'consumers': [
        {'name': 'Charlie', 'consumer_type': 'logarithmic_hydrogen', 'carrier': 'hydrogen',
         'log_scale': 200.0, 'log_shape': 5.0, 'log_max_quantity': 25.0, 'log_num_tranches': 5,
         'position': [-2, -2, 0]},
        {'name': 'Delta', 'consumer_type': 'logarithmic_electricity', 'carrier': 'electricity',
         'log_scale': 150.0, 'log_shape': 10.0, 'log_max_quantity': 50.0, 'log_num_tranches': 5,
         'position': [2, 0, 0]},
        {'name': 'Echo', 'consumer_type': 'cobb_douglas', 'carrier': 'both',
         'cd_A': 100.0, 'cd_alpha': 0.4, 'cd_beta': 0.6, 'cd_max_bundles': 15.0,
         'cd_h_per_bundle': 1.0, 'cd_e_per_bundle': 3.0, 'cd_num_tranches': 5,
         'position': [0, -2, 0]}
    ],
    'batteries': [],
    'h2_storage': [],
    'connections': []  # Explicit connections list - if empty, auto-generate
}

# Active node registry
node_registry = None


def load_node_registry():
    """Load node registry from JSON file or create default."""
    global node_registry
    if os.path.exists(NODE_REGISTRY_PATH):
        try:
            with open(NODE_REGISTRY_PATH, 'r') as f:
                node_registry = json.load(f)
        except Exception as e:
            print(f"Error loading node registry: {e}")
            node_registry = json.loads(json.dumps(DEFAULT_NODE_REGISTRY))
    else:
        node_registry = json.loads(json.dumps(DEFAULT_NODE_REGISTRY))
        save_node_registry()

    # Ensure connections list exists
    ensure_connections_exist()
    # Clean up any stale connections referencing non-existent nodes
    if clean_stale_connections():
        save_node_registry()  # Persist the cleanup


def save_node_registry():
    """Save node registry to JSON file."""
    with open(NODE_REGISTRY_PATH, 'w') as f:
        json.dump(node_registry, f, indent=2)


def generate_default_connections():
    """Generate default connections based on node types and carriers."""
    connections = []

    # Generators -> Transformers (electricity)
    for gen in node_registry['generators']:
        for trans in node_registry['transformers']:
            if trans.get('input_carrier', 'electricity') == 'electricity':
                connections.append({
                    'source': gen['name'],
                    'target': trans['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })

    # Generators -> Electricity Consumers
    for gen in node_registry['generators']:
        for cons in node_registry['consumers']:
            if cons['carrier'] in ['electricity', 'both']:
                connections.append({
                    'source': gen['name'],
                    'target': cons['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })

    # Transformers -> Hydrogen Consumers
    for trans in node_registry['transformers']:
        for cons in node_registry['consumers']:
            if cons['carrier'] in ['hydrogen', 'both']:
                connections.append({
                    'source': trans['name'],
                    'target': cons['name'],
                    'carrier': 'hydrogen',
                    'enabled': True
                })

    # Generators -> Batteries (charging)
    for gen in node_registry['generators']:
        for batt in node_registry.get('batteries', []):
            connections.append({
                'source': gen['name'],
                'target': batt['name'],
                'carrier': 'electricity',
                'enabled': True
            })

    # Batteries -> Transformers (discharging to electrolyzers)
    for batt in node_registry.get('batteries', []):
        for trans in node_registry['transformers']:
            if trans.get('input_carrier', 'electricity') == 'electricity':
                connections.append({
                    'source': batt['name'],
                    'target': trans['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })

    # Batteries -> Electricity Consumers (discharging)
    for batt in node_registry.get('batteries', []):
        for cons in node_registry['consumers']:
            if cons['carrier'] in ['electricity', 'both']:
                connections.append({
                    'source': batt['name'],
                    'target': cons['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })

    # Transformers -> H2 Storage (injection)
    for trans in node_registry['transformers']:
        for h2s in node_registry.get('h2_storage', []):
            connections.append({
                'source': trans['name'],
                'target': h2s['name'],
                'carrier': 'hydrogen',
                'enabled': True
            })

    # H2 Storage -> Hydrogen Consumers (withdrawal)
    for h2s in node_registry.get('h2_storage', []):
        for cons in node_registry['consumers']:
            if cons['carrier'] in ['hydrogen', 'both']:
                connections.append({
                    'source': h2s['name'],
                    'target': cons['name'],
                    'carrier': 'hydrogen',
                    'enabled': True
                })

    return connections


def ensure_connections_exist():
    """Ensure connections list exists and populate if empty."""
    if 'connections' not in node_registry:
        node_registry['connections'] = []
    if not node_registry['connections']:
        node_registry['connections'] = generate_default_connections()
        save_node_registry()


def get_all_node_names():
    """Get set of all node names currently in registry."""
    names = set()
    for gen in node_registry.get('generators', []):
        names.add(gen['name'])
    for trans in node_registry.get('transformers', []):
        names.add(trans['name'])
    for cons in node_registry.get('consumers', []):
        names.add(cons['name'])
    for batt in node_registry.get('batteries', []):
        names.add(batt['name'])
    for h2s in node_registry.get('h2_storage', []):
        names.add(h2s['name'])
    return names


def clean_stale_connections():
    """Remove connections that reference non-existent nodes. Returns True if any were removed."""
    valid_names = get_all_node_names()
    connections = node_registry.get('connections', [])
    original_count = len(connections)
    # Filter to keep only connections where both source and target exist
    node_registry['connections'] = [
        conn for conn in connections
        if conn['source'] in valid_names and conn['target'] in valid_names
    ]
    removed_count = original_count - len(node_registry['connections'])
    if removed_count > 0:
        print(f"Cleaned up {removed_count} stale connections")
    return removed_count > 0


def add_connections_for_new_node(node_name, node_type, carrier=None):
    """Add default connections for a newly added node."""
    connections = node_registry.get('connections', [])

    if node_type == 'generator':
        # Generator connects to all transformers and electricity/both consumers
        for trans in node_registry.get('transformers', []):
            if trans.get('input_carrier', 'electricity') == 'electricity':
                connections.append({
                    'source': node_name,
                    'target': trans['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })
        for cons in node_registry.get('consumers', []):
            if cons['carrier'] in ['electricity', 'both']:
                connections.append({
                    'source': node_name,
                    'target': cons['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })
        for batt in node_registry.get('batteries', []):
            connections.append({
                'source': node_name,
                'target': batt['name'],
                'carrier': 'electricity',
                'enabled': True
            })

    elif node_type == 'transformer':
        # Transformer receives from all generators
        for gen in node_registry.get('generators', []):
            connections.append({
                'source': gen['name'],
                'target': node_name,
                'carrier': 'electricity',
                'enabled': True
            })
        # Transformer also receives from batteries
        for batt in node_registry.get('batteries', []):
            connections.append({
                'source': batt['name'],
                'target': node_name,
                'carrier': 'electricity',
                'enabled': True
            })
        # Transformer outputs to hydrogen/both consumers
        for cons in node_registry.get('consumers', []):
            if cons['carrier'] in ['hydrogen', 'both']:
                connections.append({
                    'source': node_name,
                    'target': cons['name'],
                    'carrier': 'hydrogen',
                    'enabled': True
                })
        # Transformer outputs to H2 storage
        for h2s in node_registry.get('h2_storage', []):
            connections.append({
                'source': node_name,
                'target': h2s['name'],
                'carrier': 'hydrogen',
                'enabled': True
            })

    elif node_type == 'consumer':
        # Consumer receives from generators (if elec/both) and transformers (if h2/both)
        if carrier in ['electricity', 'both']:
            for gen in node_registry.get('generators', []):
                connections.append({
                    'source': gen['name'],
                    'target': node_name,
                    'carrier': 'electricity',
                    'enabled': True
                })
            # Also from batteries
            for batt in node_registry.get('batteries', []):
                connections.append({
                    'source': batt['name'],
                    'target': node_name,
                    'carrier': 'electricity',
                    'enabled': True
                })
        if carrier in ['hydrogen', 'both']:
            for trans in node_registry.get('transformers', []):
                connections.append({
                    'source': trans['name'],
                    'target': node_name,
                    'carrier': 'hydrogen',
                    'enabled': True
                })
            # Also from H2 storage
            for h2s in node_registry.get('h2_storage', []):
                connections.append({
                    'source': h2s['name'],
                    'target': node_name,
                    'carrier': 'hydrogen',
                    'enabled': True
                })

    elif node_type == 'battery':
        # Battery receives from all generators (charging)
        for gen in node_registry.get('generators', []):
            connections.append({
                'source': gen['name'],
                'target': node_name,
                'carrier': 'electricity',
                'enabled': True
            })
        # Battery outputs to transformers (discharging)
        for trans in node_registry.get('transformers', []):
            if trans.get('input_carrier', 'electricity') == 'electricity':
                connections.append({
                    'source': node_name,
                    'target': trans['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })
        # Battery outputs to electricity consumers (discharging)
        for cons in node_registry.get('consumers', []):
            if cons['carrier'] in ['electricity', 'both']:
                connections.append({
                    'source': node_name,
                    'target': cons['name'],
                    'carrier': 'electricity',
                    'enabled': True
                })

    elif node_type == 'h2_storage':
        # H2 storage receives from all transformers (injection)
        for trans in node_registry.get('transformers', []):
            connections.append({
                'source': trans['name'],
                'target': node_name,
                'carrier': 'hydrogen',
                'enabled': True
            })
        # H2 storage outputs to hydrogen consumers (withdrawal)
        for cons in node_registry.get('consumers', []):
            if cons['carrier'] in ['hydrogen', 'both']:
                connections.append({
                    'source': node_name,
                    'target': cons['name'],
                    'carrier': 'hydrogen',
                    'enabled': True
                })

    node_registry['connections'] = connections


def sync_nodes_to_network():
    """Synchronize node registry to network topology JSON."""
    network_nodes = []
    network_edges = []

    # Add generators
    for gen in node_registry['generators']:
        network_nodes.append({
            'name': gen['name'],
            'node_type': 'producer',
            'carrier': 'electricity',
            'position': gen.get('position', [0, 0, 0])
        })

    # Add transformers
    for trans in node_registry['transformers']:
        network_nodes.append({
            'name': trans['name'],
            'node_type': 'converter',
            'carrier': 'both',
            'position': trans.get('position', [0, 0, 0])
        })

    # Add consumers
    for cons in node_registry['consumers']:
        network_nodes.append({
            'name': cons['name'],
            'node_type': 'consumer',
            'carrier': cons['carrier'],
            'position': cons.get('position', [0, 0, 0])
        })

    # Add batteries
    for batt in node_registry.get('batteries', []):
        network_nodes.append({
            'name': batt['name'],
            'node_type': 'storage',
            'carrier': 'electricity',
            'position': batt.get('position', [3, 0, 0])
        })

    # Add H2 storage
    for h2s in node_registry.get('h2_storage', []):
        network_nodes.append({
            'name': h2s['name'],
            'node_type': 'storage',
            'carrier': 'hydrogen',
            'position': h2s.get('position', [-3, -1, 0])
        })

    # Use explicit connections (only enabled ones and only for valid nodes) for edges
    valid_names = get_all_node_names()
    for conn in node_registry.get('connections', []):
        if conn.get('enabled', True):
            # Only add edge if both source and target nodes exist
            if conn['source'] in valid_names and conn['target'] in valid_names:
                network_edges.append({
                    'source': conn['source'],
                    'target': conn['target'],
                    'carrier': conn['carrier']
                })

    # Save to network JSON
    with open(JSON_PATH, 'w') as f:
        json.dump({'nodes': network_nodes, 'edges': network_edges}, f, indent=2)


# Load node registry at startup
load_node_registry()


@app.route('/')
def index():
    """Serve the controller HTML page."""
    return render_template('controller.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current simulation configuration."""
    return jsonify(current_config)


@app.route('/api/config', methods=['POST'])
def set_config():
    """Update simulation configuration."""
    global current_config
    data = request.json

    # Update only provided values
    for key in current_config:
        if key in data:
            # Handle boolean and numeric types appropriately
            if key == 'include_echo':
                current_config[key] = bool(data[key])
            elif key == 'periods':
                current_config[key] = int(data[key])
            elif key.endswith('_num_tranches'):
                current_config[key] = int(data[key])
            else:
                current_config[key] = float(data[key])

    return jsonify({'status': 'ok', 'config': current_config})


@app.route('/api/network', methods=['GET'])
def get_network():
    """Get current network topology."""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            return jsonify(json.load(f))
    else:
        # Generate default network
        G = create_hydrogen_network(use_json=False)
        export_network_to_json(G, JSON_PATH)
        with open(JSON_PATH, 'r') as f:
            return jsonify(json.load(f))


@app.route('/api/network', methods=['POST'])
def update_network():
    """Update network node positions."""
    data = request.json

    # Load existing or create new
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r') as f:
            network = json.load(f)
    else:
        network = {'nodes': [], 'edges': []}

    # Update node positions
    if 'nodes' in data:
        node_map = {n['name']: n for n in network.get('nodes', [])}
        for node in data['nodes']:
            if node['name'] in node_map:
                node_map[node['name']]['position'] = node['position']
            else:
                node_map[node['name']] = node
        network['nodes'] = list(node_map.values())

    # Update edges if provided
    if 'edges' in data:
        network['edges'] = data['edges']

    with open(JSON_PATH, 'w') as f:
        json.dump(network, f, indent=2)

    return jsonify({'status': 'ok', 'network': network})


@app.route('/api/simulate', methods=['POST'])
def run_simulation():
    """Run the energy simulation with dynamic node configuration."""
    global last_results

    if not SIMULATION_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'Simulation not available. Install oemof: pip install oemof.solph pandas highspy'
        }), 503

    try:
        # Build generator configs from node registry
        generators = [
            GeneratorConfig(
                name=g['name'],
                capacity=g['capacity'],
                cost=g['cost'],
                profile_type=g.get('profile_type', 'solar')
            )
            for g in node_registry['generators']
        ]

        # Build transformer configs from node registry
        transformers = [
            TransformerConfig(
                name=t['name'],
                capacity=t['capacity'],
                efficiency=t['efficiency'],
                cost=t['cost'],
                input_carrier=t.get('input_carrier', 'electricity'),
                output_carrier=t.get('output_carrier', 'hydrogen')
            )
            for t in node_registry['transformers']
        ]

        # Build consumer configs from node registry
        consumers = []
        for c in node_registry['consumers']:
            consumer = ConsumerConfig(
                name=c['name'],
                consumer_type=c['consumer_type'],
                carrier=c['carrier']
            )
            # Add type-specific parameters
            if c['consumer_type'].startswith('logarithmic'):
                consumer.log_scale = c.get('log_scale', 30.0)
                consumer.log_shape = c.get('log_shape', 5.0)
                consumer.log_max_quantity = c.get('log_max_quantity', 25.0)
                consumer.log_num_tranches = int(c.get('log_num_tranches', 5))
            elif c['consumer_type'] == 'cobb_douglas':
                consumer.cd_A = c.get('cd_A', 15.0)
                consumer.cd_alpha = c.get('cd_alpha', 0.4)
                consumer.cd_beta = c.get('cd_beta', 0.6)
                consumer.cd_max_bundles = c.get('cd_max_bundles', 15.0)
                consumer.cd_h_per_bundle = c.get('cd_h_per_bundle', 1.0)
                consumer.cd_e_per_bundle = c.get('cd_e_per_bundle', 3.0)
                consumer.cd_num_tranches = int(c.get('cd_num_tranches', 5))
            consumers.append(consumer)

        # Build battery configs from node registry
        batteries = [
            BatteryConfig(
                name=b['name'],
                capacity=b.get('capacity', 100.0),
                charge_rate=b.get('charge_rate', 50.0),
                discharge_rate=b.get('discharge_rate', 50.0),
                efficiency_in=b.get('efficiency_in', 0.95),
                efficiency_out=b.get('efficiency_out', 0.95),
                loss_rate=b.get('loss_rate', 0.0002),
                initial_level=b.get('initial_level', 0.5),
                min_level=b.get('min_level', 0.1),
                cost=b.get('cost', 1.0)
            )
            for b in node_registry.get('batteries', [])
        ]

        # Build H2 storage configs from node registry
        h2_storage = [
            H2StorageConfig(
                name=h['name'],
                capacity=h.get('capacity', 50.0),
                injection_rate=h.get('injection_rate', 10.0),
                withdrawal_rate=h.get('withdrawal_rate', 10.0),
                efficiency_in=h.get('efficiency_in', 0.95),
                efficiency_out=h.get('efficiency_out', 0.99),
                loss_rate=h.get('loss_rate', 0.0001),
                initial_level=h.get('initial_level', 0.5),
                cost=h.get('cost', 0.5)
            )
            for h in node_registry.get('h2_storage', [])
        ]

        config = SimulationConfig(
            periods=int(current_config['periods']),
            grid_cost=current_config['grid_cost'],
            generators=generators,
            transformers=transformers,
            consumers=consumers,
            batteries=batteries,
            h2_storage=h2_storage
        )

        sim = HydrogenSystemSimulation(config)
        sim.build_system()
        sim.solve()

        # Get results
        summary = sim.get_summary()
        flows = sim.get_flow_results()
        price_analysis = sim.get_price_analysis()

        # Convert flows to serializable format (filter out NaN values)
        import math
        flow_data = {}
        for name, data in flows.items():
            # Handle both DataFrame and list formats (storage data comes as lists)
            if hasattr(data, 'values'):
                values = data.values.tolist()
            else:
                values = data  # Already a list
            # Filter out rows containing NaN
            clean_values = []
            for row in values:
                if isinstance(row, list):
                    if not any(isinstance(v, float) and math.isnan(v) for v in row):
                        clean_values.append(row)
                else:
                    if not (isinstance(row, float) and math.isnan(row)):
                        clean_values.append(row)
            flow_data[name] = clean_values

        last_results = {
            'status': 'ok',
            'summary': summary,
            'flows': flow_data,
            'price_analysis': price_analysis,
            'config': current_config.copy(),
            'node_registry': node_registry
        }

        return jsonify(last_results)

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc(),
            'hint': 'Make sure HiGHS solver is installed: pip install highspy'
        }), 500


@app.route('/api/results', methods=['GET'])
def get_results():
    """Get last simulation results."""
    if last_results:
        return jsonify(last_results)
    else:
        return jsonify({'status': 'no_results', 'message': 'No simulation has been run yet'})


@app.route('/api/simulate/detailed', methods=['POST'])
def run_detailed_simulation():
    """Run simulation and return detailed per-timestep log of all node activities."""
    global last_results

    if not SIMULATION_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'Simulation not available. Install oemof: pip install oemof.solph pandas highspy'
        }), 503

    try:
        # Build generator configs from node registry
        generators = [
            GeneratorConfig(
                name=g['name'],
                capacity=g['capacity'],
                cost=g['cost'],
                profile_type=g.get('profile_type', 'solar')
            )
            for g in node_registry['generators']
        ]

        # Build transformer configs from node registry
        transformers = [
            TransformerConfig(
                name=t['name'],
                capacity=t['capacity'],
                efficiency=t['efficiency'],
                cost=t['cost'],
                input_carrier=t.get('input_carrier', 'electricity'),
                output_carrier=t.get('output_carrier', 'hydrogen')
            )
            for t in node_registry['transformers']
        ]

        # Build consumer configs from node registry
        consumers = []
        for c in node_registry['consumers']:
            consumer = ConsumerConfig(
                name=c['name'],
                consumer_type=c['consumer_type'],
                carrier=c['carrier']
            )
            if c['consumer_type'].startswith('logarithmic'):
                consumer.log_scale = c.get('log_scale', 30.0)
                consumer.log_shape = c.get('log_shape', 5.0)
                consumer.log_max_quantity = c.get('log_max_quantity', 25.0)
                consumer.log_num_tranches = int(c.get('log_num_tranches', 5))
            elif c['consumer_type'] == 'cobb_douglas':
                consumer.cd_A = c.get('cd_A', 15.0)
                consumer.cd_alpha = c.get('cd_alpha', 0.4)
                consumer.cd_beta = c.get('cd_beta', 0.6)
                consumer.cd_max_bundles = c.get('cd_max_bundles', 15.0)
                consumer.cd_h_per_bundle = c.get('cd_h_per_bundle', 1.0)
                consumer.cd_e_per_bundle = c.get('cd_e_per_bundle', 3.0)
                consumer.cd_num_tranches = int(c.get('cd_num_tranches', 5))
            consumers.append(consumer)

        # Build battery configs
        batteries = [
            BatteryConfig(
                name=b['name'],
                capacity=b.get('capacity', 100.0),
                charge_rate=b.get('charge_rate', 50.0),
                discharge_rate=b.get('discharge_rate', 50.0),
                efficiency_in=b.get('efficiency_in', 0.95),
                efficiency_out=b.get('efficiency_out', 0.95),
                loss_rate=b.get('loss_rate', 0.0002),
                initial_level=b.get('initial_level', 0.5),
                min_level=b.get('min_level', 0.1),
                cost=b.get('cost', 1.0)
            )
            for b in node_registry.get('batteries', [])
        ]

        # Build H2 storage configs
        h2_storage = [
            H2StorageConfig(
                name=h['name'],
                capacity=h.get('capacity', 50.0),
                injection_rate=h.get('injection_rate', 10.0),
                withdrawal_rate=h.get('withdrawal_rate', 10.0),
                efficiency_in=h.get('efficiency_in', 0.95),
                efficiency_out=h.get('efficiency_out', 0.99),
                loss_rate=h.get('loss_rate', 0.0001),
                initial_level=h.get('initial_level', 0.5),
                cost=h.get('cost', 0.5)
            )
            for h in node_registry.get('h2_storage', [])
        ]

        config = SimulationConfig(
            periods=int(current_config['periods']),
            grid_cost=current_config['grid_cost'],
            generators=generators,
            transformers=transformers,
            consumers=consumers,
            batteries=batteries,
            h2_storage=h2_storage
        )

        sim = HydrogenSystemSimulation(config)
        sim.build_system()
        sim.solve()

        # Get detailed log
        detailed_log = sim.get_detailed_log()
        summary = sim.get_summary()

        # Sanitize NaN values for JSON serialization
        detailed_log = sanitize_nan(detailed_log)
        summary = sanitize_nan(summary)

        return jsonify({
            'status': 'ok',
            'periods': config.periods,
            'summary': summary,
            'detailed_log': detailed_log,
            'node_registry': {
                'generators': [g['name'] for g in node_registry['generators']],
                'transformers': [t['name'] for t in node_registry['transformers']],
                'consumers': [c['name'] for c in node_registry['consumers']],
                'batteries': [b['name'] for b in node_registry.get('batteries', [])],
                'h2_storage': [h['name'] for h in node_registry.get('h2_storage', [])]
            }
        })

    except Exception as e:
        import traceback
        return jsonify({
            'status': 'error',
            'message': str(e),
            'traceback': traceback.format_exc()
        }), 500


# =============================================================================
# Preset Management API Endpoints
# =============================================================================

PRESETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'presets')


@app.route('/api/presets', methods=['GET'])
def list_presets():
    """List all available preset configurations."""
    presets = []
    if os.path.exists(PRESETS_DIR):
        for filename in os.listdir(PRESETS_DIR):
            if filename.endswith('.json'):
                preset_name = filename[:-5]  # Remove .json
                preset_path = os.path.join(PRESETS_DIR, filename)
                try:
                    with open(preset_path, 'r') as f:
                        data = json.load(f)
                    presets.append({
                        'name': preset_name,
                        'filename': filename,
                        'generators': len(data.get('generators', [])),
                        'transformers': len(data.get('transformers', [])),
                        'consumers': len(data.get('consumers', [])),
                        'batteries': len(data.get('batteries', [])),
                        'h2_storage': len(data.get('h2_storage', []))
                    })
                except Exception as e:
                    print(f"Error reading preset {filename}: {e}")
    return jsonify(presets)


@app.route('/api/presets/<name>', methods=['GET'])
def get_preset(name):
    """Get a specific preset configuration."""
    preset_path = os.path.join(PRESETS_DIR, f'{name}.json')
    if not os.path.exists(preset_path):
        return jsonify({'status': 'error', 'message': f'Preset {name} not found'}), 404

    with open(preset_path, 'r') as f:
        data = json.load(f)
    return jsonify(data)


@app.route('/api/presets/<name>/load', methods=['POST'])
def load_preset(name):
    """Load a preset and replace current node registry."""
    global node_registry

    preset_path = os.path.join(PRESETS_DIR, f'{name}.json')
    if not os.path.exists(preset_path):
        return jsonify({'status': 'error', 'message': f'Preset {name} not found'}), 404

    with open(preset_path, 'r') as f:
        data = json.load(f)

    # Replace node registry with preset data
    node_registry = {
        'generators': data.get('generators', []),
        'transformers': data.get('transformers', []),
        'consumers': data.get('consumers', []),
        'batteries': data.get('batteries', []),
        'h2_storage': data.get('h2_storage', []),
        'connections': data.get('connections', [])
    }

    # If connections are empty, generate defaults
    if not node_registry['connections']:
        node_registry['connections'] = generate_default_connections()

    # Clean up any stale connections (in case preset has invalid data)
    clean_stale_connections()

    save_node_registry()
    sync_nodes_to_network()

    return jsonify({
        'status': 'ok',
        'message': f'Loaded preset: {name}',
        'nodes': node_registry
    })


# =============================================================================
# Node Management API Endpoints
# =============================================================================

@app.route('/api/nodes', methods=['GET'])
def get_all_nodes():
    """Get all configured nodes."""
    load_node_registry()  # Reload from file to sync across workers
    return jsonify(node_registry)


@app.route('/api/nodes/generators', methods=['GET', 'POST'])
def manage_generators():
    """List or add generators."""
    if request.method == 'GET':
        return jsonify(node_registry['generators'])

    # POST - add new generator
    data = request.json
    if not data.get('name'):
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400

    # Check for duplicate names
    if any(g['name'] == data['name'] for g in node_registry['generators']):
        return jsonify({'status': 'error', 'message': 'Generator with this name already exists'}), 400

    generator = {
        'name': data['name'],
        'capacity': float(data.get('capacity', 100.0)),
        'cost': float(data.get('cost', 8.0)),
        'profile_type': data.get('profile_type', 'solar'),
        'position': data.get('position', [0, 2, 0])
    }
    node_registry['generators'].append(generator)
    add_connections_for_new_node(data['name'], 'generator')
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'generator': generator})


@app.route('/api/nodes/generators/<name>', methods=['GET', 'PUT', 'DELETE'])
def manage_generator(name):
    """Get, update or delete a specific generator."""
    generators = node_registry['generators']
    gen_idx = next((i for i, g in enumerate(generators) if g['name'] == name), None)

    if gen_idx is None:
        return jsonify({'status': 'error', 'message': 'Generator not found'}), 404

    if request.method == 'GET':
        return jsonify(generators[gen_idx])

    if request.method == 'DELETE':
        del generators[gen_idx]
        clean_stale_connections()
        save_node_registry()
        sync_nodes_to_network()
        return jsonify({'status': 'ok', 'message': f'Generator {name} deleted'})

    # PUT - update
    data = request.json
    generators[gen_idx].update({
        'capacity': float(data.get('capacity', generators[gen_idx]['capacity'])),
        'cost': float(data.get('cost', generators[gen_idx]['cost'])),
        'profile_type': data.get('profile_type', generators[gen_idx].get('profile_type', 'solar')),
        'position': data.get('position', generators[gen_idx].get('position', [0, 0, 0]))
    })
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'generator': generators[gen_idx]})


@app.route('/api/nodes/transformers', methods=['GET', 'POST'])
def manage_transformers():
    """List or add transformers."""
    if request.method == 'GET':
        return jsonify(node_registry['transformers'])

    # POST - add new transformer
    data = request.json
    if not data.get('name'):
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400

    if any(t['name'] == data['name'] for t in node_registry['transformers']):
        return jsonify({'status': 'error', 'message': 'Transformer with this name already exists'}), 400

    transformer = {
        'name': data['name'],
        'capacity': float(data.get('capacity', 50.0)),
        'efficiency': float(data.get('efficiency', 0.7)),
        'cost': float(data.get('cost', 1.5)),
        'input_carrier': data.get('input_carrier', 'electricity'),
        'output_carrier': data.get('output_carrier', 'hydrogen'),
        'position': data.get('position', [-2, 0, 0])
    }
    node_registry['transformers'].append(transformer)
    add_connections_for_new_node(data['name'], 'transformer')
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'transformer': transformer})


@app.route('/api/nodes/transformers/<name>', methods=['GET', 'PUT', 'DELETE'])
def manage_transformer(name):
    """Get, update or delete a specific transformer."""
    transformers = node_registry['transformers']
    trans_idx = next((i for i, t in enumerate(transformers) if t['name'] == name), None)

    if trans_idx is None:
        return jsonify({'status': 'error', 'message': 'Transformer not found'}), 404

    if request.method == 'GET':
        return jsonify(transformers[trans_idx])

    if request.method == 'DELETE':
        del transformers[trans_idx]
        clean_stale_connections()
        save_node_registry()
        sync_nodes_to_network()
        return jsonify({'status': 'ok', 'message': f'Transformer {name} deleted'})

    # PUT - update
    data = request.json
    transformers[trans_idx].update({
        'capacity': float(data.get('capacity', transformers[trans_idx]['capacity'])),
        'efficiency': float(data.get('efficiency', transformers[trans_idx]['efficiency'])),
        'cost': float(data.get('cost', transformers[trans_idx]['cost'])),
        'input_carrier': data.get('input_carrier', transformers[trans_idx].get('input_carrier', 'electricity')),
        'output_carrier': data.get('output_carrier', transformers[trans_idx].get('output_carrier', 'hydrogen')),
        'position': data.get('position', transformers[trans_idx].get('position', [0, 0, 0]))
    })
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'transformer': transformers[trans_idx]})


@app.route('/api/nodes/consumers', methods=['GET', 'POST'])
def manage_consumers():
    """List or add consumers."""
    if request.method == 'GET':
        return jsonify(node_registry['consumers'])

    # POST - add new consumer
    data = request.json
    if not data.get('name'):
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400

    if any(c['name'] == data['name'] for c in node_registry['consumers']):
        return jsonify({'status': 'error', 'message': 'Consumer with this name already exists'}), 400

    consumer_type = data.get('consumer_type', 'logarithmic_electricity')

    consumer = {
        'name': data['name'],
        'consumer_type': consumer_type,
        'carrier': data.get('carrier', 'electricity' if 'electricity' in consumer_type else
                           'hydrogen' if 'hydrogen' in consumer_type else 'both'),
        'position': data.get('position', [0, -2, 0])
    }

    # Add type-specific parameters
    if consumer_type.startswith('logarithmic'):
        consumer.update({
            'log_scale': float(data.get('log_scale', 30.0)),
            'log_shape': float(data.get('log_shape', 5.0)),
            'log_max_quantity': float(data.get('log_max_quantity', 25.0)),
            'log_num_tranches': int(data.get('log_num_tranches', 5))
        })
    elif consumer_type == 'cobb_douglas':
        consumer.update({
            'cd_A': float(data.get('cd_A', 15.0)),
            'cd_alpha': float(data.get('cd_alpha', 0.4)),
            'cd_beta': float(data.get('cd_beta', 0.6)),
            'cd_max_bundles': float(data.get('cd_max_bundles', 15.0)),
            'cd_h_per_bundle': float(data.get('cd_h_per_bundle', 1.0)),
            'cd_e_per_bundle': float(data.get('cd_e_per_bundle', 3.0)),
            'cd_num_tranches': int(data.get('cd_num_tranches', 5))
        })

    node_registry['consumers'].append(consumer)
    add_connections_for_new_node(data['name'], 'consumer', consumer['carrier'])
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'consumer': consumer})


@app.route('/api/nodes/consumers/<name>', methods=['GET', 'PUT', 'DELETE'])
def manage_consumer(name):
    """Get, update or delete a specific consumer."""
    consumers = node_registry['consumers']
    cons_idx = next((i for i, c in enumerate(consumers) if c['name'] == name), None)

    if cons_idx is None:
        return jsonify({'status': 'error', 'message': 'Consumer not found'}), 404

    if request.method == 'GET':
        return jsonify(consumers[cons_idx])

    if request.method == 'DELETE':
        del consumers[cons_idx]
        clean_stale_connections()
        save_node_registry()
        sync_nodes_to_network()
        return jsonify({'status': 'ok', 'message': f'Consumer {name} deleted'})

    # PUT - update
    data = request.json
    consumer = consumers[cons_idx]

    # Update common fields
    consumer['position'] = data.get('position', consumer.get('position', [0, 0, 0]))

    # Update consumer_type and carrier if provided
    new_type = data.get('consumer_type', consumer['consumer_type'])
    consumer['consumer_type'] = new_type
    consumer['carrier'] = data.get('carrier', consumer.get('carrier', 'electricity'))

    # Update type-specific fields based on the NEW type
    if new_type.startswith('logarithmic'):
        consumer.update({
            'log_scale': float(data.get('log_scale', consumer.get('log_scale', 30.0))),
            'log_shape': float(data.get('log_shape', consumer.get('log_shape', 5.0))),
            'log_max_quantity': float(data.get('log_max_quantity', consumer.get('log_max_quantity', 25.0))),
            'log_num_tranches': int(data.get('log_num_tranches', consumer.get('log_num_tranches', 5)))
        })
    elif new_type == 'cobb_douglas':
        consumer.update({
            'cd_A': float(data.get('cd_A', consumer.get('cd_A', 15.0))),
            'cd_alpha': float(data.get('cd_alpha', consumer.get('cd_alpha', 0.4))),
            'cd_beta': float(data.get('cd_beta', consumer.get('cd_beta', 0.6))),
            'cd_max_bundles': float(data.get('cd_max_bundles', consumer.get('cd_max_bundles', 15.0))),
            'cd_h_per_bundle': float(data.get('cd_h_per_bundle', consumer.get('cd_h_per_bundle', 1.0))),
            'cd_e_per_bundle': float(data.get('cd_e_per_bundle', consumer.get('cd_e_per_bundle', 3.0))),
            'cd_num_tranches': int(data.get('cd_num_tranches', consumer.get('cd_num_tranches', 5)))
        })

    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'consumer': consumer})


# =============================================================================
# Storage API Endpoints
# =============================================================================

@app.route('/api/nodes/batteries', methods=['GET', 'POST'])
def manage_batteries():
    """List or add batteries."""
    if request.method == 'GET':
        return jsonify(node_registry.get('batteries', []))

    # POST - add new battery
    data = request.json
    if not data.get('name'):
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400

    if 'batteries' not in node_registry:
        node_registry['batteries'] = []

    if any(b['name'] == data['name'] for b in node_registry['batteries']):
        return jsonify({'status': 'error', 'message': 'Battery with this name already exists'}), 400

    battery = {
        'name': data['name'],
        'capacity': float(data.get('capacity', 100.0)),
        'charge_rate': float(data.get('charge_rate', 50.0)),
        'discharge_rate': float(data.get('discharge_rate', 50.0)),
        'efficiency_in': float(data.get('efficiency_in', 0.95)),
        'efficiency_out': float(data.get('efficiency_out', 0.95)),
        'loss_rate': float(data.get('loss_rate', 0.0002)),
        'initial_level': float(data.get('initial_level', 0.5)),
        'min_level': float(data.get('min_level', 0.1)),
        'cost': float(data.get('cost', 1.0)),
        'position': data.get('position', [3, 0, 0])
    }
    node_registry['batteries'].append(battery)
    add_connections_for_new_node(data['name'], 'battery')
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'battery': battery})


@app.route('/api/nodes/batteries/<name>', methods=['GET', 'PUT', 'DELETE'])
def manage_battery(name):
    """Get, update or delete a specific battery."""
    batteries = node_registry.get('batteries', [])
    batt_idx = next((i for i, b in enumerate(batteries) if b['name'] == name), None)

    if batt_idx is None:
        return jsonify({'status': 'error', 'message': 'Battery not found'}), 404

    if request.method == 'GET':
        return jsonify(batteries[batt_idx])

    if request.method == 'DELETE':
        del batteries[batt_idx]
        clean_stale_connections()
        save_node_registry()
        sync_nodes_to_network()
        return jsonify({'status': 'ok', 'message': f'Battery {name} deleted'})

    # PUT - update
    data = request.json
    batteries[batt_idx].update({
        'capacity': float(data.get('capacity', batteries[batt_idx]['capacity'])),
        'charge_rate': float(data.get('charge_rate', batteries[batt_idx].get('charge_rate', 50.0))),
        'discharge_rate': float(data.get('discharge_rate', batteries[batt_idx].get('discharge_rate', 50.0))),
        'efficiency_in': float(data.get('efficiency_in', batteries[batt_idx].get('efficiency_in', 0.95))),
        'efficiency_out': float(data.get('efficiency_out', batteries[batt_idx].get('efficiency_out', 0.95))),
        'loss_rate': float(data.get('loss_rate', batteries[batt_idx].get('loss_rate', 0.0002))),
        'initial_level': float(data.get('initial_level', batteries[batt_idx].get('initial_level', 0.5))),
        'min_level': float(data.get('min_level', batteries[batt_idx].get('min_level', 0.1))),
        'cost': float(data.get('cost', batteries[batt_idx].get('cost', 1.0))),
        'position': data.get('position', batteries[batt_idx].get('position', [3, 0, 0]))
    })
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'battery': batteries[batt_idx]})


@app.route('/api/nodes/h2_storage', methods=['GET', 'POST'])
def manage_h2_storage_list():
    """List or add H2 storage."""
    if request.method == 'GET':
        return jsonify(node_registry.get('h2_storage', []))

    # POST - add new H2 storage
    data = request.json
    if not data.get('name'):
        return jsonify({'status': 'error', 'message': 'Name is required'}), 400

    if 'h2_storage' not in node_registry:
        node_registry['h2_storage'] = []

    if any(h['name'] == data['name'] for h in node_registry['h2_storage']):
        return jsonify({'status': 'error', 'message': 'H2 storage with this name already exists'}), 400

    h2_storage = {
        'name': data['name'],
        'capacity': float(data.get('capacity', 50.0)),
        'injection_rate': float(data.get('injection_rate', 10.0)),
        'withdrawal_rate': float(data.get('withdrawal_rate', 10.0)),
        'efficiency_in': float(data.get('efficiency_in', 0.95)),
        'efficiency_out': float(data.get('efficiency_out', 0.99)),
        'loss_rate': float(data.get('loss_rate', 0.0001)),
        'initial_level': float(data.get('initial_level', 0.5)),
        'cost': float(data.get('cost', 0.5)),
        'position': data.get('position', [-3, -1, 0])
    }
    node_registry['h2_storage'].append(h2_storage)
    add_connections_for_new_node(data['name'], 'h2_storage')
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'h2_storage': h2_storage})


@app.route('/api/nodes/h2_storage/<name>', methods=['GET', 'PUT', 'DELETE'])
def manage_h2_storage(name):
    """Get, update or delete a specific H2 storage."""
    h2_storage_list = node_registry.get('h2_storage', [])
    h2_idx = next((i for i, h in enumerate(h2_storage_list) if h['name'] == name), None)

    if h2_idx is None:
        return jsonify({'status': 'error', 'message': 'H2 storage not found'}), 404

    if request.method == 'GET':
        return jsonify(h2_storage_list[h2_idx])

    if request.method == 'DELETE':
        del h2_storage_list[h2_idx]
        clean_stale_connections()
        save_node_registry()
        sync_nodes_to_network()
        return jsonify({'status': 'ok', 'message': f'H2 storage {name} deleted'})

    # PUT - update
    data = request.json
    h2_storage_list[h2_idx].update({
        'capacity': float(data.get('capacity', h2_storage_list[h2_idx]['capacity'])),
        'injection_rate': float(data.get('injection_rate', h2_storage_list[h2_idx].get('injection_rate', 10.0))),
        'withdrawal_rate': float(data.get('withdrawal_rate', h2_storage_list[h2_idx].get('withdrawal_rate', 10.0))),
        'efficiency_in': float(data.get('efficiency_in', h2_storage_list[h2_idx].get('efficiency_in', 0.95))),
        'efficiency_out': float(data.get('efficiency_out', h2_storage_list[h2_idx].get('efficiency_out', 0.99))),
        'loss_rate': float(data.get('loss_rate', h2_storage_list[h2_idx].get('loss_rate', 0.0001))),
        'initial_level': float(data.get('initial_level', h2_storage_list[h2_idx].get('initial_level', 0.5))),
        'cost': float(data.get('cost', h2_storage_list[h2_idx].get('cost', 0.5))),
        'position': data.get('position', h2_storage_list[h2_idx].get('position', [-3, -1, 0]))
    })
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'h2_storage': h2_storage_list[h2_idx]})


# =============================================================================
# Connection Management API Endpoints
# =============================================================================

@app.route('/api/connections', methods=['GET'])
def get_connections():
    """Get all connections."""
    load_node_registry()  # Reload from file to sync across workers
    ensure_connections_exist()
    return jsonify(node_registry.get('connections', []))


@app.route('/api/connections/regenerate', methods=['POST'])
def regenerate_connections():
    """Regenerate default connections from node types."""
    load_node_registry()  # Reload from file to sync across workers
    node_registry['connections'] = generate_default_connections()
    save_node_registry()
    sync_nodes_to_network()
    return jsonify({
        'status': 'ok',
        'connections': node_registry['connections'],
        'message': f'Regenerated {len(node_registry["connections"])} connections'
    })


@app.route('/api/connections/toggle', methods=['POST'])
def toggle_connection():
    """Toggle a connection's enabled state."""
    load_node_registry()  # Reload from file to sync across workers

    data = request.json
    source = data.get('source')
    target = data.get('target')
    carrier = data.get('carrier')

    if not source or not target:
        return jsonify({'status': 'error', 'message': 'Source and target required'}), 400

    # Find and toggle the connection
    connections = node_registry.get('connections', [])
    for conn in connections:
        if conn['source'] == source and conn['target'] == target:
            if carrier and conn['carrier'] != carrier:
                continue
            conn['enabled'] = not conn.get('enabled', True)
            save_node_registry()
            sync_nodes_to_network()
            return jsonify({
                'status': 'ok',
                'connection': conn,
                'message': f'Connection {"enabled" if conn["enabled"] else "disabled"}'
            })

    return jsonify({'status': 'error', 'message': 'Connection not found'}), 404


@app.route('/api/connections', methods=['POST'])
def add_connection():
    """Add a new connection."""
    load_node_registry()  # Reload from file to sync across workers

    data = request.json
    source = data.get('source')
    target = data.get('target')
    carrier = data.get('carrier', 'electricity')

    if not source or not target:
        return jsonify({'status': 'error', 'message': 'Source and target required'}), 400

    # Check if connection already exists
    connections = node_registry.get('connections', [])
    for conn in connections:
        if conn['source'] == source and conn['target'] == target and conn['carrier'] == carrier:
            return jsonify({'status': 'error', 'message': 'Connection already exists'}), 400

    # Add new connection
    new_conn = {
        'source': source,
        'target': target,
        'carrier': carrier,
        'enabled': True
    }
    connections.append(new_conn)
    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'connection': new_conn})


@app.route('/api/connections', methods=['DELETE'])
def delete_connection():
    """Delete a connection."""
    load_node_registry()  # Reload from file to sync across workers

    data = request.json
    source = data.get('source')
    target = data.get('target')
    carrier = data.get('carrier')

    if not source or not target:
        return jsonify({'status': 'error', 'message': 'Source and target required'}), 400

    connections = node_registry.get('connections', [])
    for i, conn in enumerate(connections):
        if conn['source'] == source and conn['target'] == target:
            if carrier and conn['carrier'] != carrier:
                continue
            del connections[i]
            save_node_registry()
            sync_nodes_to_network()
            return jsonify({'status': 'ok', 'message': 'Connection deleted'})

    return jsonify({'status': 'error', 'message': 'Connection not found'}), 404


@app.route('/api/layout', methods=['GET'])
def get_layout():
    """Get current node positions as a layout."""
    load_node_registry()  # Reload from file to sync across workers
    layout = {}
    for gen in node_registry['generators']:
        layout[gen['name']] = gen.get('position', [0, 2, 0])
    for trans in node_registry['transformers']:
        layout[trans['name']] = trans.get('position', [-2, 0, 0])
    for cons in node_registry['consumers']:
        layout[cons['name']] = cons.get('position', [0, -2, 0])
    for batt in node_registry.get('batteries', []):
        layout[batt['name']] = batt.get('position', [3, 0, 0])
    for h2s in node_registry.get('h2_storage', []):
        layout[h2s['name']] = h2s.get('position', [-3, -1, 0])
    return jsonify(layout)


@app.route('/api/layout', methods=['POST'])
def update_layout():
    """Update node positions from a layout dict."""
    data = request.json
    if not data:
        return jsonify({'status': 'error', 'message': 'No layout data provided'}), 400

    # Update positions in each node collection
    for gen in node_registry['generators']:
        if gen['name'] in data:
            gen['position'] = data[gen['name']]
    for trans in node_registry['transformers']:
        if trans['name'] in data:
            trans['position'] = data[trans['name']]
    for cons in node_registry['consumers']:
        if cons['name'] in data:
            cons['position'] = data[cons['name']]
    for batt in node_registry.get('batteries', []):
        if batt['name'] in data:
            batt['position'] = data[batt['name']]
    for h2s in node_registry.get('h2_storage', []):
        if h2s['name'] in data:
            h2s['position'] = data[h2s['name']]

    save_node_registry()
    sync_nodes_to_network()

    return jsonify({'status': 'ok', 'message': f'Updated {len(data)} node positions'})


@app.route('/api/blender/export', methods=['POST'])
def export_to_blender():
    """Export network to JSON for Blender to read."""
    try:
        G = create_hydrogen_network(use_json=True)
        export_network_to_json(G, JSON_PATH)

        return jsonify({
            'status': 'ok',
            'message': f'Exported to {JSON_PATH}',
            'path': JSON_PATH
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/blender/launch', methods=['POST'])
def launch_blender():
    """Launch Blender with the visualization script."""
    try:
        blender_script = os.path.join(SCRIPT_DIR, 'blender_visualize.py')

        # Try common Blender paths
        blender_paths = [
            r'C:\Program Files\Blender Foundation\Blender 4.5\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 4.4\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 4.3\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 4.2\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 4.1\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 4.0\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 3.6\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender\blender.exe',
            'blender',  # If in PATH
            '/Applications/Blender.app/Contents/MacOS/Blender',
            '/usr/bin/blender',
        ]

        blender_exe = None
        for path in blender_paths:
            if os.path.exists(path) or path == 'blender':
                blender_exe = path
                break

        if blender_exe:
            # Launch Blender in background
            subprocess.Popen([blender_exe, '--python', blender_script])
            return jsonify({
                'status': 'ok',
                'message': 'Blender launched',
                'script': blender_script
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Blender not found. Please launch manually.',
                'script': blender_script
            }), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/blender/sync', methods=['POST'])
def sync_from_blender():
    """
    Read network_data.json that was exported from Blender.
    This assumes the user ran sync_from_blender() in Blender first.
    """
    try:
        if os.path.exists(JSON_PATH):
            with open(JSON_PATH, 'r') as f:
                network = json.load(f)

            return jsonify({
                'status': 'ok',
                'message': 'Synced from Blender',
                'network': network
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'No network_data.json found. Run sync_from_blender() in Blender first.'
            }), 404

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# =============================================================================
# Deploy Webhook (for VPS updates without SSH key)
# =============================================================================

@app.route('/api/deploy', methods=['POST'])
def deploy_webhook():
    """
    Pull latest code from GitHub and reload.
    Call this endpoint to update the VPS without SSH access.
    """
    try:
        import subprocess as sp

        # Pull latest from GitHub
        result = sp.run(
            ['git', 'pull', 'origin', 'main'],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=60
        )

        output = result.stdout + result.stderr

        if result.returncode == 0:
            # Reload node registry in case it changed
            load_node_registry()

            return jsonify({
                'status': 'ok',
                'message': 'Deployed successfully! Restart service for full effect.',
                'output': output,
                'hint': 'Run: sudo systemctl restart hydrogen'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Git pull failed',
                'output': output
            }), 500

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    print("=" * 60)
    print("  Hydrogen Simulation Controller")
    print("=" * 60)
    print(f"\nOpen http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=True, port=5000)
