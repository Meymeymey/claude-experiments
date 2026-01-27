"""
Flask-based controller for the hydrogen simulation.
Provides a web interface to control Blender sync, run simulations, and adjust parameters.
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

# Optional: simulation requires oemof which needs HiGHS solver
SIMULATION_AVAILABLE = False
try:
    from simulation import (
        SimulationConfig, HydrogenSystemSimulation,
        LogarithmicUtility, CobbDouglasUtility
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
    """Run the energy simulation with current config."""
    global last_results

    if not SIMULATION_AVAILABLE:
        return jsonify({
            'status': 'error',
            'message': 'Simulation not available. Install oemof: pip install oemof.solph pandas highspy'
        }), 503

    try:
        # Create utility function objects from config
        charlie_utility = LogarithmicUtility(
            scale=current_config['charlie_scale'],
            shape=current_config['charlie_shape'],
            max_quantity=current_config['charlie_max_quantity'],
            num_tranches=int(current_config['charlie_num_tranches'])
        )

        delta_utility = LogarithmicUtility(
            scale=current_config['delta_scale'],
            shape=current_config['delta_shape'],
            max_quantity=current_config['delta_max_quantity'],
            num_tranches=int(current_config['delta_num_tranches'])
        )

        echo_utility = CobbDouglasUtility(
            A=current_config['echo_A'],
            alpha=current_config['echo_alpha'],
            beta=current_config['echo_beta'],
            max_bundles=current_config['echo_max_bundles'],
            h_per_bundle=current_config['echo_h_per_bundle'],
            e_per_bundle=current_config['echo_e_per_bundle'],
            num_tranches=int(current_config['echo_num_tranches'])
        )

        config = SimulationConfig(
            periods=int(current_config['periods']),
            alpha_capacity=current_config['alpha_capacity'],
            alpha_cost=current_config['alpha_cost'],
            bravo_capacity=current_config['bravo_capacity'],
            bravo_efficiency=current_config['bravo_efficiency'],
            bravo_cost=current_config['bravo_cost'],
            grid_cost=current_config['grid_cost'],
            charlie_utility=charlie_utility,
            delta_utility=delta_utility,
            echo_utility=echo_utility,
            include_echo=current_config['include_echo'],
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
        for name, df in flows.items():
            # Replace NaN with None for JSON compatibility, then filter
            values = df.values.tolist()
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
            'config': current_config.copy()
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


if __name__ == '__main__':
    print("=" * 60)
    print("  Hydrogen Simulation Controller")
    print("=" * 60)
    print(f"\nOpen http://localhost:5000 in your browser")
    print("\nPress Ctrl+C to stop the server")
    print("=" * 60)

    app.run(debug=True, port=5000)
