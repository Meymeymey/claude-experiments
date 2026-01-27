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

# Optional: simulation requires oemof which needs CBC solver
SIMULATION_AVAILABLE = False
try:
    from simulation import SimulationConfig, HydrogenSystemSimulation
    SIMULATION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Simulation not available ({e})")
    print("Install with: pip install oemof.solph pandas")
    print("Also install CBC solver: conda install -c conda-forge coincbc")

app = Flask(__name__, template_folder='.', static_folder='.')
CORS(app)

# Store current configuration
current_config = {
    'periods': 24,
    'alpha_capacity': 100.0,
    'bravo_capacity': 50.0,
    'bravo_efficiency': 0.7,
    'charlie_demand': 10.0,
    'delta_demand': 30.0,
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
            'message': 'Simulation not available. Install oemof: pip install oemof.solph pandas',
            'hint': 'Also install CBC solver: conda install -c conda-forge coincbc'
        }), 503

    try:
        config = SimulationConfig(
            periods=int(current_config['periods']),
            alpha_capacity=current_config['alpha_capacity'],
            bravo_capacity=current_config['bravo_capacity'],
            bravo_efficiency=current_config['bravo_efficiency'],
            charlie_demand=current_config['charlie_demand'],
            delta_demand=current_config['delta_demand'],
        )

        sim = HydrogenSystemSimulation(config)
        sim.build_system()
        sim.solve()

        # Get results
        summary = sim.get_summary()
        flows = sim.get_flow_results()

        # Convert flows to serializable format
        flow_data = {}
        for name, df in flows.items():
            flow_data[name] = df.values.tolist()

        last_results = {
            'status': 'ok',
            'summary': summary,
            'flows': flow_data,
            'config': current_config.copy()
        }

        return jsonify(last_results)

    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e),
            'hint': 'Make sure CBC solver is installed: conda install -c conda-forge coincbc'
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
            'blender',  # If in PATH
            r'C:\Program Files\Blender Foundation\Blender 4.0\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender 3.6\blender.exe',
            r'C:\Program Files\Blender Foundation\Blender\blender.exe',
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
