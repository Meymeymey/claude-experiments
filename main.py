"""
Main entry point for the Hydrogen System Simulation.

This script:
1. Creates the network topology using NetworkX
2. Runs the energy simulation using oemof
3. Exports data for Blender visualization
"""

import os
import sys

from network import (
    create_hydrogen_network,
    print_network_info,
    export_network_to_json,
)
from simulation import (
    SimulationConfig,
    run_simulation,
)


def main():
    """Run the complete hydrogen system simulation pipeline."""
    print("=" * 60)
    print("  Hydrogen System Simulation")
    print("=" * 60)

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Step 1: Create network topology
    print("\n[Step 1/3] Creating network topology...")
    network = create_hydrogen_network()
    print_network_info(network)

    # Export network for Blender
    network_json_path = os.path.join(script_dir, "network_data.json")
    export_network_to_json(network, network_json_path)

    # Step 2: Configure and run simulation
    print("\n[Step 2/3] Running energy simulation...")

    config = SimulationConfig(
        periods=24,              # 24 hours
        alpha_capacity=100.0,    # 100 kW electricity production
        bravo_capacity=50.0,     # 50 kW electrolyzer capacity
        bravo_efficiency=0.7,    # 70% efficiency
        charlie_demand=10.0,     # 10 kg/h hydrogen demand
        delta_demand=30.0,       # 30 kW electricity demand
    )

    print(f"\nSimulation Configuration:")
    print(f"  - Periods: {config.periods} hours")
    print(f"  - Alpha (Producer) capacity: {config.alpha_capacity} kW")
    print(f"  - Bravo (Electrolyzer) capacity: {config.bravo_capacity} kW")
    print(f"  - Bravo (Electrolyzer) efficiency: {config.bravo_efficiency * 100}%")
    print(f"  - Charlie (H2 Consumer) demand: {config.charlie_demand} kg/h")
    print(f"  - Delta (Elec Consumer) demand: {config.delta_demand} kW")

    try:
        sim = run_simulation(config)
        sim.print_results()

        # Export results for Blender
        results_json_path = os.path.join(script_dir, "simulation_results.json")
        sim.export_results_to_json(results_json_path)
    except Exception as e:
        print(f"\nWarning: Simulation failed: {e}")
        print("This may be due to missing solver (CBC). Install with:")
        print("  conda install -c conda-forge coincbc")
        print("  or: apt-get install coinor-cbc (Linux)")
        print("\nContinuing without simulation results...")

    # Step 3: Instructions for Blender
    print("\n[Step 3/3] Blender Visualization")
    print("-" * 40)
    print("To visualize in Blender:")
    print(f"  1. Open Blender")
    print(f"  2. Go to 'Scripting' workspace")
    print(f"  3. Open: {os.path.join(script_dir, 'blender_visualize.py')}")
    print(f"  4. Click 'Run Script'")
    print("\nOr run from command line:")
    print(f"  blender --python \"{os.path.join(script_dir, 'blender_visualize.py')}\"")

    print("\n" + "=" * 60)
    print("  Simulation Complete!")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
