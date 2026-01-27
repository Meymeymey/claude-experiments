# Hydrogen System Simulation

A simulation of a hydrogen/electricity network with Blender visualization.

## System Architecture

```
Alpha (Electricity Producer)
    |
    | electricity
    v
+---+---+
|       |
v       v
Bravo   Delta (Electricity Consumer)
(Electrolyzer)
    |
    | hydrogen
    v
Charlie (Hydrogen Consumer)
```

## Components

| Node | Type | Function |
|------|------|----------|
| Alpha | Producer | Generates electricity (e.g., solar/wind) |
| Bravo | Converter | Electrolyzer - converts electricity to hydrogen |
| Charlie | Consumer | Consumes hydrogen (e.g., fuel cell, industrial) |
| Delta | Consumer | Consumes electricity directly |

## Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install CBC solver (required for oemof)
# On Windows (via conda):
conda install -c conda-forge coincbc

# On Linux:
sudo apt-get install coinor-cbc

# On macOS:
brew install cbc
```

## Usage

### 1. Run the simulation

```bash
python main.py
```

This will:
- Create the network topology
- Run the energy flow simulation
- Export `network_data.json` and `simulation_results.json`

### 2. Visualize in Blender

1. Open Blender
2. Go to **Scripting** workspace
3. Open `blender_visualize.py`
4. Click **Run Script**

Or from command line:
```bash
blender --python blender_visualize.py
```

## File Structure

```
hydrogen_simulation/
├── main.py              # Main entry point
├── network.py           # NetworkX topology definition
├── simulation.py        # oemof energy simulation
├── blender_visualize.py # Blender visualization script
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## Configuration

Edit `main.py` to change simulation parameters:

```python
config = SimulationConfig(
    periods=24,              # Simulation time periods (hours)
    alpha_capacity=100.0,    # Electricity production capacity (kW)
    bravo_capacity=50.0,     # Electrolyzer capacity (kW)
    bravo_efficiency=0.7,    # Electrolyzer efficiency (70%)
    charlie_demand=10.0,     # Hydrogen demand (kg/h)
    delta_demand=30.0,       # Electricity demand (kW)
)
```

## Blender Visualization

The visualization uses different shapes and colors:

**Node Types:**
- 🟢 **Cone** (Green) - Producers
- 🟡 **Cube** (Yellow) - Converters
- 🔴 **Sphere** (Red) - Consumers

**Pipe Colors:**
- **Gold** - Electricity cables
- **Blue** - Hydrogen pipes
