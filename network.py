"""
Network topology module using NetworkX.
Defines the spatial layout of the hydrogen/electricity system.
"""

import networkx as nx
import json
from dataclasses import dataclass
from typing import Dict, Tuple, List


@dataclass
class Node:
    """Represents a point in the hydrogen/electricity network."""
    name: str
    node_type: str  # 'producer', 'converter', 'consumer'
    carrier: str    # 'electricity', 'hydrogen', 'both'
    position: Tuple[float, float, float]  # x, y, z coordinates for Blender


@dataclass
class Pipe:
    """Represents a connection (pipe/cable) between nodes."""
    source: str
    target: str
    carrier: str  # 'electricity' or 'hydrogen'


def create_hydrogen_network() -> nx.DiGraph:
    """
    Create the network topology for the hydrogen system.

    Layout:
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
    """
    G = nx.DiGraph()

    # Define nodes with their 3D positions (x, y, z) - flat on XY plane
    nodes = {
        'Alpha': Node(
            name='Alpha',
            node_type='producer',
            carrier='electricity',
            position=(0.0, 2.0, 0.0)  # Top center
        ),
        'Bravo': Node(
            name='Bravo',
            node_type='converter',
            carrier='both',  # Converts electricity to hydrogen
            position=(-2.0, 0.0, 0.0)  # Middle left
        ),
        'Charlie': Node(
            name='Charlie',
            node_type='consumer',
            carrier='hydrogen',
            position=(-2.0, -2.0, 0.0)  # Bottom left
        ),
        'Delta': Node(
            name='Delta',
            node_type='consumer',
            carrier='electricity',
            position=(2.0, 0.0, 0.0)  # Middle right
        ),
    }

    # Add nodes to graph
    for name, node in nodes.items():
        G.add_node(
            name,
            node_type=node.node_type,
            carrier=node.carrier,
            position=node.position
        )

    # Define connections (pipes/cables)
    pipes = [
        Pipe('Alpha', 'Bravo', 'electricity'),   # Power to electrolyzer
        Pipe('Alpha', 'Delta', 'electricity'),   # Power to electricity consumer
        Pipe('Bravo', 'Charlie', 'hydrogen'),    # Hydrogen to consumer
    ]

    # Add edges to graph
    for pipe in pipes:
        G.add_edge(
            pipe.source,
            pipe.target,
            carrier=pipe.carrier
        )

    return G


def get_node_positions(G: nx.DiGraph) -> Dict[str, Tuple[float, float, float]]:
    """Extract node positions from the graph."""
    return {node: data['position'] for node, data in G.nodes(data=True)}


def get_edges_by_carrier(G: nx.DiGraph, carrier: str) -> List[Tuple[str, str]]:
    """Get all edges that transport a specific carrier."""
    return [
        (u, v) for u, v, data in G.edges(data=True)
        if data['carrier'] == carrier
    ]


def export_network_to_json(G: nx.DiGraph, filepath: str) -> None:
    """Export network data to JSON for Blender import."""
    data = {
        'nodes': [
            {
                'name': node,
                'node_type': attrs['node_type'],
                'carrier': attrs['carrier'],
                'position': list(attrs['position'])
            }
            for node, attrs in G.nodes(data=True)
        ],
        'edges': [
            {
                'source': u,
                'target': v,
                'carrier': attrs['carrier']
            }
            for u, v, attrs in G.edges(data=True)
        ]
    }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Network exported to {filepath}")


def print_network_info(G: nx.DiGraph) -> None:
    """Print network topology information."""
    print("\n=== Network Topology ===")
    print(f"Nodes: {G.number_of_nodes()}")
    print(f"Edges: {G.number_of_edges()}")

    print("\nNodes:")
    for node, data in G.nodes(data=True):
        print(f"  {node}: {data['node_type']} ({data['carrier']}) at {data['position']}")

    print("\nConnections:")
    for u, v, data in G.edges(data=True):
        print(f"  {u} -> {v}: {data['carrier']}")


if __name__ == "__main__":
    # Test the network creation
    G = create_hydrogen_network()
    print_network_info(G)
    export_network_to_json(G, "network_data.json")
