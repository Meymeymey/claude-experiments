"""
Connection types for the hydrogen simulation network.

This module provides connection types that represent energy flows
between nodes in the network:
- Connection: Base connection between any two nodes
- ElectricityConnection: Electricity flow connection
- HydrogenConnection: Hydrogen flow connection
"""

from .base import Connection, ConnectionType
from .electricity import ElectricityConnection
from .hydrogen import HydrogenConnection
from .factory import ConnectionFactory

__all__ = [
    'Connection',
    'ConnectionType',
    'ElectricityConnection',
    'HydrogenConnection',
    'ConnectionFactory',
]
