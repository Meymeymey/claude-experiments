"""
Hydrogen connection type for the hydrogen simulation network.

Specialized connection type for hydrogen flows with pressure and
flow-specific attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .base import Connection, ConnectionType


@dataclass
class HydrogenConnection(Connection):
    """
    Hydrogen flow connection.

    Represents a hydrogen pipeline or connection between nodes with
    optional pressure and flow specifications.

    Attributes:
        source: Name of the source node
        target: Name of the target node
        enabled: Whether the connection is active
        capacity: Maximum flow rate in kg/h
        loss: Leakage/loss fraction
        pressure: Operating pressure in bar (optional)
        diameter: Pipe diameter in mm (optional)
    """

    _pressure: Optional[float] = field(default=None)
    _diameter: Optional[float] = field(default=None)

    def __post_init__(self):
        """Initialize with hydrogen carrier."""
        self._carrier = ConnectionType.HYDROGEN
        super().__post_init__()

    # Pressure property
    @property
    def pressure(self) -> Optional[float]:
        """Get the operating pressure in bar."""
        return self._pressure

    @pressure.setter
    def pressure(self, value: Optional[float]) -> None:
        """Set the operating pressure in bar."""
        if value is not None and value <= 0:
            raise ValueError("Pressure must be positive")
        self._pressure = float(value) if value is not None else None

    # Diameter property
    @property
    def diameter(self) -> Optional[float]:
        """Get the pipe diameter in mm."""
        return self._diameter

    @diameter.setter
    def diameter(self, value: Optional[float]) -> None:
        """Set the pipe diameter in mm."""
        if value is not None and value <= 0:
            raise ValueError("Diameter must be positive")
        self._diameter = float(value) if value is not None else None

    @property
    def cross_section_area(self) -> Optional[float]:
        """Calculate pipe cross-section area in mm²."""
        if self._diameter is None:
            return None
        import math
        return math.pi * (self._diameter / 2) ** 2

    def calculate_velocity(self, flow_rate: float, density: float = 0.089) -> Optional[float]:
        """
        Calculate flow velocity for given mass flow rate.

        Args:
            flow_rate: Mass flow rate in kg/h
            density: Hydrogen density in kg/m³ (default: 0.089 at STP)

        Returns:
            Velocity in m/s, or None if diameter not set
        """
        if self._diameter is None:
            return None

        # Convert diameter from mm to m
        d_m = self._diameter / 1000

        # Cross-section area in m²
        import math
        area = math.pi * (d_m / 2) ** 2

        # Volume flow rate in m³/s
        vol_flow = (flow_rate / density) / 3600

        # Velocity in m/s
        return vol_flow / area

    def to_dict(self) -> Dict[str, Any]:
        """Serialize hydrogen connection to dictionary."""
        data = super().to_dict()
        data['connection_type'] = 'hydrogen'
        if self._pressure is not None:
            data['pressure'] = self._pressure
        if self._diameter is not None:
            data['diameter'] = self._diameter
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HydrogenConnection':
        """Create HydrogenConnection from dictionary."""
        connection = cls(
            _source=data.get('source', ''),
            _target=data.get('target', ''),
            _enabled=data.get('enabled', True),
            _capacity=data.get('capacity'),
            _loss=float(data.get('loss', 0.0)),
            _pressure=data.get('pressure'),
            _diameter=data.get('diameter'),
        )
        return connection

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        cap = f", {self._capacity}kg/h" if self._capacity else ""
        press = f", {self._pressure}bar" if self._pressure else ""
        return f"HydrogenConnection({self._source}->{self._target}{cap}{press}, {status})"
