"""
Electricity connection type for the hydrogen simulation network.

Specialized connection type for electricity flows with voltage and
power-specific attributes.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from .base import Connection, ConnectionType


@dataclass
class ElectricityConnection(Connection):
    """
    Electricity flow connection.

    Represents an electrical connection between nodes with optional
    voltage level and power rating specifications.

    Attributes:
        source: Name of the source node
        target: Name of the target node
        enabled: Whether the connection is active
        capacity: Maximum power transfer capacity in kW
        loss: Transmission loss fraction
        voltage: Operating voltage in V (optional)
        power_factor: Power factor (optional, default 1.0)
    """

    _voltage: Optional[float] = field(default=None)
    _power_factor: float = field(default=1.0)

    def __post_init__(self):
        """Initialize with electricity carrier."""
        self._carrier = ConnectionType.ELECTRICITY
        super().__post_init__()

    # Voltage property
    @property
    def voltage(self) -> Optional[float]:
        """Get the operating voltage in V."""
        return self._voltage

    @voltage.setter
    def voltage(self, value: Optional[float]) -> None:
        """Set the operating voltage in V."""
        if value is not None and value <= 0:
            raise ValueError("Voltage must be positive")
        self._voltage = float(value) if value is not None else None

    # Power factor property
    @property
    def power_factor(self) -> float:
        """Get the power factor."""
        return self._power_factor

    @power_factor.setter
    def power_factor(self, value: float) -> None:
        """Set the power factor."""
        if not 0 < value <= 1:
            raise ValueError("Power factor must be between 0 and 1")
        self._power_factor = float(value)

    @property
    def apparent_power_capacity(self) -> Optional[float]:
        """Calculate apparent power capacity in kVA."""
        if self._capacity is None:
            return None
        return self._capacity / self._power_factor

    def calculate_current(self, power: float) -> Optional[float]:
        """
        Calculate current for given power transfer.

        Args:
            power: Power in kW

        Returns:
            Current in A, or None if voltage not set
        """
        if self._voltage is None or self._voltage <= 0:
            return None
        # I = P / (V * PF) for single phase
        return (power * 1000) / (self._voltage * self._power_factor)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize electricity connection to dictionary."""
        data = super().to_dict()
        data['connection_type'] = 'electricity'
        if self._voltage is not None:
            data['voltage'] = self._voltage
        data['power_factor'] = self._power_factor
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ElectricityConnection':
        """Create ElectricityConnection from dictionary."""
        connection = cls(
            _source=data.get('source', ''),
            _target=data.get('target', ''),
            _enabled=data.get('enabled', True),
            _capacity=data.get('capacity'),
            _loss=float(data.get('loss', 0.0)),
            _voltage=data.get('voltage'),
            _power_factor=float(data.get('power_factor', 1.0)),
        )
        return connection

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        cap = f", {self._capacity}kW" if self._capacity else ""
        volt = f", {self._voltage}V" if self._voltage else ""
        return f"ElectricityConnection({self._source}->{self._target}{cap}{volt}, {status})"
