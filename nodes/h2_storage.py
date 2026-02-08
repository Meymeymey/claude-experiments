"""
Hydrogen storage node type.

Represents hydrogen storage systems (tanks, caverns) with injection/withdrawal
rates, compression efficiencies, and loss characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseNode, NodeType, Carrier


@dataclass
class H2Storage(BaseNode):
    """
    Hydrogen storage node.

    Represents a hydrogen storage system (tank or cavern) with configurable
    capacity, injection/withdrawal rates, compression efficiency, and losses.

    Attributes:
        name: Unique identifier for the storage
        capacity: Storage capacity in kg
        injection_rate: Maximum injection rate in kg/h
        withdrawal_rate: Maximum withdrawal rate in kg/h
        efficiency_in: Compression/injection efficiency (0-1)
        efficiency_out: Withdrawal efficiency (0-1)
        loss_rate: Hourly self-discharge/leakage rate (fraction)
        initial_level: Initial fill level (fraction 0-1)
        cost: Cost per kg throughput in ct/kg
    """

    _capacity: float = field(default=50.0)
    _injection_rate: float = field(default=10.0)
    _withdrawal_rate: float = field(default=10.0)
    _efficiency_in: float = field(default=0.95)
    _efficiency_out: float = field(default=0.99)
    _loss_rate: float = field(default=0.0001)
    _initial_level: float = field(default=0.5)
    _cost: float = field(default=0.5)

    # Capacity property
    @property
    def capacity(self) -> float:
        """Get storage capacity in kg."""
        return self._capacity

    @capacity.setter
    def capacity(self, value: float) -> None:
        """Set storage capacity in kg."""
        if value <= 0:
            raise ValueError("Capacity must be positive")
        self._capacity = float(value)

    # Injection rate property
    @property
    def injection_rate(self) -> float:
        """Get maximum injection rate in kg/h."""
        return self._injection_rate

    @injection_rate.setter
    def injection_rate(self, value: float) -> None:
        """Set maximum injection rate in kg/h."""
        if value < 0:
            raise ValueError("Injection rate must be non-negative")
        self._injection_rate = float(value)

    # Withdrawal rate property
    @property
    def withdrawal_rate(self) -> float:
        """Get maximum withdrawal rate in kg/h."""
        return self._withdrawal_rate

    @withdrawal_rate.setter
    def withdrawal_rate(self, value: float) -> None:
        """Set maximum withdrawal rate in kg/h."""
        if value < 0:
            raise ValueError("Withdrawal rate must be non-negative")
        self._withdrawal_rate = float(value)

    # Efficiency in property
    @property
    def efficiency_in(self) -> float:
        """Get injection/compression efficiency (0-1)."""
        return self._efficiency_in

    @efficiency_in.setter
    def efficiency_in(self, value: float) -> None:
        """Set injection/compression efficiency (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        self._efficiency_in = float(value)

    # Efficiency out property
    @property
    def efficiency_out(self) -> float:
        """Get withdrawal efficiency (0-1)."""
        return self._efficiency_out

    @efficiency_out.setter
    def efficiency_out(self, value: float) -> None:
        """Set withdrawal efficiency (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        self._efficiency_out = float(value)

    # Loss rate property
    @property
    def loss_rate(self) -> float:
        """Get hourly leakage rate."""
        return self._loss_rate

    @loss_rate.setter
    def loss_rate(self, value: float) -> None:
        """Set hourly leakage rate."""
        if not 0 <= value <= 1:
            raise ValueError("Loss rate must be between 0 and 1")
        self._loss_rate = float(value)

    # Initial level property
    @property
    def initial_level(self) -> float:
        """Get initial fill level (fraction 0-1)."""
        return self._initial_level

    @initial_level.setter
    def initial_level(self, value: float) -> None:
        """Set initial fill level (fraction 0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Initial level must be between 0 and 1")
        self._initial_level = float(value)

    # Cost property
    @property
    def cost(self) -> float:
        """Get cost per kg throughput in ct/kg."""
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        """Set cost per kg throughput in ct/kg."""
        if value < 0:
            raise ValueError("Cost must be non-negative")
        self._cost = float(value)

    @property
    def node_type(self) -> NodeType:
        """Return the node type."""
        return NodeType.H2_STORAGE

    @property
    def carrier(self) -> Carrier:
        """Return the carrier type (always hydrogen for H2 storage)."""
        return Carrier.HYDROGEN

    @property
    def round_trip_efficiency(self) -> float:
        """Calculate round-trip efficiency."""
        return self._efficiency_in * self._efficiency_out

    @property
    def initial_mass(self) -> float:
        """Calculate initial stored hydrogen in kg."""
        return self._capacity * self._initial_level

    def validate(self) -> bool:
        """Validate H2 storage configuration."""
        if not self._name:
            raise ValueError("H2 storage name is required")
        if self._capacity <= 0:
            raise ValueError("Capacity must be positive")
        if self._injection_rate < 0 or self._withdrawal_rate < 0:
            raise ValueError("Injection/withdrawal rates must be non-negative")
        if not 0 <= self._efficiency_in <= 1:
            raise ValueError("Injection efficiency must be between 0 and 1")
        if not 0 <= self._efficiency_out <= 1:
            raise ValueError("Withdrawal efficiency must be between 0 and 1")
        if not 0 <= self._loss_rate <= 1:
            raise ValueError("Loss rate must be between 0 and 1")
        if not 0 <= self._initial_level <= 1:
            raise ValueError("Initial level must be between 0 and 1")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize H2 storage to dictionary."""
        data = self._base_to_dict()
        data.update({
            'capacity': self._capacity,
            'injection_rate': self._injection_rate,
            'withdrawal_rate': self._withdrawal_rate,
            'efficiency_in': self._efficiency_in,
            'efficiency_out': self._efficiency_out,
            'loss_rate': self._loss_rate,
            'initial_level': self._initial_level,
            'cost': self._cost,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'H2Storage':
        """Create H2Storage from dictionary."""
        storage = cls(
            _name=data.get('name', ''),
            _capacity=float(data.get('capacity', 50.0)),
            _injection_rate=float(data.get('injection_rate', 10.0)),
            _withdrawal_rate=float(data.get('withdrawal_rate', 10.0)),
            _efficiency_in=float(data.get('efficiency_in', 0.95)),
            _efficiency_out=float(data.get('efficiency_out', 0.99)),
            _loss_rate=float(data.get('loss_rate', 0.0001)),
            _initial_level=float(data.get('initial_level', 0.5)),
            _cost=float(data.get('cost', 0.5)),
            _position=tuple(data.get('position', [0, 0, 0])),
            _enabled=data.get('enabled', True),
        )
        return storage

    def calculate_injection_time(self, mass: float) -> float:
        """
        Calculate time to inject a given mass of hydrogen.

        Args:
            mass: Mass to inject in kg

        Returns:
            Time in hours
        """
        if self._injection_rate <= 0:
            return float('inf')
        return mass / (self._injection_rate * self._efficiency_in)

    def calculate_withdrawal_time(self, mass: float) -> float:
        """
        Calculate time to withdraw a given mass of hydrogen.

        Args:
            mass: Mass to withdraw in kg

        Returns:
            Time in hours
        """
        if self._withdrawal_rate <= 0:
            return float('inf')
        return mass / (self._withdrawal_rate * self._efficiency_out)

    def __repr__(self) -> str:
        return (f"H2Storage(name='{self._name}', capacity={self._capacity}kg, "
                f"inject={self._injection_rate}kg/h, withdraw={self._withdrawal_rate}kg/h, "
                f"eff_rt={self.round_trip_efficiency*100:.1f}%)")
