"""
Base connection class for the hydrogen simulation network.

Provides common functionality for all connection types.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum


class ConnectionType(Enum):
    """Types of connections in the network."""
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"


@dataclass
class Connection:
    """
    Base connection class representing an energy flow between nodes.

    Connections represent the flow of energy carriers (electricity or hydrogen)
    between source and target nodes in the network.

    Attributes:
        source: Name of the source node
        target: Name of the target node
        carrier: Type of energy carrier
        enabled: Whether the connection is active
        capacity: Optional capacity limit for the connection
        loss: Optional transmission loss (fraction)
    """

    _source: str = field(default="")
    _target: str = field(default="")
    _carrier: ConnectionType = field(default=ConnectionType.ELECTRICITY)
    _enabled: bool = field(default=True)
    _capacity: Optional[float] = field(default=None)
    _loss: float = field(default=0.0)

    def __post_init__(self):
        """Convert string carrier to enum if needed."""
        if isinstance(self._carrier, str):
            self._carrier = ConnectionType(self._carrier.lower())

    # Source property
    @property
    def source(self) -> str:
        """Get the source node name."""
        return self._source

    @source.setter
    def source(self, value: str) -> None:
        """Set the source node name."""
        if not value or not isinstance(value, str):
            raise ValueError("Source must be a non-empty string")
        self._source = value

    # Target property
    @property
    def target(self) -> str:
        """Get the target node name."""
        return self._target

    @target.setter
    def target(self, value: str) -> None:
        """Set the target node name."""
        if not value or not isinstance(value, str):
            raise ValueError("Target must be a non-empty string")
        self._target = value

    # Carrier property
    @property
    def carrier(self) -> ConnectionType:
        """Get the energy carrier type."""
        return self._carrier

    @carrier.setter
    def carrier(self, value: ConnectionType | str) -> None:
        """Set the energy carrier type."""
        if isinstance(value, str):
            value = ConnectionType(value.lower())
        elif not isinstance(value, ConnectionType):
            raise ValueError(f"Invalid carrier type: {value}")
        self._carrier = value

    # Enabled property
    @property
    def enabled(self) -> bool:
        """Check if the connection is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the connection."""
        self._enabled = bool(value)

    # Capacity property
    @property
    def capacity(self) -> Optional[float]:
        """Get the capacity limit (None if unlimited)."""
        return self._capacity

    @capacity.setter
    def capacity(self, value: Optional[float]) -> None:
        """Set the capacity limit."""
        if value is not None and value < 0:
            raise ValueError("Capacity must be non-negative")
        self._capacity = float(value) if value is not None else None

    # Loss property
    @property
    def loss(self) -> float:
        """Get the transmission loss fraction."""
        return self._loss

    @loss.setter
    def loss(self, value: float) -> None:
        """Set the transmission loss fraction."""
        if not 0 <= value <= 1:
            raise ValueError("Loss must be between 0 and 1")
        self._loss = float(value)

    @property
    def efficiency(self) -> float:
        """Calculate transmission efficiency (1 - loss)."""
        return 1.0 - self._loss

    def validate(self) -> bool:
        """Validate connection configuration."""
        if not self._source:
            raise ValueError("Source node is required")
        if not self._target:
            raise ValueError("Target node is required")
        if self._source == self._target:
            raise ValueError("Source and target cannot be the same")
        if self._capacity is not None and self._capacity < 0:
            raise ValueError("Capacity must be non-negative")
        if not 0 <= self._loss <= 1:
            raise ValueError("Loss must be between 0 and 1")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize connection to dictionary."""
        data = {
            'source': self._source,
            'target': self._target,
            'carrier': self._carrier.value,
            'enabled': self._enabled,
            'loss': self._loss,
        }
        if self._capacity is not None:
            data['capacity'] = self._capacity
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Connection':
        """Create Connection from dictionary."""
        carrier = data.get('carrier', 'electricity')
        if isinstance(carrier, str):
            carrier = ConnectionType(carrier.lower())

        connection = cls(
            _source=data.get('source', ''),
            _target=data.get('target', ''),
            _carrier=carrier,
            _enabled=data.get('enabled', True),
            _capacity=data.get('capacity'),
            _loss=float(data.get('loss', 0.0)),
        )
        return connection

    def get_id(self) -> str:
        """Get unique identifier for this connection."""
        return f"{self._source}->{self._target}:{self._carrier.value}"

    def __repr__(self) -> str:
        status = "enabled" if self._enabled else "disabled"
        cap = f", cap={self._capacity}" if self._capacity else ""
        return f"Connection({self._source}->{self._target}, {self._carrier.value}, {status}{cap})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Connection):
            return False
        return (self._source == other._source and
                self._target == other._target and
                self._carrier == other._carrier)

    def __hash__(self) -> int:
        return hash((self._source, self._target, self._carrier))
