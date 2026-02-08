"""
Base node class for the hydrogen simulation system.

Provides common functionality and interface for all node types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Optional, Tuple


class NodeType(Enum):
    """Enumeration of all node types in the system."""
    GENERATOR = "generator"
    TRANSFORMER = "transformer"
    CONSUMER = "consumer"
    BATTERY = "battery"
    H2_STORAGE = "h2_storage"


class Carrier(Enum):
    """Energy carrier types."""
    ELECTRICITY = "electricity"
    HYDROGEN = "hydrogen"
    BOTH = "both"


@dataclass
class BaseNode(ABC):
    """
    Abstract base class for all nodes in the hydrogen simulation network.

    All node types inherit from this class and must implement:
    - to_dict(): Serialize node configuration to dictionary
    - from_dict(): Create node from dictionary (class method)
    - validate(): Validate node configuration
    """

    _name: str = field(default="")
    _position: Tuple[float, float, float] = field(default=(0.0, 0.0, 0.0))
    _enabled: bool = field(default=True)

    # Property getters and setters for name
    @property
    def name(self) -> str:
        """Get the node name."""
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        """Set the node name."""
        if not value or not isinstance(value, str):
            raise ValueError("Node name must be a non-empty string")
        self._name = value

    # Property getters and setters for position
    @property
    def position(self) -> Tuple[float, float, float]:
        """Get the 3D position for visualization."""
        return self._position

    @position.setter
    def position(self, value: Tuple[float, float, float]) -> None:
        """Set the 3D position for visualization."""
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            raise ValueError("Position must be a tuple/list of 3 floats (x, y, z)")
        self._position = tuple(float(v) for v in value)

    # Property getters and setters for enabled
    @property
    def enabled(self) -> bool:
        """Check if node is enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """Enable or disable the node."""
        self._enabled = bool(value)

    @property
    @abstractmethod
    def node_type(self) -> NodeType:
        """Return the type of this node."""
        pass

    @property
    @abstractmethod
    def carrier(self) -> Carrier:
        """Return the primary carrier type for this node."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node configuration to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseNode':
        """Create node instance from dictionary."""
        pass

    @abstractmethod
    def validate(self) -> bool:
        """
        Validate node configuration.

        Returns:
            True if configuration is valid

        Raises:
            ValueError: If configuration is invalid
        """
        pass

    def _base_to_dict(self) -> Dict[str, Any]:
        """Get base node properties as dictionary."""
        return {
            'name': self._name,
            'position': list(self._position),
            'enabled': self._enabled,
            'node_type': self.node_type.value,
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self._name}')"
