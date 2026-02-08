"""
Generator node type for electricity production.

Supports different generation profiles:
- solar: Variable output following solar patterns
- wind: Variable output following wind patterns
- flat: Constant output at rated capacity
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from enum import Enum

from .base import BaseNode, NodeType, Carrier


class ProfileType(Enum):
    """Generator profile types."""
    SOLAR = "solar"
    WIND = "wind"
    FLAT = "flat"


@dataclass
class Generator(BaseNode):
    """
    Electricity generator node.

    Represents a power source in the network that produces electricity
    at a given cost with a capacity limit. The output profile can vary
    based on the profile_type (solar, wind, or flat).

    Attributes:
        name: Unique identifier for the generator
        capacity: Maximum output power in kW
        cost: Production cost in ct/kWh
        profile_type: Output profile ('solar', 'wind', 'flat')
        start_hour: Hour when production can start (0-23, None=no restriction)
        end_hour: Hour when production must stop (0-23, None=no restriction)
        custom_profile: Custom 24-hour availability profile (0-1 values)
    """

    _capacity: float = field(default=100.0)
    _cost: float = field(default=8.0)
    _profile_type: ProfileType = field(default=ProfileType.SOLAR)
    _start_hour: Optional[int] = field(default=None)
    _end_hour: Optional[int] = field(default=None)
    _custom_profile: Optional[List[float]] = field(default=None)

    def __post_init__(self):
        """Convert string profile_type to enum if needed."""
        if isinstance(self._profile_type, str):
            self._profile_type = ProfileType(self._profile_type.lower())

    # Capacity property
    @property
    def capacity(self) -> float:
        """Get maximum output capacity in kW."""
        return self._capacity

    @capacity.setter
    def capacity(self, value: float) -> None:
        """Set maximum output capacity in kW."""
        if value < 0:
            raise ValueError("Capacity must be non-negative")
        self._capacity = float(value)

    # Cost property
    @property
    def cost(self) -> float:
        """Get production cost in ct/kWh."""
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        """Set production cost in ct/kWh."""
        if value < 0:
            raise ValueError("Cost must be non-negative")
        self._cost = float(value)

    # Profile type property
    @property
    def profile_type(self) -> ProfileType:
        """Get the output profile type."""
        return self._profile_type

    @profile_type.setter
    def profile_type(self, value: ProfileType | str) -> None:
        """Set the output profile type."""
        if isinstance(value, str):
            value = ProfileType(value.lower())
        elif not isinstance(value, ProfileType):
            raise ValueError(f"Invalid profile type: {value}")
        self._profile_type = value

    # Start hour property
    @property
    def start_hour(self) -> Optional[int]:
        """Get the start hour for production window (0-23, None=no restriction)."""
        return self._start_hour

    @start_hour.setter
    def start_hour(self, value: Optional[int]) -> None:
        """Set the start hour for production window."""
        if value is not None:
            if not isinstance(value, int) or not 0 <= value <= 23:
                raise ValueError("Start hour must be an integer 0-23 or None")
        self._start_hour = value

    # End hour property
    @property
    def end_hour(self) -> Optional[int]:
        """Get the end hour for production window (0-23, None=no restriction)."""
        return self._end_hour

    @end_hour.setter
    def end_hour(self, value: Optional[int]) -> None:
        """Set the end hour for production window."""
        if value is not None:
            if not isinstance(value, int) or not 0 <= value <= 23:
                raise ValueError("End hour must be an integer 0-23 or None")
        self._end_hour = value

    # Custom profile property
    @property
    def custom_profile(self) -> Optional[List[float]]:
        """Get custom hourly availability profile (list of 24 floats 0-1)."""
        return self._custom_profile

    @custom_profile.setter
    def custom_profile(self, value: Optional[List[float]]) -> None:
        """Set custom hourly availability profile."""
        if value is not None:
            if not isinstance(value, list) or len(value) != 24:
                raise ValueError("Custom profile must be a list of 24 floats")
            if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in value):
                raise ValueError("Custom profile values must be floats between 0 and 1")
            value = [float(v) for v in value]
        self._custom_profile = value

    @property
    def node_type(self) -> NodeType:
        """Return the node type."""
        return NodeType.GENERATOR

    @property
    def carrier(self) -> Carrier:
        """Return the carrier type (always electricity for generators)."""
        return Carrier.ELECTRICITY

    def validate(self) -> bool:
        """Validate generator configuration."""
        if not self._name:
            raise ValueError("Generator name is required")
        if self._capacity < 0:
            raise ValueError("Capacity must be non-negative")
        if self._cost < 0:
            raise ValueError("Cost must be non-negative")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize generator to dictionary."""
        data = self._base_to_dict()
        data.update({
            'capacity': self._capacity,
            'cost': self._cost,
            'profile_type': self._profile_type.value,
            'start_hour': self._start_hour,
            'end_hour': self._end_hour,
            'custom_profile': self._custom_profile,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Generator':
        """Create Generator from dictionary."""
        profile = data.get('profile_type', 'solar')
        if isinstance(profile, str):
            profile = ProfileType(profile.lower())

        generator = cls(
            _name=data.get('name', ''),
            _capacity=float(data.get('capacity', 100.0)),
            _cost=float(data.get('cost', 8.0)),
            _profile_type=profile,
            _start_hour=data.get('start_hour'),
            _end_hour=data.get('end_hour'),
            _custom_profile=data.get('custom_profile'),
            _position=tuple(data.get('position', [0, 0, 0])),
            _enabled=data.get('enabled', True),
        )
        return generator

    def get_profile_factors(self, periods: int = 24) -> List[float]:
        """
        Generate availability profile factors for each time period.

        Priority:
        1. If custom_profile is set, use it (extended/repeated for periods)
        2. Otherwise use profile_type (solar/wind/flat)
        3. Apply time window restrictions (start_hour/end_hour) as mask

        Args:
            periods: Number of time periods (default 24 for hourly)

        Returns:
            List of capacity factors between 0 and 1
        """
        import math

        # Step 1: Get base profile
        if self._custom_profile is not None:
            # Use custom profile, extend/repeat for multi-day simulations
            base_factors = []
            for t in range(periods):
                hour = t % 24
                base_factors.append(self._custom_profile[hour])

        elif self._profile_type == ProfileType.FLAT:
            base_factors = [1.0] * periods

        elif self._profile_type == ProfileType.SOLAR:
            # Solar profile: peaks at midday (hour 12)
            base_factors = []
            for t in range(periods):
                hour = t % 24
                if 6 <= hour <= 18:
                    # Sinusoidal pattern for daylight hours
                    factor = math.sin(math.pi * (hour - 6) / 12)
                else:
                    factor = 0.0
                base_factors.append(factor)

        elif self._profile_type == ProfileType.WIND:
            # Wind profile: variable with some pattern
            base_factors = []
            for t in range(periods):
                hour = t % 24
                # Base wind + daily variation
                factor = 0.4 + 0.3 * math.sin(math.pi * hour / 12) + 0.2 * math.sin(math.pi * hour / 6)
                factor = max(0.1, min(1.0, factor))
                base_factors.append(factor)

        else:
            base_factors = [1.0] * periods

        # Step 2: Apply time window mask
        if self._start_hour is not None or self._end_hour is not None:
            start = self._start_hour if self._start_hour is not None else 0
            end = self._end_hour if self._end_hour is not None else 23

            for t in range(periods):
                hour = t % 24
                if start <= end:
                    # Normal window (e.g., 8-18)
                    if not (start <= hour <= end):
                        base_factors[t] = 0.0
                else:
                    # Overnight window (e.g., 22-6)
                    if not (hour >= start or hour <= end):
                        base_factors[t] = 0.0

        return base_factors

    def __repr__(self) -> str:
        return f"Generator(name='{self._name}', capacity={self._capacity}kW, cost={self._cost}ct/kWh, profile={self._profile_type.value})"
