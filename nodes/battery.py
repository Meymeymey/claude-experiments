"""
Battery node type for electricity storage.

Represents battery energy storage systems (BESS) with charge/discharge
rates, efficiencies, and loss characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseNode, NodeType, Carrier


@dataclass
class Battery(BaseNode):
    """
    Electricity storage (battery) node.

    Represents a battery energy storage system with configurable capacity,
    charge/discharge rates, round-trip efficiency, and self-discharge losses.

    Attributes:
        name: Unique identifier for the battery
        capacity: Storage capacity in kWh
        charge_rate: Maximum charging power in kW
        discharge_rate: Maximum discharging power in kW
        efficiency_in: Charging efficiency (0-1)
        efficiency_out: Discharging efficiency (0-1)
        loss_rate: Hourly self-discharge rate (fraction)
        initial_level: Initial state of charge (fraction 0-1)
        min_level: Minimum allowed state of charge (fraction 0-1)
        cost: Cost per kWh throughput in ct/kWh
    """

    _capacity: float = field(default=100.0)
    _charge_rate: float = field(default=50.0)
    _discharge_rate: float = field(default=50.0)
    _efficiency_in: float = field(default=0.95)
    _efficiency_out: float = field(default=0.95)
    _loss_rate: float = field(default=0.0002)
    _initial_level: float = field(default=0.5)
    _min_level: float = field(default=0.1)
    _cost: float = field(default=1.0)

    # Capacity property
    @property
    def capacity(self) -> float:
        """Get storage capacity in kWh."""
        return self._capacity

    @capacity.setter
    def capacity(self, value: float) -> None:
        """Set storage capacity in kWh."""
        if value <= 0:
            raise ValueError("Capacity must be positive")
        self._capacity = float(value)

    # Charge rate property
    @property
    def charge_rate(self) -> float:
        """Get maximum charging power in kW."""
        return self._charge_rate

    @charge_rate.setter
    def charge_rate(self, value: float) -> None:
        """Set maximum charging power in kW."""
        if value < 0:
            raise ValueError("Charge rate must be non-negative")
        self._charge_rate = float(value)

    # Discharge rate property
    @property
    def discharge_rate(self) -> float:
        """Get maximum discharging power in kW."""
        return self._discharge_rate

    @discharge_rate.setter
    def discharge_rate(self, value: float) -> None:
        """Set maximum discharging power in kW."""
        if value < 0:
            raise ValueError("Discharge rate must be non-negative")
        self._discharge_rate = float(value)

    # Efficiency in property
    @property
    def efficiency_in(self) -> float:
        """Get charging efficiency (0-1)."""
        return self._efficiency_in

    @efficiency_in.setter
    def efficiency_in(self, value: float) -> None:
        """Set charging efficiency (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        self._efficiency_in = float(value)

    # Efficiency out property
    @property
    def efficiency_out(self) -> float:
        """Get discharging efficiency (0-1)."""
        return self._efficiency_out

    @efficiency_out.setter
    def efficiency_out(self, value: float) -> None:
        """Set discharging efficiency (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        self._efficiency_out = float(value)

    # Loss rate property
    @property
    def loss_rate(self) -> float:
        """Get hourly self-discharge rate."""
        return self._loss_rate

    @loss_rate.setter
    def loss_rate(self, value: float) -> None:
        """Set hourly self-discharge rate."""
        if not 0 <= value <= 1:
            raise ValueError("Loss rate must be between 0 and 1")
        self._loss_rate = float(value)

    # Initial level property
    @property
    def initial_level(self) -> float:
        """Get initial state of charge (fraction 0-1)."""
        return self._initial_level

    @initial_level.setter
    def initial_level(self, value: float) -> None:
        """Set initial state of charge (fraction 0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Initial level must be between 0 and 1")
        self._initial_level = float(value)

    # Min level property
    @property
    def min_level(self) -> float:
        """Get minimum allowed state of charge (fraction 0-1)."""
        return self._min_level

    @min_level.setter
    def min_level(self, value: float) -> None:
        """Set minimum allowed state of charge (fraction 0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Min level must be between 0 and 1")
        self._min_level = float(value)

    # Cost property
    @property
    def cost(self) -> float:
        """Get cost per kWh throughput in ct/kWh."""
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        """Set cost per kWh throughput in ct/kWh."""
        if value < 0:
            raise ValueError("Cost must be non-negative")
        self._cost = float(value)

    @property
    def node_type(self) -> NodeType:
        """Return the node type."""
        return NodeType.BATTERY

    @property
    def carrier(self) -> Carrier:
        """Return the carrier type (always electricity for batteries)."""
        return Carrier.ELECTRICITY

    @property
    def round_trip_efficiency(self) -> float:
        """Calculate round-trip efficiency."""
        return self._efficiency_in * self._efficiency_out

    @property
    def usable_capacity(self) -> float:
        """Calculate usable capacity considering min level."""
        return self._capacity * (1 - self._min_level)

    @property
    def initial_energy(self) -> float:
        """Calculate initial stored energy in kWh."""
        return self._capacity * self._initial_level

    def validate(self) -> bool:
        """Validate battery configuration."""
        if not self._name:
            raise ValueError("Battery name is required")
        if self._capacity <= 0:
            raise ValueError("Capacity must be positive")
        if self._charge_rate < 0 or self._discharge_rate < 0:
            raise ValueError("Charge/discharge rates must be non-negative")
        if not 0 <= self._efficiency_in <= 1:
            raise ValueError("Charging efficiency must be between 0 and 1")
        if not 0 <= self._efficiency_out <= 1:
            raise ValueError("Discharging efficiency must be between 0 and 1")
        if not 0 <= self._loss_rate <= 1:
            raise ValueError("Loss rate must be between 0 and 1")
        if not 0 <= self._initial_level <= 1:
            raise ValueError("Initial level must be between 0 and 1")
        if not 0 <= self._min_level <= 1:
            raise ValueError("Min level must be between 0 and 1")
        if self._min_level > self._initial_level:
            raise ValueError("Initial level cannot be below min level")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize battery to dictionary."""
        data = self._base_to_dict()
        data.update({
            'capacity': self._capacity,
            'charge_rate': self._charge_rate,
            'discharge_rate': self._discharge_rate,
            'efficiency_in': self._efficiency_in,
            'efficiency_out': self._efficiency_out,
            'loss_rate': self._loss_rate,
            'initial_level': self._initial_level,
            'min_level': self._min_level,
            'cost': self._cost,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Battery':
        """Create Battery from dictionary."""
        battery = cls(
            _name=data.get('name', ''),
            _capacity=float(data.get('capacity', 100.0)),
            _charge_rate=float(data.get('charge_rate', 50.0)),
            _discharge_rate=float(data.get('discharge_rate', 50.0)),
            _efficiency_in=float(data.get('efficiency_in', 0.95)),
            _efficiency_out=float(data.get('efficiency_out', 0.95)),
            _loss_rate=float(data.get('loss_rate', 0.0002)),
            _initial_level=float(data.get('initial_level', 0.5)),
            _min_level=float(data.get('min_level', 0.1)),
            _cost=float(data.get('cost', 1.0)),
            _position=tuple(data.get('position', [0, 0, 0])),
            _enabled=data.get('enabled', True),
        )
        return battery

    def calculate_charge_time(self, energy: float) -> float:
        """
        Calculate time to charge a given amount of energy.

        Args:
            energy: Energy to charge in kWh

        Returns:
            Time in hours
        """
        if self._charge_rate <= 0:
            return float('inf')
        return energy / (self._charge_rate * self._efficiency_in)

    def calculate_discharge_time(self, energy: float) -> float:
        """
        Calculate time to discharge a given amount of energy.

        Args:
            energy: Energy to discharge in kWh

        Returns:
            Time in hours
        """
        if self._discharge_rate <= 0:
            return float('inf')
        return energy / (self._discharge_rate * self._efficiency_out)

    def __repr__(self) -> str:
        return (f"Battery(name='{self._name}', capacity={self._capacity}kWh, "
                f"charge={self._charge_rate}kW, discharge={self._discharge_rate}kW, "
                f"eff_rt={self.round_trip_efficiency*100:.1f}%)")
