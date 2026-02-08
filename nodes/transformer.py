"""
Transformer node type for energy conversion.

Primarily used for electrolyzers that convert electricity to hydrogen,
but supports any input/output carrier combination.
"""

from dataclasses import dataclass, field
from typing import Dict, Any

from .base import BaseNode, NodeType, Carrier


@dataclass
class Transformer(BaseNode):
    """
    Energy transformer/converter node.

    Represents energy conversion equipment like electrolyzers that convert
    one energy carrier to another with a given efficiency and cost.

    Attributes:
        name: Unique identifier for the transformer
        capacity: Maximum output power in kW (or kg/h for hydrogen)
        efficiency: Conversion efficiency (0-1)
        cost: Transformation cost in ct/kWh of input
        input_carrier: Input energy type ('electricity')
        output_carrier: Output energy type ('hydrogen')
        conversion_delay: Delay in time periods between input and output (0=instant)
    """

    _capacity: float = field(default=50.0)
    _efficiency: float = field(default=0.7)
    _cost: float = field(default=1.5)
    _input_carrier: Carrier = field(default=Carrier.ELECTRICITY)
    _output_carrier: Carrier = field(default=Carrier.HYDROGEN)
    _conversion_delay: int = field(default=0)

    def __post_init__(self):
        """Convert string carriers to enum if needed."""
        if isinstance(self._input_carrier, str):
            self._input_carrier = Carrier(self._input_carrier.lower())
        if isinstance(self._output_carrier, str):
            self._output_carrier = Carrier(self._output_carrier.lower())

    # Capacity property
    @property
    def capacity(self) -> float:
        """Get maximum output capacity."""
        return self._capacity

    @capacity.setter
    def capacity(self, value: float) -> None:
        """Set maximum output capacity."""
        if value < 0:
            raise ValueError("Capacity must be non-negative")
        self._capacity = float(value)

    # Efficiency property
    @property
    def efficiency(self) -> float:
        """Get conversion efficiency (0-1)."""
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value: float) -> None:
        """Set conversion efficiency (0-1)."""
        if not 0 <= value <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        self._efficiency = float(value)

    # Cost property
    @property
    def cost(self) -> float:
        """Get transformation cost in ct/kWh input."""
        return self._cost

    @cost.setter
    def cost(self, value: float) -> None:
        """Set transformation cost in ct/kWh input."""
        if value < 0:
            raise ValueError("Cost must be non-negative")
        self._cost = float(value)

    # Input carrier property
    @property
    def input_carrier(self) -> Carrier:
        """Get the input energy carrier type."""
        return self._input_carrier

    @input_carrier.setter
    def input_carrier(self, value: Carrier | str) -> None:
        """Set the input energy carrier type."""
        if isinstance(value, str):
            value = Carrier(value.lower())
        elif not isinstance(value, Carrier):
            raise ValueError(f"Invalid carrier type: {value}")
        self._input_carrier = value

    # Output carrier property
    @property
    def output_carrier(self) -> Carrier:
        """Get the output energy carrier type."""
        return self._output_carrier

    @output_carrier.setter
    def output_carrier(self, value: Carrier | str) -> None:
        """Set the output energy carrier type."""
        if isinstance(value, str):
            value = Carrier(value.lower())
        elif not isinstance(value, Carrier):
            raise ValueError(f"Invalid carrier type: {value}")
        self._output_carrier = value

    # Conversion delay property
    @property
    def conversion_delay(self) -> int:
        """Get conversion delay in time periods (hours)."""
        return self._conversion_delay

    @conversion_delay.setter
    def conversion_delay(self, value: int) -> None:
        """Set conversion delay in time periods (hours)."""
        if not isinstance(value, int) or value < 0:
            raise ValueError("Conversion delay must be a non-negative integer")
        self._conversion_delay = value

    @property
    def node_type(self) -> NodeType:
        """Return the node type."""
        return NodeType.TRANSFORMER

    @property
    def carrier(self) -> Carrier:
        """Return the primary carrier (output carrier for transformers)."""
        return self._output_carrier

    def validate(self) -> bool:
        """Validate transformer configuration."""
        if not self._name:
            raise ValueError("Transformer name is required")
        if self._capacity < 0:
            raise ValueError("Capacity must be non-negative")
        if not 0 <= self._efficiency <= 1:
            raise ValueError("Efficiency must be between 0 and 1")
        if self._cost < 0:
            raise ValueError("Cost must be non-negative")
        if self._input_carrier == self._output_carrier:
            raise ValueError("Input and output carriers must be different")
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serialize transformer to dictionary."""
        data = self._base_to_dict()
        data.update({
            'capacity': self._capacity,
            'efficiency': self._efficiency,
            'cost': self._cost,
            'input_carrier': self._input_carrier.value,
            'output_carrier': self._output_carrier.value,
            'conversion_delay': self._conversion_delay,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Transformer':
        """Create Transformer from dictionary."""
        input_carrier = data.get('input_carrier', 'electricity')
        output_carrier = data.get('output_carrier', 'hydrogen')

        if isinstance(input_carrier, str):
            input_carrier = Carrier(input_carrier.lower())
        if isinstance(output_carrier, str):
            output_carrier = Carrier(output_carrier.lower())

        transformer = cls(
            _name=data.get('name', ''),
            _capacity=float(data.get('capacity', 50.0)),
            _efficiency=float(data.get('efficiency', 0.7)),
            _cost=float(data.get('cost', 1.5)),
            _input_carrier=input_carrier,
            _output_carrier=output_carrier,
            _conversion_delay=int(data.get('conversion_delay', 0)),
            _position=tuple(data.get('position', [0, 0, 0])),
            _enabled=data.get('enabled', True),
        )
        return transformer

    def calculate_output(self, input_power: float) -> float:
        """
        Calculate output power given input power.

        Args:
            input_power: Input power in kW

        Returns:
            Output power after efficiency losses
        """
        return input_power * self._efficiency

    def calculate_required_input(self, desired_output: float) -> float:
        """
        Calculate required input power for desired output.

        Args:
            desired_output: Desired output power

        Returns:
            Required input power
        """
        if self._efficiency <= 0:
            return float('inf')
        return desired_output / self._efficiency

    def get_output_availability_profile(self, periods: int = 24) -> list:
        """
        Get output availability profile considering conversion delay.

        Returns a profile where output is zero for the first 'delay' periods,
        then 1.0 for the remaining periods.

        Args:
            periods: Number of time periods

        Returns:
            List of availability factors (0 or 1) for each period
        """
        if self._conversion_delay <= 0:
            return [1.0] * periods

        delay = min(self._conversion_delay, periods)
        return [0.0] * delay + [1.0] * (periods - delay)

    def __repr__(self) -> str:
        return (f"Transformer(name='{self._name}', capacity={self._capacity}kW, "
                f"efficiency={self._efficiency*100:.0f}%, cost={self._cost}ct/kWh, "
                f"{self._input_carrier.value}->{self._output_carrier.value})")
