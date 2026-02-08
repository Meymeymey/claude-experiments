"""
Consumer node type for energy demand.

Supports different utility function types:
- Logarithmic: Single-carrier with diminishing marginal utility
- Cobb-Douglas: Dual-carrier with complementary consumption
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from enum import Enum
import math

from .base import BaseNode, NodeType, Carrier


class ConsumerType(Enum):
    """Types of consumer utility functions."""
    LOGARITHMIC_ELECTRICITY = "logarithmic_electricity"
    LOGARITHMIC_HYDROGEN = "logarithmic_hydrogen"
    COBB_DOUGLAS = "cobb_douglas"


@dataclass
class Consumer(BaseNode):
    """
    Energy consumer node with utility-based demand.

    Consumers have utility functions that determine their willingness to pay
    for energy. The optimizer maximizes total welfare by balancing consumer
    utility against production costs.

    Attributes:
        name: Unique identifier for the consumer
        consumer_type: Type of utility function
        carrier: Primary energy carrier consumed

    Logarithmic utility params (single-carrier):
        log_scale: Controls maximum willingness to pay
        log_shape: Controls diminishing rate
        log_max_quantity: Maximum consumption per period
        log_num_tranches: Number of LP approximation tranches

    Cobb-Douglas params (dual-carrier):
        cd_A: Scale parameter
        cd_alpha: Hydrogen exponent
        cd_beta: Electricity exponent
        cd_max_bundles: Maximum bundles per period
        cd_h_per_bundle: kg hydrogen per bundle
        cd_e_per_bundle: kWh electricity per bundle
        cd_num_tranches: Number of LP approximation tranches

    Temporal params (peak/off-peak):
        peak_hours: List of peak demand hours (0-23)
        peak_multiplier: Consumption multiplier for peak hours
        offpeak_multiplier: Consumption multiplier for off-peak hours
    """

    _consumer_type: ConsumerType = field(default=ConsumerType.LOGARITHMIC_HYDROGEN)
    _carrier: Carrier = field(default=Carrier.HYDROGEN)

    # Logarithmic utility parameters
    _log_scale: float = field(default=30.0)
    _log_shape: float = field(default=5.0)
    _log_max_quantity: float = field(default=25.0)
    _log_num_tranches: int = field(default=5)

    # Cobb-Douglas utility parameters
    _cd_A: float = field(default=15.0)
    _cd_alpha: float = field(default=0.4)
    _cd_beta: float = field(default=0.6)
    _cd_max_bundles: float = field(default=15.0)
    _cd_h_per_bundle: float = field(default=1.0)
    _cd_e_per_bundle: float = field(default=3.0)
    _cd_num_tranches: int = field(default=5)

    # Peak/off-peak temporal parameters
    _peak_hours: Optional[List[int]] = field(default=None)
    _peak_multiplier: float = field(default=1.0)
    _offpeak_multiplier: float = field(default=1.0)

    def __post_init__(self):
        """Convert string types to enum if needed."""
        if isinstance(self._consumer_type, str):
            self._consumer_type = ConsumerType(self._consumer_type.lower())
        if isinstance(self._carrier, str):
            self._carrier = Carrier(self._carrier.lower())

    # Consumer type property
    @property
    def consumer_type(self) -> ConsumerType:
        """Get the consumer utility function type."""
        return self._consumer_type

    @consumer_type.setter
    def consumer_type(self, value: ConsumerType | str) -> None:
        """Set the consumer utility function type."""
        if isinstance(value, str):
            value = ConsumerType(value.lower())
        self._consumer_type = value

    # Carrier property (override base)
    @property
    def carrier(self) -> Carrier:
        """Get the primary energy carrier consumed."""
        return self._carrier

    @carrier.setter
    def carrier(self, value: Carrier | str) -> None:
        """Set the primary energy carrier consumed."""
        if isinstance(value, str):
            value = Carrier(value.lower())
        self._carrier = value

    # Logarithmic utility getters/setters
    @property
    def log_scale(self) -> float:
        """Get logarithmic scale parameter."""
        return self._log_scale

    @log_scale.setter
    def log_scale(self, value: float) -> None:
        """Set logarithmic scale parameter."""
        if value <= 0:
            raise ValueError("Log scale must be positive")
        self._log_scale = float(value)

    @property
    def log_shape(self) -> float:
        """Get logarithmic shape parameter."""
        return self._log_shape

    @log_shape.setter
    def log_shape(self, value: float) -> None:
        """Set logarithmic shape parameter."""
        if value <= 0:
            raise ValueError("Log shape must be positive")
        self._log_shape = float(value)

    @property
    def log_max_quantity(self) -> float:
        """Get maximum consumption quantity."""
        return self._log_max_quantity

    @log_max_quantity.setter
    def log_max_quantity(self, value: float) -> None:
        """Set maximum consumption quantity."""
        if value <= 0:
            raise ValueError("Max quantity must be positive")
        self._log_max_quantity = float(value)

    @property
    def log_num_tranches(self) -> int:
        """Get number of LP tranches."""
        return self._log_num_tranches

    @log_num_tranches.setter
    def log_num_tranches(self, value: int) -> None:
        """Set number of LP tranches."""
        if value < 1:
            raise ValueError("Number of tranches must be at least 1")
        self._log_num_tranches = int(value)

    # Cobb-Douglas utility getters/setters
    @property
    def cd_A(self) -> float:
        """Get Cobb-Douglas scale parameter."""
        return self._cd_A

    @cd_A.setter
    def cd_A(self, value: float) -> None:
        """Set Cobb-Douglas scale parameter."""
        if value <= 0:
            raise ValueError("CD scale (A) must be positive")
        self._cd_A = float(value)

    @property
    def cd_alpha(self) -> float:
        """Get Cobb-Douglas hydrogen exponent."""
        return self._cd_alpha

    @cd_alpha.setter
    def cd_alpha(self, value: float) -> None:
        """Set Cobb-Douglas hydrogen exponent."""
        if not 0 < value < 1:
            raise ValueError("CD alpha must be between 0 and 1")
        self._cd_alpha = float(value)

    @property
    def cd_beta(self) -> float:
        """Get Cobb-Douglas electricity exponent."""
        return self._cd_beta

    @cd_beta.setter
    def cd_beta(self, value: float) -> None:
        """Set Cobb-Douglas electricity exponent."""
        if not 0 < value < 1:
            raise ValueError("CD beta must be between 0 and 1")
        self._cd_beta = float(value)

    @property
    def cd_max_bundles(self) -> float:
        """Get maximum bundles per period."""
        return self._cd_max_bundles

    @cd_max_bundles.setter
    def cd_max_bundles(self, value: float) -> None:
        """Set maximum bundles per period."""
        if value <= 0:
            raise ValueError("Max bundles must be positive")
        self._cd_max_bundles = float(value)

    @property
    def cd_h_per_bundle(self) -> float:
        """Get kg hydrogen per bundle."""
        return self._cd_h_per_bundle

    @cd_h_per_bundle.setter
    def cd_h_per_bundle(self, value: float) -> None:
        """Set kg hydrogen per bundle."""
        if value <= 0:
            raise ValueError("Hydrogen per bundle must be positive")
        self._cd_h_per_bundle = float(value)

    @property
    def cd_e_per_bundle(self) -> float:
        """Get kWh electricity per bundle."""
        return self._cd_e_per_bundle

    @cd_e_per_bundle.setter
    def cd_e_per_bundle(self, value: float) -> None:
        """Set kWh electricity per bundle."""
        if value <= 0:
            raise ValueError("Electricity per bundle must be positive")
        self._cd_e_per_bundle = float(value)

    @property
    def cd_num_tranches(self) -> int:
        """Get number of CD LP tranches."""
        return self._cd_num_tranches

    @cd_num_tranches.setter
    def cd_num_tranches(self, value: int) -> None:
        """Set number of CD LP tranches."""
        if value < 1:
            raise ValueError("Number of tranches must be at least 1")
        self._cd_num_tranches = int(value)

    # Peak/off-peak temporal properties
    @property
    def peak_hours(self) -> Optional[List[int]]:
        """Get list of peak demand hours (0-23)."""
        return self._peak_hours

    @peak_hours.setter
    def peak_hours(self, value: Optional[List[int]]) -> None:
        """Set list of peak demand hours."""
        if value is not None:
            if not isinstance(value, list):
                raise ValueError("Peak hours must be a list of integers")
            if not all(isinstance(h, int) and 0 <= h <= 23 for h in value):
                raise ValueError("Peak hours must be integers 0-23")
            value = sorted(set(value))  # Remove duplicates, sort
        self._peak_hours = value

    @property
    def peak_multiplier(self) -> float:
        """Get consumption multiplier for peak hours."""
        return self._peak_multiplier

    @peak_multiplier.setter
    def peak_multiplier(self, value: float) -> None:
        """Set consumption multiplier for peak hours."""
        if value < 0:
            raise ValueError("Peak multiplier must be non-negative")
        self._peak_multiplier = float(value)

    @property
    def offpeak_multiplier(self) -> float:
        """Get consumption multiplier for off-peak hours."""
        return self._offpeak_multiplier

    @offpeak_multiplier.setter
    def offpeak_multiplier(self, value: float) -> None:
        """Set consumption multiplier for off-peak hours."""
        if value < 0:
            raise ValueError("Off-peak multiplier must be non-negative")
        self._offpeak_multiplier = float(value)

    @property
    def node_type(self) -> NodeType:
        """Return the node type."""
        return NodeType.CONSUMER

    @property
    def is_dual_carrier(self) -> bool:
        """Check if consumer uses both electricity and hydrogen."""
        return self._consumer_type == ConsumerType.COBB_DOUGLAS

    def validate(self) -> bool:
        """Validate consumer configuration."""
        if not self._name:
            raise ValueError("Consumer name is required")

        if self._consumer_type in (ConsumerType.LOGARITHMIC_ELECTRICITY,
                                   ConsumerType.LOGARITHMIC_HYDROGEN):
            if self._log_scale <= 0:
                raise ValueError("Log scale must be positive")
            if self._log_shape <= 0:
                raise ValueError("Log shape must be positive")
            if self._log_max_quantity <= 0:
                raise ValueError("Max quantity must be positive")

        elif self._consumer_type == ConsumerType.COBB_DOUGLAS:
            if self._cd_A <= 0:
                raise ValueError("CD scale (A) must be positive")
            if not 0 < self._cd_alpha < 1:
                raise ValueError("CD alpha must be between 0 and 1")
            if not 0 < self._cd_beta < 1:
                raise ValueError("CD beta must be between 0 and 1")

        return True

    # Utility calculation methods
    def logarithmic_utility(self, x: float) -> float:
        """Calculate logarithmic utility at consumption level x."""
        return self._log_scale * math.log(1 + x / self._log_shape)

    def logarithmic_marginal_utility(self, x: float) -> float:
        """Calculate logarithmic marginal utility (willingness to pay)."""
        return self._log_scale / (self._log_shape + x)

    def cobb_douglas_utility(self, h: float, e: float) -> float:
        """Calculate Cobb-Douglas utility at consumption levels (h, e)."""
        if h <= 0 or e <= 0:
            return 0
        return self._cd_A * (h ** self._cd_alpha) * (e ** self._cd_beta)

    def cobb_douglas_bundle_utility(self, n_bundles: float) -> float:
        """Calculate utility from n bundles."""
        h = n_bundles * self._cd_h_per_bundle
        e = n_bundles * self._cd_e_per_bundle
        return self.cobb_douglas_utility(h, e)

    def cobb_douglas_marginal_bundle_utility(self, n_bundles: float) -> float:
        """Calculate marginal utility of an additional bundle."""
        if n_bundles <= 0:
            n_bundles = 0.1
        h = n_bundles * self._cd_h_per_bundle
        e = n_bundles * self._cd_e_per_bundle
        dU_dh = self._cd_A * self._cd_alpha * (h ** (self._cd_alpha - 1)) * (e ** self._cd_beta)
        dU_de = self._cd_A * self._cd_beta * (h ** self._cd_alpha) * (e ** (self._cd_beta - 1))
        return dU_dh * self._cd_h_per_bundle + dU_de * self._cd_e_per_bundle

    def generate_tranches(self) -> List[Tuple[float, float]]:
        """
        Generate piecewise linear tranches to approximate the utility function.

        Returns:
            List of (quantity, willingness_to_pay) tuples
        """
        if self._consumer_type in (ConsumerType.LOGARITHMIC_ELECTRICITY,
                                   ConsumerType.LOGARITHMIC_HYDROGEN):
            tranches = []
            tranche_size = self._log_max_quantity / self._log_num_tranches
            for i in range(self._log_num_tranches):
                midpoint = (i + 0.5) * tranche_size
                wtp = self.logarithmic_marginal_utility(midpoint)
                tranches.append((tranche_size, wtp))
            return tranches

        elif self._consumer_type == ConsumerType.COBB_DOUGLAS:
            tranches = []
            tranche_size = self._cd_max_bundles / self._cd_num_tranches
            for i in range(self._cd_num_tranches):
                midpoint = (i + 0.5) * tranche_size
                wtp = self.cobb_douglas_marginal_bundle_utility(midpoint)
                tranches.append((tranche_size, wtp))
            return tranches

        return []

    def get_consumption_multipliers(self, periods: int = 24) -> List[float]:
        """
        Get consumption capacity multipliers for each time period.

        Returns list of multipliers that scale the max consumption per period.
        If peak_hours is None, returns [1.0] * periods (no time variation).

        Args:
            periods: Number of time periods

        Returns:
            List of multipliers for each period
        """
        if self._peak_hours is None:
            return [1.0] * periods

        multipliers = []
        peak_set = set(self._peak_hours)

        for t in range(periods):
            hour = t % 24
            if hour in peak_set:
                multipliers.append(self._peak_multiplier)
            else:
                multipliers.append(self._offpeak_multiplier)

        return multipliers

    def to_dict(self) -> Dict[str, Any]:
        """Serialize consumer to dictionary."""
        data = self._base_to_dict()
        data.update({
            'consumer_type': self._consumer_type.value,
            'carrier': self._carrier.value,
            # Logarithmic params
            'log_scale': self._log_scale,
            'log_shape': self._log_shape,
            'log_max_quantity': self._log_max_quantity,
            'log_num_tranches': self._log_num_tranches,
            # Cobb-Douglas params
            'cd_A': self._cd_A,
            'cd_alpha': self._cd_alpha,
            'cd_beta': self._cd_beta,
            'cd_max_bundles': self._cd_max_bundles,
            'cd_h_per_bundle': self._cd_h_per_bundle,
            'cd_e_per_bundle': self._cd_e_per_bundle,
            'cd_num_tranches': self._cd_num_tranches,
            # Peak/off-peak temporal params
            'peak_hours': self._peak_hours,
            'peak_multiplier': self._peak_multiplier,
            'offpeak_multiplier': self._offpeak_multiplier,
        })
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Consumer':
        """Create Consumer from dictionary."""
        consumer_type = data.get('consumer_type', 'logarithmic_hydrogen')
        carrier = data.get('carrier', 'hydrogen')

        if isinstance(consumer_type, str):
            consumer_type = ConsumerType(consumer_type.lower())
        if isinstance(carrier, str):
            carrier = Carrier(carrier.lower())

        consumer = cls(
            _name=data.get('name', ''),
            _consumer_type=consumer_type,
            _carrier=carrier,
            # Logarithmic params
            _log_scale=float(data.get('log_scale', 30.0)),
            _log_shape=float(data.get('log_shape', 5.0)),
            _log_max_quantity=float(data.get('log_max_quantity', 25.0)),
            _log_num_tranches=int(data.get('log_num_tranches', 5)),
            # Cobb-Douglas params
            _cd_A=float(data.get('cd_A', 15.0)),
            _cd_alpha=float(data.get('cd_alpha', 0.4)),
            _cd_beta=float(data.get('cd_beta', 0.6)),
            _cd_max_bundles=float(data.get('cd_max_bundles', 15.0)),
            _cd_h_per_bundle=float(data.get('cd_h_per_bundle', 1.0)),
            _cd_e_per_bundle=float(data.get('cd_e_per_bundle', 3.0)),
            _cd_num_tranches=int(data.get('cd_num_tranches', 5)),
            # Peak/off-peak temporal params
            _peak_hours=data.get('peak_hours'),
            _peak_multiplier=float(data.get('peak_multiplier', 1.0)),
            _offpeak_multiplier=float(data.get('offpeak_multiplier', 1.0)),
            # Base params
            _position=tuple(data.get('position', [0, 0, 0])),
            _enabled=data.get('enabled', True),
        )
        return consumer

    def __repr__(self) -> str:
        if self._consumer_type == ConsumerType.COBB_DOUGLAS:
            return (f"Consumer(name='{self._name}', type=cobb_douglas, "
                    f"A={self._cd_A}, alpha={self._cd_alpha}, beta={self._cd_beta})")
        else:
            return (f"Consumer(name='{self._name}', type={self._consumer_type.value}, "
                    f"scale={self._log_scale}, shape={self._log_shape})")
