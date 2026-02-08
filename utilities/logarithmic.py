"""
Logarithmic utility function implementation.

U(x) = scale * ln(1 + x/shape)

Properties:
- Marginal utility: MU(x) = scale / (shape + x)
- Diminishing marginal utility as x increases
- scale controls overall willingness to pay
- shape controls how fast marginal utility decreases
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import math

from .demand import DemandTranche, generate_tranches


@dataclass
class LogarithmicUtility:
    """
    Logarithmic utility function for single-carrier consumption.

    Models consumer preferences with diminishing marginal utility.
    Used for consumers who only consume electricity OR hydrogen.

    Attributes:
        scale: 'a' parameter - controls max willingness to pay
        shape: 'b' parameter - controls diminishing rate
        max_quantity: Maximum consumption per period
        num_tranches: Number of tranches for LP approximation
    """

    _scale: float = field(default=30.0)
    _shape: float = field(default=5.0)
    _max_quantity: float = field(default=25.0)
    _num_tranches: int = field(default=5)

    # Scale property
    @property
    def scale(self) -> float:
        """Get the scale parameter (a)."""
        return self._scale

    @scale.setter
    def scale(self, value: float) -> None:
        """Set the scale parameter (a)."""
        if value <= 0:
            raise ValueError("Scale must be positive")
        self._scale = float(value)

    # Shape property
    @property
    def shape(self) -> float:
        """Get the shape parameter (b)."""
        return self._shape

    @shape.setter
    def shape(self, value: float) -> None:
        """Set the shape parameter (b)."""
        if value <= 0:
            raise ValueError("Shape must be positive")
        self._shape = float(value)

    # Max quantity property
    @property
    def max_quantity(self) -> float:
        """Get the maximum consumption per period."""
        return self._max_quantity

    @max_quantity.setter
    def max_quantity(self, value: float) -> None:
        """Set the maximum consumption per period."""
        if value <= 0:
            raise ValueError("Max quantity must be positive")
        self._max_quantity = float(value)

    # Num tranches property
    @property
    def num_tranches(self) -> int:
        """Get the number of LP tranches."""
        return self._num_tranches

    @num_tranches.setter
    def num_tranches(self, value: int) -> None:
        """Set the number of LP tranches."""
        if value < 1:
            raise ValueError("Number of tranches must be at least 1")
        self._num_tranches = int(value)

    def utility(self, x: float) -> float:
        """
        Calculate utility at consumption level x.

        U(x) = scale * ln(1 + x/shape)

        Args:
            x: Consumption quantity

        Returns:
            Utility value
        """
        if x < 0:
            return 0
        return self._scale * math.log(1 + x / self._shape)

    def marginal_utility(self, x: float) -> float:
        """
        Calculate marginal utility (willingness to pay) at consumption level x.

        MU(x) = scale / (shape + x)

        Args:
            x: Consumption quantity

        Returns:
            Marginal utility (willingness to pay per unit)
        """
        return self._scale / (self._shape + x)

    def inverse_marginal_utility(self, price: float) -> float:
        """
        Calculate quantity demanded at a given price.

        Inverse of MU(x) = scale / (shape + x)
        x = scale/price - shape

        Args:
            price: Price per unit

        Returns:
            Quantity demanded (capped at max_quantity)
        """
        if price <= 0:
            return self._max_quantity
        x = (self._scale / price) - self._shape
        return max(0, min(x, self._max_quantity))

    def generate_tranches(self) -> List[Tuple[float, float]]:
        """
        Generate piecewise linear tranches to approximate the utility function.

        Returns:
            List of (quantity, willingness_to_pay) tuples
        """
        tranches = []
        tranche_size = self._max_quantity / self._num_tranches

        for i in range(self._num_tranches):
            # Use midpoint of tranche for marginal utility calculation
            midpoint = (i + 0.5) * tranche_size
            wtp = self.marginal_utility(midpoint)
            tranches.append((tranche_size, wtp))

        return tranches

    def generate_demand_tranches(self) -> List[DemandTranche]:
        """
        Generate DemandTranche objects for LP approximation.

        Returns:
            List of DemandTranche objects
        """
        return generate_tranches(
            marginal_utility_fn=self.marginal_utility,
            max_quantity=self._max_quantity,
            num_tranches=self._num_tranches,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'scale': self._scale,
            'shape': self._shape,
            'max_quantity': self._max_quantity,
            'num_tranches': self._num_tranches,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LogarithmicUtility':
        """Create LogarithmicUtility from dictionary."""
        return cls(
            _scale=float(data.get('scale', 30.0)),
            _shape=float(data.get('shape', 5.0)),
            _max_quantity=float(data.get('max_quantity', 25.0)),
            _num_tranches=int(data.get('num_tranches', 5)),
        )

    def __repr__(self) -> str:
        return (f"LogarithmicUtility(scale={self._scale}, shape={self._shape}, "
                f"max_q={self._max_quantity}, tranches={self._num_tranches})")
