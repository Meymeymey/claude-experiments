"""
Demand tranche classes for piecewise linear LP approximation.

Tranches are used to approximate non-linear utility functions
for use in linear programming optimization.
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable


@dataclass
class DemandTranche:
    """
    A demand tranche with quantity and willingness to pay.

    Represents a segment of the piecewise linear demand curve
    used in LP optimization.

    Attributes:
        quantity: Amount of energy per period (kWh or kg)
        price: Willingness to pay per unit (ct/unit)
        tranche_index: Position in the tranche sequence (0-indexed)
    """

    _quantity: float
    _price: float
    _tranche_index: int = 0

    @property
    def quantity(self) -> float:
        """Get the quantity for this tranche."""
        return self._quantity

    @quantity.setter
    def quantity(self, value: float) -> None:
        """Set the quantity for this tranche."""
        if value < 0:
            raise ValueError("Quantity must be non-negative")
        self._quantity = float(value)

    @property
    def price(self) -> float:
        """Get the willingness to pay (price) for this tranche."""
        return self._price

    @price.setter
    def price(self, value: float) -> None:
        """Set the willingness to pay (price) for this tranche."""
        if value < 0:
            raise ValueError("Price must be non-negative")
        self._price = float(value)

    @property
    def tranche_index(self) -> int:
        """Get the tranche index in the sequence."""
        return self._tranche_index

    @property
    def total_value(self) -> float:
        """Calculate total value (utility) of this tranche."""
        return self._quantity * self._price

    def to_dict(self) -> dict:
        """Serialize tranche to dictionary."""
        return {
            'quantity': self._quantity,
            'price': self._price,
            'tranche_index': self._tranche_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'DemandTranche':
        """Create DemandTranche from dictionary."""
        return cls(
            _quantity=float(data.get('quantity', 0)),
            _price=float(data.get('price', 0)),
            _tranche_index=int(data.get('tranche_index', 0)),
        )

    def __repr__(self) -> str:
        return f"DemandTranche(q={self._quantity:.2f}, p={self._price:.2f}ct)"


def generate_tranches(
    marginal_utility_fn: Callable[[float], float],
    max_quantity: float,
    num_tranches: int
) -> List[DemandTranche]:
    """
    Generate piecewise linear tranches from a marginal utility function.

    Creates tranches by sampling the marginal utility function at
    the midpoint of each tranche segment.

    Args:
        marginal_utility_fn: Function that returns marginal utility at quantity x
        max_quantity: Maximum total consumption
        num_tranches: Number of tranches to generate

    Returns:
        List of DemandTranche objects with decreasing prices
    """
    if num_tranches < 1:
        raise ValueError("Number of tranches must be at least 1")
    if max_quantity <= 0:
        raise ValueError("Max quantity must be positive")

    tranches = []
    tranche_size = max_quantity / num_tranches

    for i in range(num_tranches):
        # Use midpoint of tranche for marginal utility calculation
        midpoint = (i + 0.5) * tranche_size
        wtp = marginal_utility_fn(midpoint)

        tranches.append(DemandTranche(
            _quantity=tranche_size,
            _price=wtp,
            _tranche_index=i,
        ))

    return tranches


def tranches_to_tuples(tranches: List[DemandTranche]) -> List[Tuple[float, float]]:
    """Convert list of tranches to list of (quantity, price) tuples."""
    return [(t.quantity, t.price) for t in tranches]


def calculate_total_utility(tranches: List[DemandTranche], consumed: float) -> float:
    """
    Calculate total utility from consuming up to a given quantity.

    Args:
        tranches: List of demand tranches
        consumed: Total quantity consumed

    Returns:
        Total utility value
    """
    total = 0.0
    remaining = consumed

    for tranche in sorted(tranches, key=lambda t: t.tranche_index):
        if remaining <= 0:
            break

        qty = min(remaining, tranche.quantity)
        total += qty * tranche.price
        remaining -= qty

    return total


def calculate_consumer_surplus(
    tranches: List[DemandTranche],
    consumed: float,
    market_price: float
) -> float:
    """
    Calculate consumer surplus at a given market price.

    Consumer surplus = Total utility - Total expenditure

    Args:
        tranches: List of demand tranches
        consumed: Total quantity consumed
        market_price: Market price per unit

    Returns:
        Consumer surplus value
    """
    total_utility = calculate_total_utility(tranches, consumed)
    total_expenditure = consumed * market_price
    return total_utility - total_expenditure
