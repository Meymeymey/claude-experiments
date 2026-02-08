"""
Cobb-Douglas utility function implementation.

U(h, e) = A * h^α * e^β

Properties:
- A: scale parameter (overall utility multiplier)
- alpha: exponent for hydrogen (h)
- beta: exponent for electricity (e)
- The optimal ratio h:e = α:β at any price ratio
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any
import math

from .demand import DemandTranche, generate_tranches


@dataclass
class CobbDouglasUtility:
    """
    Cobb-Douglas utility function for dual-carrier consumption.

    Models consumer preferences where both hydrogen and electricity
    are consumed together in complementary fashion.

    For LP approximation, consumption is modeled as "bundles" containing
    fixed ratios of hydrogen and electricity.

    Attributes:
        A: Scale parameter (overall utility multiplier)
        alpha: Hydrogen exponent (share of "budget")
        beta: Electricity exponent (share of "budget")
        max_bundles: Maximum number of bundles per period
        h_per_bundle: kg hydrogen per bundle
        e_per_bundle: kWh electricity per bundle
        num_tranches: Number of tranches for LP approximation
    """

    _A: float = field(default=10.0)
    _alpha: float = field(default=0.4)
    _beta: float = field(default=0.6)
    _max_bundles: float = field(default=20.0)
    _h_per_bundle: float = field(default=1.0)
    _e_per_bundle: float = field(default=3.0)
    _num_tranches: int = field(default=5)

    # A (scale) property
    @property
    def A(self) -> float:
        """Get the scale parameter."""
        return self._A

    @A.setter
    def A(self, value: float) -> None:
        """Set the scale parameter."""
        if value <= 0:
            raise ValueError("Scale (A) must be positive")
        self._A = float(value)

    # Alpha property
    @property
    def alpha(self) -> float:
        """Get the hydrogen exponent."""
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set the hydrogen exponent."""
        if not 0 < value < 1:
            raise ValueError("Alpha must be between 0 and 1 (exclusive)")
        self._alpha = float(value)

    # Beta property
    @property
    def beta(self) -> float:
        """Get the electricity exponent."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        """Set the electricity exponent."""
        if not 0 < value < 1:
            raise ValueError("Beta must be between 0 and 1 (exclusive)")
        self._beta = float(value)

    # Max bundles property
    @property
    def max_bundles(self) -> float:
        """Get the maximum bundles per period."""
        return self._max_bundles

    @max_bundles.setter
    def max_bundles(self, value: float) -> None:
        """Set the maximum bundles per period."""
        if value <= 0:
            raise ValueError("Max bundles must be positive")
        self._max_bundles = float(value)

    # H per bundle property
    @property
    def h_per_bundle(self) -> float:
        """Get kg hydrogen per bundle."""
        return self._h_per_bundle

    @h_per_bundle.setter
    def h_per_bundle(self, value: float) -> None:
        """Set kg hydrogen per bundle."""
        if value <= 0:
            raise ValueError("H per bundle must be positive")
        self._h_per_bundle = float(value)

    # E per bundle property
    @property
    def e_per_bundle(self) -> float:
        """Get kWh electricity per bundle."""
        return self._e_per_bundle

    @e_per_bundle.setter
    def e_per_bundle(self, value: float) -> None:
        """Set kWh electricity per bundle."""
        if value <= 0:
            raise ValueError("E per bundle must be positive")
        self._e_per_bundle = float(value)

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

    @property
    def max_hydrogen(self) -> float:
        """Calculate maximum hydrogen consumption."""
        return self._max_bundles * self._h_per_bundle

    @property
    def max_electricity(self) -> float:
        """Calculate maximum electricity consumption."""
        return self._max_bundles * self._e_per_bundle

    def utility(self, h: float, e: float) -> float:
        """
        Calculate utility at consumption levels (h, e).

        U(h, e) = A * h^α * e^β

        Args:
            h: Hydrogen consumption in kg
            e: Electricity consumption in kWh

        Returns:
            Utility value
        """
        if h <= 0 or e <= 0:
            return 0
        return self._A * (h ** self._alpha) * (e ** self._beta)

    def bundle_utility(self, n_bundles: float) -> float:
        """
        Calculate utility from n bundles of (h_per_bundle, e_per_bundle).

        Args:
            n_bundles: Number of bundles

        Returns:
            Utility value
        """
        h = n_bundles * self._h_per_bundle
        e = n_bundles * self._e_per_bundle
        return self.utility(h, e)

    def marginal_bundle_utility(self, n_bundles: float) -> float:
        """
        Calculate marginal utility of an additional bundle.

        dU/dn = dU/dh * dh/dn + dU/de * de/dn

        Args:
            n_bundles: Current number of bundles

        Returns:
            Marginal utility of one more bundle
        """
        if n_bundles <= 0:
            n_bundles = 0.1  # Small value to avoid division issues

        h = n_bundles * self._h_per_bundle
        e = n_bundles * self._e_per_bundle

        # Partial derivatives
        dU_dh = self._A * self._alpha * (h ** (self._alpha - 1)) * (e ** self._beta)
        dU_de = self._A * self._beta * (h ** self._alpha) * (e ** (self._beta - 1))

        # Chain rule
        return dU_dh * self._h_per_bundle + dU_de * self._e_per_bundle

    def marginal_utility_h(self, h: float, e: float) -> float:
        """
        Calculate marginal utility with respect to hydrogen.

        ∂U/∂h = A * α * h^(α-1) * e^β

        Args:
            h: Current hydrogen consumption
            e: Current electricity consumption

        Returns:
            Marginal utility of hydrogen
        """
        if h <= 0 or e <= 0:
            return float('inf') if h <= 0 else 0
        return self._A * self._alpha * (h ** (self._alpha - 1)) * (e ** self._beta)

    def marginal_utility_e(self, h: float, e: float) -> float:
        """
        Calculate marginal utility with respect to electricity.

        ∂U/∂e = A * β * h^α * e^(β-1)

        Args:
            h: Current hydrogen consumption
            e: Current electricity consumption

        Returns:
            Marginal utility of electricity
        """
        if h <= 0 or e <= 0:
            return float('inf') if e <= 0 else 0
        return self._A * self._beta * (h ** self._alpha) * (e ** (self._beta - 1))

    def optimal_ratio(self) -> Tuple[float, float]:
        """
        Calculate optimal consumption ratio h:e.

        At optimum: MUh/Ph = MUe/Pe
        This gives h:e = α:β (normalized to prices)

        Returns:
            Tuple of (h_ratio, e_ratio) normalized to sum to 1
        """
        total = self._alpha + self._beta
        return (self._alpha / total, self._beta / total)

    def generate_tranches(self) -> List[Tuple[float, float]]:
        """
        Generate tranches for bundle consumption.

        Each bundle uses h_per_bundle hydrogen and e_per_bundle electricity.

        Returns:
            List of (bundle_quantity, willingness_to_pay) tuples
        """
        tranches = []
        tranche_size = self._max_bundles / self._num_tranches

        for i in range(self._num_tranches):
            midpoint = (i + 0.5) * tranche_size
            wtp = self.marginal_bundle_utility(midpoint)
            tranches.append((tranche_size, wtp))

        return tranches

    def generate_demand_tranches(self) -> List[DemandTranche]:
        """
        Generate DemandTranche objects for LP approximation.

        Returns:
            List of DemandTranche objects for bundle consumption
        """
        return generate_tranches(
            marginal_utility_fn=self.marginal_bundle_utility,
            max_quantity=self._max_bundles,
            num_tranches=self._num_tranches,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'A': self._A,
            'alpha': self._alpha,
            'beta': self._beta,
            'max_bundles': self._max_bundles,
            'h_per_bundle': self._h_per_bundle,
            'e_per_bundle': self._e_per_bundle,
            'num_tranches': self._num_tranches,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CobbDouglasUtility':
        """Create CobbDouglasUtility from dictionary."""
        return cls(
            _A=float(data.get('A', 10.0)),
            _alpha=float(data.get('alpha', 0.4)),
            _beta=float(data.get('beta', 0.6)),
            _max_bundles=float(data.get('max_bundles', 20.0)),
            _h_per_bundle=float(data.get('h_per_bundle', 1.0)),
            _e_per_bundle=float(data.get('e_per_bundle', 3.0)),
            _num_tranches=int(data.get('num_tranches', 5)),
        )

    def __repr__(self) -> str:
        return (f"CobbDouglasUtility(A={self._A}, alpha={self._alpha}, beta={self._beta}, "
                f"bundles={self._max_bundles}, h/b={self._h_per_bundle}, e/b={self._e_per_bundle})")
