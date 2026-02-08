"""
Utility functions and classes for the hydrogen simulation system.

This module provides:
- Utility function implementations (Logarithmic, Cobb-Douglas)
- Demand tranche generation
- Economic calculations
"""

from .demand import DemandTranche, generate_tranches
from .logarithmic import LogarithmicUtility
from .cobb_douglas import CobbDouglasUtility

__all__ = [
    'DemandTranche',
    'generate_tranches',
    'LogarithmicUtility',
    'CobbDouglasUtility',
]
