"""Numerical Green's functions for variable-coefficient elliptic PDEs."""

from .bempp_bridge import BemppOperators, CavitySolution
from .mfs import MFSSolution, solve_mfs
from .pde import PDE
from .split import SplitGreen

__all__ = [
    "PDE",
    "SplitGreen",
    "BemppOperators",
    "CavitySolution",
    "MFSSolution",
    "solve_mfs",
]
__version__ = "0.1.0"
