"""Split operator time propagation for the time-dependent Schroedinger
equation.

All quantities are in (Hartree) atomic units, hbar = m = 1.
"""

from .grid import *
from .potentials import *
from .wavepackets import *
from .timeprop import *
from . import analyzers

__version__ = "0.2.0"
