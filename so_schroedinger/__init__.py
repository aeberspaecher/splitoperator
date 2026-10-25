"""Split operator time propagation for the time-dependent Schroedinger
equation.

All quantities are in (Hartree) atomic units, hbar = m = 1.
"""

from .grid import *
from .potentials import *
from .wavepackets import *
from .timeprop import *
from . import analyzers
from .animation import animate1d, animate2d

__version__ = "0.3.0"
