"""Split operator time propagation for the time-dependent Schroedinger
equation.

All quantities are in (Hartree) atomic units, hbar = m = 1.
"""

from . import analyzers, plotting
from .animation import animate1d, animate2d
from .eigenstates import *
from .grid import *
from .potentials import *
from .timeprop import *
from .wavepackets import *

__version__ = "0.4.0"
