"""Animate a Gaussian wavepacket scattering off a barrier.

Run with ``python examples/animate_barrier.py`` to show the animation, or pass
a file name (e.g. ``barrier.gif`` or ``barrier.mp4``) to save it.
"""

import sys

import matplotlib.pyplot as plt

import so_schroedinger as so
from so_schroedinger.animation import animate1d

size = (-60.0, 60.0)
psi0 = so.Gaussian1dEnergy(x0=-20.0, E0=2.0, delta_E=0.5)
barrier = so.GaussBarrier1d(x0=0.0, V0=2.0, sigma=0.7)
absorber = so.ExpAbsorber1d(0.0, -1.0, 5.0, 5.0, *size)
grid = so.Grid1d(size, 2**11, barrier, psi0, absorber=absorber)

anim = animate1d(grid, so.SplitOperator(), dt=0.02, n_frames=150, steps_per_frame=10, close_figure=False)

if(len(sys.argv) > 1):
    anim.save(sys.argv[1], fps=25)
else:
    plt.show()
