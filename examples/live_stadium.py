"""Watch a wavepacket in the stadium billiard while it is being computed.

With ``live=True`` the animation propagates frame by frame as it is drawn,
so the window opens at once and no frames are stored. Run with
``python examples/live_stadium.py``.
"""

import matplotlib.pyplot as plt

import so_schroedinger as so

size = ((-4.5, 4.5), (-2.5, 2.5))
stadium = so.StadiumPotential2d(a=2.0, R=2.0, V0=400.0, w=0.04)
psi0 = so.Gaussian2d(x0=0.0, y0=0.0, sigma_x=0.4, px=8.0, py=3.0)
grid = so.Grid2d(size, (384, 216), stadium, psi0)

anim = so.animate2d(
    grid, so.SplitOperator(), dt=0.002, n_frames=2000, steps_per_frame=10, mode="phase", close_figure=False, live=True
)
plt.show()
