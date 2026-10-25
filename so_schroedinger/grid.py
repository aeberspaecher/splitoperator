"""Grids hold the discretized wavefunction and potential."""

import logging

import numpy as np

# list of exported names:
__all__ = ["Grid", "Grid1d", "Grid2d"]

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# Grids:
class Grid:
    """A grid object holds discrete representations of the wavepacket and the
    potential at a given time t. It also knows about the initial
    wavepacket."""

    def __init__(self, dim, potential, psi0, absorber=None):
        """Create a grid.

        Parameters
        ----------
        dim : int
            Spatial dimension.
        potential : Potential object
            The potential used in potential discretization.
        psi0 : WavePacket object
            Initial wavepacket object.
        absorber : Potential object, optional
            Absorbing boundary potential.
        """

        self.dim = dim
        self._potential = potential
        self._absorber = absorber
        self._psi0 = psi0
        self.t = 0.0
        self.update_pot = potential.is_time_dependent

    @property
    def psi(self):
        r"""Current wavefunction $\psi(t)$."""
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = np.asarray(value, dtype=complex)

    @property
    def V(self):
        """Current potential $V(t)$, including the absorber."""
        return self._pot

    @property
    def psi_norm(self):
        r"""Norm $\Vert\psi\Vert$."""
        return np.sqrt(np.sum(np.abs(self._psi) ** 2) * self.dV)

    @property
    def density(self):
        r"""Probability density $\vert\psi\vert^2$."""
        return np.abs(self._psi) ** 2

    def update_potential(self, t):
        """Update potential for time $t$."""

        self._pot = np.asarray(self._potential.V(self._r, t), dtype=complex) + self._absorber0
        self._pot_t = t

    def renormalize_psi(self):
        r"""Renormalize $\psi$ such that $\Vert\psi\Vert = 1$."""

        self._psi = self._psi / self.psi_norm

    def _initialize(self):
        """Sample absorber, potential and initial wavefunction on the grid
        positions ``self._r``; called by derived classes once the grid is set
        up."""

        if self._absorber is not None:
            self._absorber0 = np.asarray(self._absorber.V(self._r, 0.0), dtype=complex)
        else:
            self._absorber0 = 0.0
        self.update_potential(0.0)
        self.psi = self._psi0.psi(self._r)

        log.info("Maximum potential (real) value on grid V_max = %s", np.max(np.real(self._pot)))
        log.info("Minimum potential (real) value on grid V_min = %s", np.min(np.real(self._pot)))
        log.info("Maximum kinetic energy supported by grid T_max = %s", np.max(self.k_squared / 2.0))

    def suggest_time_step(self, safety=3.0):
        r"""Suggest a time step from the potential landscape elevation. The
        highest difference in total energy that is possible is $\Delta E =
        \Delta V$, which corresponds to $\delta\omega = \Delta V$ (atomic
        units!) or $\delta t = 2\pi/\Delta V$.

        Do not use this for tiny or vanishing potentials as e.g. in
        free-particle propagation.

        Parameters
        ----------
        safety : float, optional
            Divide computed dt by this number for safer propagation.

        Returns
        -------
        dt : float
            Suggested time step.
        """

        return 2 * np.pi / (safety * (np.max(np.real(self.V)) - np.min(np.real(self.V))))


class Grid1d(Grid):
    r"""One dimensional periodic grid.

    The grid points are $x_j = x_\mathrm{min} + j\,\Delta x$ with $\Delta x =
    (x_\mathrm{max} - x_\mathrm{min})/N$, i.e. the right boundary is not
    included, as appropriate for the periodicity implied by the FFT.
    """

    def __init__(self, size, grid_samples, potential, psi0, absorber=None):
        """Create a 1d grid.

        Parameters
        ----------
        size : tuple of float
            Grid boundaries (x_min, x_max).
        grid_samples : int
            Number of grid points; powers of two are fastest.
        potential : Potential object
            The potential used in potential discretization.
        psi0 : WavePacket object
            Initial wavepacket object.
        absorber : Potential object, optional
            Absorbing boundary potential.
        """

        super().__init__(1, potential, psi0, absorber)
        if np.ndim(grid_samples) > 0:
            grid_samples = grid_samples[0]
        self.grid_samples = int(grid_samples)
        self.x_extent = float(size[1]) - float(size[0])
        self.x_vals = np.linspace(size[0], size[1], self.grid_samples, endpoint=False)
        self.dx = self.x_extent / self.grid_samples
        self.k_vals = 2 * np.pi * np.fft.fftfreq(self.grid_samples, self.dx)

        self.dV = self.dx
        self.k_squared = self.k_vals**2
        self._r = self.x_vals
        self._initialize()

    def make_psi_periodic(self):
        """Try to avoid Gibbs' phenomenon by manually making the wavefunction
        periodic.
        """

        psi_val = (self._psi[0] + self._psi[-1]) / 2.0
        self._psi[0] = psi_val
        self._psi[-1] = psi_val

    def expectation_x(self):
        r"""Position expectation value $\langle x\rangle$."""
        return np.sum(self.x_vals * self.density) / np.sum(self.density)

    def expectation_p(self):
        r"""Momentum expectation value $\langle p\rangle$, computed in momentum
        space."""
        Psi = np.fft.fft(self._psi)
        return np.sum(self.k_vals * np.abs(Psi) ** 2) / np.sum(np.abs(Psi) ** 2)

    def width_x(self):
        r"""Position uncertainty $\Delta x$."""
        x_mean = self.expectation_x()
        return np.sqrt(np.sum((self.x_vals - x_mean) ** 2 * self.density) / np.sum(self.density))


class Grid2d(Grid):
    r"""Two dimensional periodic grid.

    Arrays are indexed as ``psi[i, j]`` with $x_i$ along the first and $y_j$
    along the second axis (``indexing="ij"``). As in 1d, the upper
    boundaries are not included. 2d potentials and wavepackets are evaluated
    with ``r = (X, Y)``, a tuple of coordinate arrays.
    """

    def __init__(self, size, grid_samples, potential, psi0, absorber=None):
        """Create a 2d grid.

        Parameters
        ----------
        size : tuple
            Grid boundaries ((x_min, x_max), (y_min, y_max)).
        grid_samples : int or tuple of int
            Number of grid points (N_x, N_y); powers of two are fastest.
        potential : Potential object
            The potential used in potential discretization.
        psi0 : WavePacket object
            Initial wavepacket object.
        absorber : Potential object, optional
            Absorbing boundary potential.
        """

        super().__init__(2, potential, psi0, absorber)
        if np.ndim(grid_samples) == 0:
            grid_samples = (grid_samples, grid_samples)
        self.grid_samples = (int(grid_samples[0]), int(grid_samples[1]))
        (x0, x1), (y0, y1) = size
        self.x_extent, self.y_extent = float(x1) - float(x0), float(y1) - float(y0)
        self.x_vals = np.linspace(x0, x1, self.grid_samples[0], endpoint=False)
        self.y_vals = np.linspace(y0, y1, self.grid_samples[1], endpoint=False)
        self.dx = self.x_extent / self.grid_samples[0]
        self.dy = self.y_extent / self.grid_samples[1]
        self.dV = self.dx * self.dy
        self.X, self.Y = np.meshgrid(self.x_vals, self.y_vals, indexing="ij")
        self.kx_vals = 2 * np.pi * np.fft.fftfreq(self.grid_samples[0], self.dx)
        self.ky_vals = 2 * np.pi * np.fft.fftfreq(self.grid_samples[1], self.dy)
        self.k_squared = self.kx_vals[:, None] ** 2 + self.ky_vals[None, :] ** 2
        self._r = (self.X, self.Y)
        self._initialize()

    @property
    def extent(self):
        """(x_min, x_max, y_min, y_max), e.g. for ``imshow``."""
        return (self.x_vals[0], self.x_vals[0] + self.x_extent, self.y_vals[0], self.y_vals[0] + self.y_extent)

    def expectation_r(self):
        r"""Position expectation value $(\langle x\rangle, \langle y\rangle)$."""
        rho = self.density
        return np.sum(self.X * rho) / np.sum(rho), np.sum(self.Y * rho) / np.sum(rho)

    def width_r(self):
        r"""Position uncertainties $(\Delta x, \Delta y)$."""
        rho = self.density / np.sum(self.density)
        x_mean, y_mean = np.sum(self.X * rho), np.sum(self.Y * rho)
        return np.sqrt(np.sum((self.X - x_mean) ** 2 * rho)), np.sqrt(np.sum((self.Y - y_mean) ** 2 * rho))

    def expectation_p(self):
        r"""Momentum expectation value $(\langle p_x\rangle, \langle p_y\rangle)$."""
        rho_k = np.abs(np.fft.fft2(self._psi)) ** 2
        norm = np.sum(rho_k)
        return np.sum(self.kx_vals[:, None] * rho_k) / norm, np.sum(self.ky_vals[None, :] * rho_k) / norm
