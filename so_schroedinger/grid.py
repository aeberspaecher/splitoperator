"""Grids hold the discretized wavefunction and potential.
"""

import logging

import numpy as np

# list of exported names:
__all__ = ["Grid", "Grid1d"]

log = logging.getLogger(__name__)


#-----------------------------------------------------------------------
# Grids:
class Grid:
    """A grid object holds discrete representations of the wavepacket and the
    potential at a given time t. It also knows about the initial
    wavepacket. """

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
        self.updatePot = potential.isTimeDependent

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

    def suggestTimeStep(self, safety=3.0):
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

        return 2*np.pi/(safety*(np.max(np.real(self.V)) - np.min(np.real(self.V))))


class Grid1d(Grid):
    r"""One dimensional periodic grid.

    The grid points are $x_j = x_\mathrm{min} + j\,\Delta x$ with $\Delta x =
    (x_\mathrm{max} - x_\mathrm{min})/N$, i.e. the right boundary is not
    included, as appropriate for the periodicity implied by the FFT.
    """

    def __init__(self, size, gridsamples, potential, psi0, absorber=None):
        """Create a 1d grid.

        Parameters
        ----------
        size : tuple of float
            Grid boundaries (x_min, x_max).
        gridsamples : int
            Number of grid points; powers of two are fastest.
        potential : Potential object
            The potential used in potential discretization.
        psi0 : WavePacket object
            Initial wavepacket object.
        absorber : Potential object, optional
            Absorbing boundary potential.
        """

        super().__init__(1, potential, psi0, absorber)
        if(np.ndim(gridsamples) > 0):
            gridsamples = gridsamples[0]
        self.gridSamples = int(gridsamples)
        self.x_extent = float(size[1]) - float(size[0])
        self.xVals = np.linspace(size[0], size[1], self.gridSamples, endpoint=False)
        self.dx = self.x_extent/self.gridSamples
        self.kVals = 2*np.pi*np.fft.fftfreq(self.gridSamples, self.dx)

        if(absorber is not None):
            self._absorber0 = np.asarray(absorber.V(self.xVals, 0.0), dtype=complex)
        else:
            self._absorber0 = np.zeros(self.gridSamples, dtype=complex)
        self.potentialUpdate(0.0)
        self.psi = psi0.psi(self.xVals)

        log.info("Maximum potential (real) value on grid V_max = %s", np.max(np.real(self._pot)))
        log.info("Minimum potential (real) value on grid V_min = %s", np.min(np.real(self._pot)))
        log.info("Maximum kinetic energy supported by grid T_max = %s", np.max(self.kVals**2/2.0))

    @property
    def psiNorm(self):
        r"""Norm $\Vert\psi\Vert$."""
        return np.sqrt(np.sum(np.abs(self._psi)**2)*self.dx)

    @property
    def density(self):
        r"""Probability density $\vert\psi\vert^2$."""
        return np.abs(self._psi)**2

    def potentialUpdate(self, t):
        """Update potential for time $t$.
        """

        self._pot = np.asarray(self._potential.V(self.xVals, t), dtype=complex) + self._absorber0

    def renormalizePsi(self):
        r"""Renormalize $\psi$ such that $\Vert\psi\Vert = 1$.
        """

        self._psi = self._psi/self.psiNorm

    def makePsiPeriodic(self):
        """Try to avoid Gibbs' phenomenon by manually making the wavefunction
        periodic.
        """

        psiVal = (self._psi[0] + self._psi[-1])/2.0
        self._psi[0] = psiVal
        self._psi[-1] = psiVal

    def expectationX(self):
        r"""Position expectation value $\langle x\rangle$."""
        return np.sum(self.xVals*self.density)/np.sum(self.density)

    def expectationP(self):
        r"""Momentum expectation value $\langle p\rangle$, computed in momentum
        space."""
        Psi = np.fft.fft(self._psi)
        return np.sum(self.kVals*np.abs(Psi)**2)/np.sum(np.abs(Psi)**2)

    def widthX(self):
        r"""Position uncertainty $\Delta x$."""
        xMean = self.expectationX()
        return np.sqrt(np.sum((self.xVals - xMean)**2*self.density)/np.sum(self.density))
