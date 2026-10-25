"""Simple quantum time propagation.

All physical quantities are in (Hartree) atomic units
(numerical values: hbar = e = m = k_B = 1; c=1/alpha), cmp. also
http://en.wikipedia.org/wiki/Atomic_units
"""

import logging

import numpy as np

# list of exported symbols:
__all__ = ["Propagator", "SplitOperator"]

log = logging.getLogger(__name__)


#-----------------------------------------------------------------------
class Propagator:
    """Base class for wavepacket time-propagation.
    """

    def step(self, grid, dt):
        """To be implemented in subclasses.
        """

        raise NotImplementedError


class SplitOperator(Propagator):
    r"""Split operator time propagation.

    Uses the symmetrized (Strang) splitting
    $e^{-iH\Delta t} \approx e^{-iV\Delta t/2} e^{-iT\Delta t} e^{-iV\Delta t/2}$
    with a local error of $O(\Delta t^3)$. The kinetic part is applied in
    momentum space via FFT.

    A time step with negative imaginary part, $\Delta t = -i\tau$, performs
    imaginary time evolution, which relaxes to the ground state.
    """

    def __init__(self, renormalizeEveryNSteps=0):
        """Create a split operator propagator.

        Parameters
        ----------
        renormalizeEveryNSteps : int, optional
            Renormalize the wavefunction after this many steps have been made.
            Use 0 if you do not want renormalization.
        """

        self._stepcount = 0
        self._doRenormalize = int(renormalizeEveryNSteps)
        self.groundStateEnergy = None

    def step(self, grid, dt):
        """Advance time by one step dt.

        Parameters
        ----------
        grid : Grid object
            The grid that holds all information needed to advance time.
        dt : complex
            Time step.
        """

        self._stepcount += 1

        if(grid.updatePot):
            grid.potentialUpdate(grid.t)

        normBefore = grid.psiNorm
        self._applyPotOperatorPart(grid, dt/2.0)
        self._applyKinOperatorPart(grid, dt)
        self._applyPotOperatorPart(grid, dt/2.0)

        grid.t = grid.t + np.abs(dt)  # the abs() call is needed for imaginary time evolution

        # estimate ground state energy in case of imaginary time evolution:
        # once the evolution has converged to the ground state, one step
        # multiplies psi by exp(-E_0 tau), cmp. Koeberle, Dissertation (2011),
        # p. 36
        if(np.imag(dt) < 0.0):
            self.groundStateEnergy = np.log(grid.psiNorm/normBefore)/np.imag(dt)
            log.debug("Current ground state energy estimate E_0 = %s", self.groundStateEnergy)

        if(self._doRenormalize and self._stepcount % self._doRenormalize == 0):
            grid.renormalizePsi()

    def propagate(self, grid, dt, nSteps, callback=None, every=1):
        """Do several time steps.

        Parameters
        ----------
        grid : Grid object
            The grid to be propagated.
        dt : complex
            Time step.
        nSteps : int
            Number of steps.
        callback : callable, optional
            Called as ``callback(grid)`` after every ``every`` steps.
        every : int, optional
            Callback interval.
        """

        for i in range(1, int(nSteps) + 1):
            self.step(grid, dt)
            if(callback is not None and i % every == 0):
                callback(grid)

    def _applyPotOperatorPart(self, grid, dt):
        """Apply potential energy operator part of the propagator in position
        space (where the position operator is diagonal).
        """

        grid.psi = grid.psi*np.exp(-1j*grid.V*dt)

    def _applyKinOperatorPart(self, grid, dt):
        """Apply kinetic energy operator part of the propagation.
        Use momentum representation in which the momentum operator is diagonal.

        Fourier transform -> Apply kinetic energy operator part
        -> inverse Fourier transform.
        """

        Psi = np.fft.fft(grid.psi)
        Psi *= np.exp(-1j*grid.kVals**2*dt/2.0)
        grid.psi = np.fft.ifft(Psi)
