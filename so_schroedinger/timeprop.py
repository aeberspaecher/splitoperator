"""Simple quantum time propagation in 1d and 2d.

All physical quantities are in (Hartree) atomic units
(numerical values: hbar = e = m = k_B = 1; c=1/alpha), cmp. also
http://en.wikipedia.org/wiki/Atomic_units
"""

import logging

import numpy as np

# list of exported symbols:
__all__ = ["Propagator", "SplitOperator"]

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------
class Propagator:
    """Base class for wavepacket time-propagation."""

    def step(self, grid, dt):
        """To be implemented in subclasses."""

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

    def __init__(self, renormalize_every=0):
        """Create a split operator propagator.

        Parameters
        ----------
        renormalize_every : int, optional
            Renormalize the wavefunction after this many steps have been made.
            Use 0 if you do not want renormalization.
        """

        self._stepcount = 0
        self._renormalize_every = int(renormalize_every)
        self.ground_state_energy = None
        self._kin_cache = (None, None, None)  # (k_squared array, dt, exp(-i k^2 dt/2))
        self._pot_cache = (None, None, None)  # (V array, dt, exp(-i V dt))

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

        if grid.update_pot:
            grid.update_potential(grid.t)

        norm_before = grid.psi_norm
        self._apply_potential_part(grid, dt / 2.0)
        self._apply_kinetic_part(grid, dt)
        self._apply_potential_part(grid, dt / 2.0)

        grid.t = grid.t + np.abs(dt)  # the abs() call is needed for imaginary time evolution

        # estimate ground state energy in case of imaginary time evolution:
        # once the evolution has converged to the ground state, one step
        # multiplies psi by exp(-E_0 tau), cmp. Koeberle, Dissertation (2011),
        # p. 36
        if np.imag(dt) < 0.0:
            self.ground_state_energy = np.log(grid.psi_norm / norm_before) / np.imag(dt)
            log.debug("Current ground state energy estimate E_0 = %s", self.ground_state_energy)

        if self._renormalize_every and self._stepcount % self._renormalize_every == 0:
            grid.renormalize_psi()

    def propagate(self, grid, dt, n_steps, callback=None, every=1):
        """Do several time steps.

        Parameters
        ----------
        grid : Grid object
            The grid to be propagated.
        dt : complex
            Time step.
        n_steps : int
            Number of steps.
        callback : callable or sequence of callables, optional
            Called as ``callback(grid)`` after every ``every`` steps, e.g. an
            analyzer.
        every : int, optional
            Callback interval.
        """

        callbacks = [] if callback is None else [callback] if callable(callback) else list(callback)
        for i in range(1, int(n_steps) + 1):
            self.step(grid, dt)
            if i % every == 0:
                for cb in callbacks:
                    cb(grid)

    def _apply_potential_part(self, grid, dt):
        """Apply potential energy operator part of the propagator in position
        space (where the position operator is diagonal).
        """

        V, dt_cached, factor = self._pot_cache
        if V is not grid.V or dt_cached != dt:  # the grid creates a new array on each update
            factor = np.exp(-1j * grid.V * dt)
            self._pot_cache = (grid.V, dt, factor)
        grid.psi = grid.psi * factor

    def _apply_kinetic_part(self, grid, dt):
        """Apply kinetic energy operator part of the propagation.
        Use momentum representation in which the momentum operator is diagonal.

        Fourier transform -> Apply kinetic energy operator part
        -> inverse Fourier transform.
        """

        k_squared, dt_cached, factor = self._kin_cache
        if k_squared is not grid.k_squared or dt_cached != dt:
            factor = np.exp(-1j * grid.k_squared * dt / 2.0)
            self._kin_cache = (grid.k_squared, dt, factor)
        Psi = np.fft.fftn(grid.psi)
        Psi *= factor
        grid.psi = np.fft.ifftn(Psi)
