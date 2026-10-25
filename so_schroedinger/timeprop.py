"""Simple quantum time propagation in 1d and 2d.

All physical quantities are in (Hartree) atomic units
(numerical values: hbar = e = m = k_B = 1; c=1/alpha), cmp. also
http://en.wikipedia.org/wiki/Atomic_units
"""

import logging

import numpy as np
import scipy.fft

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

    For a time-dependent potential, the opening half step uses $V(t)$ and
    the closing one $V(t+\Delta t)$,
    $e^{-iV(t+\Delta t)\Delta t/2} e^{-iT\Delta t} e^{-iV(t)\Delta t/2}$,
    which keeps the scheme symmetric in time and second order accurate.

    In :meth:`propagate`, the closing potential half step of one step and the
    opening half step of the next one use the same potential and are merged
    into a single full step as long as no callback or renormalization comes
    in between. This gives the same result with one array multiplication less
    per step, and the potential is evaluated only once per step.

    With ``order=4`` or ``order=6``, one step is a symmetric composition of
    3 or 9 Strang steps $S(c_j\Delta t)$ (Yoshida's triple jump, applied once
    or twice), which makes the global error $O(\Delta t^4)$ or
    $O(\Delta t^6)$ for the cost of 3 or 9 FFT pairs per step. Some of the
    $c_j$ are negative, so the substeps briefly go back in time; this is
    harmless in real time but not in imaginary time, where a backward
    kinetic step amplifies high momenta, so higher orders need a real
    $\Delta t$.

    A time step with negative imaginary part, $\Delta t = -i\tau$, performs
    imaginary time evolution, which relaxes to the ground state.
    """

    def __init__(self, renormalize_every=0, workers=-1, order=2):
        """Create a split operator propagator.

        Parameters
        ----------
        renormalize_every : int, optional
            Renormalize the wavefunction after this many steps have been made.
            Use 0 if you do not want renormalization.
        order : {2, 4, 6}, optional
            Order of the splitting in the time step.
        workers : int, optional
            Number of threads for the FFTs, passed on to ``scipy.fft``; -1
            uses all CPU cores. Other FFT libraries can be plugged in with
            ``scipy.fft.set_backend()``, e.g. ``pyfftw.interfaces.scipy_fft``.
        """

        if order not in (2, 4, 6):
            raise ValueError("order must be 2, 4 or 6")
        self.order = order
        self._coeffs = _triple_jump(order)
        self._stepcount = 0
        self._renormalize_every = int(renormalize_every)
        self.workers = workers
        self.ground_state_energy = None
        self._kin_cache = (None, {})  # (k_squared array, {dt: exp(-i k^2 dt/2)})
        self._pot_cache = (None, {})  # (V array, {dt: exp(-i V dt)})

    def step(self, grid, dt):
        """Advance time by one step dt.

        Parameters
        ----------
        grid : Grid object
            The grid that holds all information needed to advance time.
        dt : complex
            Time step.
        """

        self._steps(grid, dt, 1)

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
        n_steps = int(n_steps)
        done = 0
        while done < n_steps:
            # steps up to the next callback or renormalization
            n = n_steps - done
            if callbacks:
                n = min(n, every - done % every)
            if self._renormalize_every:
                n = min(n, self._renormalize_every - self._stepcount % self._renormalize_every)
            self._steps(grid, dt, n)
            done += n
            if callbacks and done % every == 0:
                for cb in callbacks:
                    cb(grid)

    def _steps(self, grid, dt, n):
        """Do n steps, merging adjacent potential half steps."""

        if self.order > 2 and np.imag(dt) != 0.0:
            raise ValueError(f"order={self.order} needs a real time step")
        norm_before = grid.psi_norm
        if grid.update_pot and grid._pot_t != grid.t:
            grid.update_potential(grid.t)
        # one step is the composition S(c_1 dt) ... S(c_m dt) of Strang steps;
        # the potential half steps between two of them are merged
        coeffs = self._coeffs
        m = len(coeffs)
        self._apply_potential_part(grid, coeffs[0] * dt / 2.0)
        for i in range(n * m):
            c = coeffs[i % m]
            self._apply_kinetic_part(grid, c * dt)
            grid.t = grid.t + c * np.abs(dt)  # the abs() call is needed for imaginary time evolution
            if grid.update_pot:
                grid.update_potential(grid.t)
            c_next = coeffs[(i + 1) % m] if i < n * m - 1 else 0.0
            self._apply_potential_part(grid, (c + c_next) * dt / 2.0)
        self._stepcount += n

        # estimate ground state energy in case of imaginary time evolution:
        # once the evolution has converged to the ground state, one step
        # multiplies psi by exp(-E_0 tau), cmp. Koeberle, Dissertation (2011),
        # p. 36
        if np.imag(dt) < 0.0:
            self.ground_state_energy = np.log(grid.psi_norm / norm_before) / (n * np.imag(dt))
            log.debug("Current ground state energy estimate E_0 = %s", self.ground_state_energy)

        if self._renormalize_every and self._stepcount % self._renormalize_every == 0:
            grid.renormalize_psi()

    def _apply_potential_part(self, grid, dt):
        """Apply potential energy operator part of the propagator in position
        space (where the position operator is diagonal).
        """

        V, factors = self._pot_cache
        if V is not grid.V:  # the grid creates a new array on each update
            factors = {}
            self._pot_cache = (grid.V, factors)
        if dt not in factors:
            factors[dt] = np.exp(-1j * grid.V * dt)
        grid.psi *= factors[dt]

    def _apply_kinetic_part(self, grid, dt):
        """Apply kinetic energy operator part of the propagation.
        Use momentum representation in which the momentum operator is diagonal.

        Fourier transform -> Apply kinetic energy operator part
        -> inverse Fourier transform.
        """

        k_squared, factors = self._kin_cache
        if k_squared is not grid.k_squared:
            factors = {}
            self._kin_cache = (grid.k_squared, factors)
        if dt not in factors:
            factors[dt] = np.exp(-1j * grid.k_squared * dt / 2.0)
        Psi = scipy.fft.fftn(grid.psi, workers=self.workers)
        Psi *= factors[dt]
        grid.psi = scipy.fft.ifftn(Psi, workers=self.workers, overwrite_x=True)


def _triple_jump(order):
    """Coefficients c_j of a symmetric composition of Strang steps of the
    given order, from Yoshida's triple jump (Phys. Lett. A 150, 262 (1990)).
    """

    coeffs = [1.0]
    for p in range(2, order, 2):  # raise the order from p to p + 2
        w1 = 1.0 / (2.0 - 2.0 ** (1.0 / (p + 1)))
        w0 = 1.0 - 2.0 * w1
        coeffs = [c * w for w in (w1, w0, w1) for c in coeffs]
    return coeffs
