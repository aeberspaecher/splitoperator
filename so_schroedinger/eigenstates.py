r"""Eigenstates from imaginary time evolution.

Propagating in imaginary time, $\Delta t = -i\tau$, damps each eigenstate
as $e^{-E_n\tau}$, so any start state relaxes to the ground state. Excited
states follow by repeatedly projecting out the states found before
(Gram-Schmidt), which leaves the lowest remaining state to survive.
"""

import logging

import numpy as np

from .timeprop import SplitOperator

# list of exported names:
__all__ = ["ground_state", "lowest_eigenstates"]

log = logging.getLogger(__name__)


def lowest_eigenstates(grid, n_states, dt=0.01, tol=1e-10, max_steps=100_000, check_every=20, rng=None):
    r"""Find the lowest eigenstates by imaginary time evolution.

    Each state starts from random noise, which overlaps with every
    eigenstate, and is propagated in imaginary time. Every ``check_every``
    steps the states found before are projected out and the state is
    renormalized. A state counts as converged when its energy
    $\langle H\rangle$ changes by less than ``tol`` (relative to
    $\max(1, |E|)$) between two checks.

    The states are eigenstates of the split operator propagator, which
    differ from the exact ones by $O(\Delta t^2)$; the energies are
    variational and accurate to $O(\Delta t^4)$. Convergence is slow for
    nearly degenerate states: the error decays as $e^{-(E_{n+1}-E_n)\tau}$.

    Parameters
    ----------
    grid : Grid object
        Grid with a time-independent potential. Afterwards ``grid.psi`` holds
        the ground state; ``grid.t`` is unchanged.
    n_states : int
        Number of states.
    dt : float, optional
        Imaginary time step $\tau$.
    tol : float, optional
        Convergence tolerance for the energy.
    max_steps : int, optional
        Maximum number of steps per state; a warning is logged if a state
        has not converged by then.
    check_every : int, optional
        Steps between projections and convergence checks.
    rng : numpy.random.Generator or int, optional
        Random generator or seed for the start states.

    Returns
    -------
    energies : array
        Energies $E_0 \le E_1 \le \dots$
    states : list of arrays
        Normalized, mutually orthogonal eigenstates.
    """

    if grid.update_pot:
        raise ValueError("Eigenstates need a time-independent potential.")

    rng = np.random.default_rng(rng)
    prop = SplitOperator()
    tau = -1j * abs(dt)
    t_start = grid.t
    shape = np.shape(grid.psi)
    energies, states = [], []

    for n in range(int(n_states)):
        grid.psi = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        _project_out(grid, states)
        E_last, steps = np.inf, 0
        while True:
            prop.propagate(grid, tau, check_every)
            steps += check_every
            _project_out(grid, states)
            E = grid.energy()
            if abs(E - E_last) < tol * max(1.0, abs(E)):
                break
            if steps >= max_steps:
                log.warning("State %d not converged after %d steps (last change %g).", n, steps, abs(E - E_last))
                break
            E_last = E
        log.info("E_%d = %s after %d steps", n, E, steps)
        energies.append(E)
        states.append(grid.psi.copy())

    grid.psi = states[0].copy()
    grid.t = t_start
    return np.array(energies), states


def ground_state(grid, dt=0.01, tol=1e-10, max_steps=100_000, rng=None):
    r"""Relax ``grid.psi`` to the ground state by imaginary time evolution.

    See :func:`lowest_eigenstates` for the parameters.

    Returns
    -------
    E0 : float
        Ground state energy; ``grid.psi`` holds the ground state.
    """

    energies, _ = lowest_eigenstates(grid, 1, dt=dt, tol=tol, max_steps=max_steps, rng=rng)
    return energies[0]


def _project_out(grid, states):
    """Remove the components along ``states`` from ``grid.psi`` and
    renormalize (Gram-Schmidt, done twice for numerical stability)."""

    psi = grid.psi
    for _ in range(2):
        for phi in states:
            psi = psi - np.sum(np.conj(phi) * psi) * grid.dV * phi
    grid.psi = psi
    grid.renormalize_psi()
