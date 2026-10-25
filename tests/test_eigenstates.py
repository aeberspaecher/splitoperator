import numpy as np
import pytest

import so_schroedinger as so


def overlaps(states, dV):
    return np.array([[np.sum(np.conj(a) * b) * dV for b in states] for a in states])


def test_oscillator_spectrum_1d():
    omega = 1.5
    grid = so.Grid1d((-12.0, 12.0), 2**9, so.HarmOscPot(omega), so.Gaussian1d(0.0, 1.0, 0.0))
    energies, states = so.lowest_eigenstates(grid, 5, rng=0)
    assert energies == pytest.approx(omega * (np.arange(5) + 0.5), abs=1e-7)
    assert overlaps(states, grid.dV) == pytest.approx(np.eye(5), abs=1e-8)
    assert grid.t == 0.0
    assert grid.psi == pytest.approx(states[0])
    # parity alternates
    for n, phi in enumerate(states):
        assert np.abs(phi[1:] - (-1) ** n * phi[:0:-1]).max() < 1e-4 * np.abs(phi).max()


def test_degenerate_oscillator_states_2d():
    grid = so.Grid2d(((-7.0, 7.0), (-7.0, 7.0)), 64, so.HarmOscPot2d(1.0), so.Gaussian2d(0.0, 0.0, 1.0))
    energies, states = so.lowest_eigenstates(grid, 6, rng=0)
    assert energies == pytest.approx([1.0, 2.0, 2.0, 3.0, 3.0, 3.0], abs=1e-6)
    assert overlaps(states, grid.dV) == pytest.approx(np.eye(6), abs=1e-6)


def test_ground_state_and_energy():
    grid = so.Grid1d((-12.0, 12.0), 2**9, so.HarmOscPot(2.0), so.Gaussian1d(3.0, 0.5, 1.0))
    E0 = so.ground_state(grid, rng=0)
    assert E0 == pytest.approx(1.0, abs=1e-8)
    assert grid.energy() == pytest.approx(E0)
    # virial theorem; the state itself is only accurate to O(dt^2)
    assert grid.expectation_kinetic() == pytest.approx(0.5, abs=1e-4)
    assert grid.expectation_potential() == pytest.approx(0.5, abs=1e-4)


def test_time_dependent_potential_is_rejected():
    ramp = so.MovingRamp1d(0.0, 1.0, 0.5, True, "sin", 1.0, 1.0, 0.0)
    grid = so.Grid1d((-10.0, 10.0), 2**8, ramp, so.Gaussian1d(0.0, 1.0, 0.0))
    with pytest.raises(ValueError):
        so.ground_state(grid)
