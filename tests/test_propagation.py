import numpy as np
import pytest

import so_schroedinger as so


def make_grid(potential, psi0, size=(-40.0, 40.0), samples=2**11, absorber=None):
    return so.Grid1d(size, samples, potential, psi0, absorber=absorber)


def test_norm_is_conserved():
    grid = make_grid(so.GaussBarrier1d(5.0, 2.0, 1.0), so.Gaussian1d(-10.0, 2.0, 2.0))
    prop = so.SplitOperator()
    norm0 = grid.psi_norm
    assert norm0 == pytest.approx(1.0, abs=1e-10)
    prop.propagate(grid, 0.01, 500)
    assert grid.psi_norm == pytest.approx(norm0, abs=1e-10)


def test_free_gaussian_spreads_analytically():
    sigma, x0, p0 = 1.5, -5.0, 2.0
    grid = make_grid(so.ConstantPotential(0.0), so.Gaussian1d(x0, sigma, p0))
    prop = so.SplitOperator()
    dt, n = 0.01, 400
    prop.propagate(grid, dt, n)
    t = dt * n
    assert grid.t == pytest.approx(t)
    assert grid.expectation_x() == pytest.approx(x0 + p0 * t, abs=1e-6)
    assert grid.expectation_p() == pytest.approx(p0, abs=1e-6)
    assert grid.width_x() == pytest.approx(sigma / np.sqrt(2) * np.sqrt(1 + t**2 / sigma**4), rel=1e-6)


def test_coherent_state_oscillates_without_spreading():
    omega, x0, p0 = 1.0, 3.0, 1.0
    psi0 = so.CoherentState(x0, omega, p0)
    grid = make_grid(so.HarmOscPot(omega), psi0, size=(-20.0, 20.0), samples=2**10)
    prop = so.SplitOperator()
    dt, n = 0.005, 1000
    prop.propagate(grid, dt, n)
    t = dt * n
    assert grid.expectation_x() == pytest.approx(x0 * np.cos(omega * t) + p0 / omega * np.sin(omega * t), abs=1e-4)
    assert grid.width_x() == pytest.approx(psi0.sigma / np.sqrt(2), rel=1e-4)


def test_imaginary_time_finds_oscillator_ground_state():
    omega = 2.0
    grid = make_grid(so.HarmOscPot(omega), so.Gaussian1d(1.0, 0.5, 0.0), size=(-15.0, 15.0), samples=2**9)
    prop = so.SplitOperator(renormalize_every=1)
    prop.propagate(grid, -0.005j, 2000)
    assert prop.ground_state_energy == pytest.approx(omega / 2, rel=1e-4)
    assert grid.expectation_x() == pytest.approx(0.0, abs=1e-6)


def test_absorber_removes_outgoing_packet():
    size = (-40.0, 40.0)
    absorber = so.ExpAbsorber1d(0.0, -2.0, 5.0, 5.0, size[0], size[1])
    grid = make_grid(so.ConstantPotential(0.0), so.Gaussian1d(0.0, 2.0, 5.0), size=size, absorber=absorber)
    assert np.all(np.imag(grid.V) <= 0.0)
    so.SplitOperator().propagate(grid, 0.01, 1500)
    assert grid.psi_norm < 0.05
