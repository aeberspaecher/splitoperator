import numpy as np
import pytest

import so_schroedinger as so
from so_schroedinger.analyzers import (
    AutoCorrelation,
    EigenfunctionBuilder,
    EigenstateFilter,
    Flux,
    IntegratedDensity,
    Observables,
)


def oscillator_grid(omega=1.0):
    return so.Grid1d((-12.0, 12.0), 256, so.HarmOscPot(omega), so.Gaussian1d(1.0, 1.0, 0.5))


def oscillator_states(x):
    exact0 = np.exp(-(x**2) / 2) / np.pi**0.25
    exact1 = np.sqrt(2) * x * np.exp(-(x**2) / 2) / np.pi**0.25
    return exact0, exact1


def test_observables_free_gaussian_1d():
    sigma, x0, p0 = 1.5, -5.0, 2.0
    grid = so.Grid1d((-40.0, 40.0), 2**11, so.ConstantPotential(0.0), so.Gaussian1d(x0, sigma, p0))
    obs = Observables()
    obs(grid)
    so.SplitOperator().propagate(grid, 0.01, 400, callback=obs, every=40)
    t = obs.t
    assert len(obs) == 11
    assert t[-1] == pytest.approx(4.0)
    assert obs.norm == pytest.approx(np.ones(11), abs=1e-10)
    assert obs.position == pytest.approx(x0 + p0 * t, abs=1e-6)
    assert obs.momentum == pytest.approx(np.full(11, p0), abs=1e-6)
    assert obs.width == pytest.approx(sigma / np.sqrt(2) * np.sqrt(1 + t**2 / sigma**4), rel=1e-6)


def test_observables_free_gaussian_2d():
    sx, sy, px, py = 1.0, 2.0, 1.5, -0.5
    grid = so.Grid2d(((-20, 20), (-16, 16)), 128, so.ConstantPotential(0.0), so.Gaussian2d(-5.0, 2.0, sx, sy, px, py))
    obs = Observables()
    obs(grid)
    so.SplitOperator().propagate(grid, 0.02, 100, callback=obs, every=50)
    t = obs.t[:, None]
    assert obs.position.shape == (3, 2)
    assert obs.position == pytest.approx(np.array([-5.0, 2.0]) + np.array([px, py]) * t, abs=1e-6)
    assert obs.momentum == pytest.approx(np.tile([px, py], (3, 1)), abs=1e-6)
    sigmas = np.array([sx, sy])
    assert obs.width == pytest.approx(sigmas / np.sqrt(2) * np.sqrt(1 + t**2 / sigmas**4), rel=1e-6)


def test_autocorrelation_spectrum_has_oscillator_peaks():
    omega = 1.0
    grid = oscillator_grid(omega)
    corr = AutoCorrelation()
    corr(grid)
    so.SplitOperator().propagate(grid, 0.02, 2**13, callback=corr)
    assert corr.C[0] == pytest.approx(1.0)
    E, spec = corr.spectrum()
    assert np.all(np.diff(E) > 0)
    peaks = [
        E[i]
        for i in range(1, len(E) - 1)
        if spec[i] > spec[i - 1] and spec[i] > spec[i + 1] and spec[i] > 0.05 * spec.max()
    ]
    assert peaks[:3] == pytest.approx([0.5, 1.5, 2.5], abs=0.01)


def test_autocorrelation_spectrum_needs_equidistant_times():
    grid = oscillator_grid()
    corr = AutoCorrelation()
    prop = so.SplitOperator()
    for dt in [0.1, 0.1, 0.2]:
        prop.step(grid, dt)
        corr(grid)
    with pytest.raises(ValueError):
        corr.spectrum()


def test_eigenstate_filter_finds_oscillator_states():
    grid = oscillator_grid()
    t_max, dt = 150.0, 0.02
    filt = EigenstateFilter([0.5, 1.5], t_max)
    filt(grid)
    so.SplitOperator().propagate(grid, dt, int(t_max / dt) + 100, callback=filt)  # data after t_max is ignored
    phi0, phi1 = filt.eigenstates()
    exact0, exact1 = oscillator_states(grid.x_vals)
    assert abs(np.sum(exact0 * phi0) * grid.dx) == pytest.approx(1.0, abs=1e-3)
    assert abs(np.sum(exact1 * phi1) * grid.dx) == pytest.approx(1.0, abs=1e-3)


def test_eigenstate_filter_2d():
    grid = so.Grid2d(((-8, 8), (-8, 8)), 64, so.HarmOscPot2d(1.0, 2.0), so.Gaussian2d(0.5, 0.0, 1.0, 0.7))
    t_max, dt = 60.0, 0.02
    filt = EigenstateFilter(1.5, t_max)
    filt(grid)
    so.SplitOperator().propagate(grid, dt, int(t_max / dt), callback=filt, every=2)
    (phi,) = filt.eigenstates()
    exact = np.exp(-(grid.X**2) / 2 - grid.Y**2) * (2 / np.pi**2) ** 0.25
    assert abs(np.sum(exact * phi) * grid.dV) == pytest.approx(1.0, abs=1e-3)


def test_eigenfunction_builder_matches_filter():
    grid = oscillator_grid()
    builder = EigenfunctionBuilder()
    builder(grid)
    so.SplitOperator().propagate(grid, 0.02, 4000, callback=builder, every=5)
    filt = EigenstateFilter([0.5, 1.5], grid.t)
    for t, psi in zip(builder._t, builder._psi, strict=True):
        filt._add(t, psi, grid.dV)
    phi = builder.eigenstates([0.5, 1.5])
    assert phi == pytest.approx(filt.eigenstates())
    exact0, _ = oscillator_states(grid.x_vals)
    assert abs(np.sum(exact0 * phi[0]) * grid.dx) == pytest.approx(1.0, abs=1e-3)


def test_integrated_density():
    grid = so.Grid1d((-40.0, 40.0), 2**10, so.ConstantPotential(0.0), so.Gaussian1d(-5.0, 1.0, 1.0))
    acc = IntegratedDensity()
    acc(grid)
    so.SplitOperator().propagate(grid, 0.01, 300, callback=acc, every=3)
    assert np.sum(acc.density) * grid.dx == pytest.approx(3.0)
    assert np.sum(acc.density) * grid.dx == pytest.approx(3.0)  # reading does not change the result


def test_flux_measures_transmission():
    size = (-80.0, 80.0)
    grid = so.Grid1d(size, 2**12, so.GaussBarrier1d(0.0, 1.0, 1.0), so.Gaussian1d(-20.0, 3.0, 1.2))
    flux = Flux(10.0)
    flux(grid)
    so.SplitOperator().propagate(grid, 0.02, 2500, callback=flux)
    T = flux.transferred
    assert 0.1 < T < 0.9
    assert flux.cumulative()[-1] == pytest.approx(T)
    assert T == pytest.approx(np.sum(grid.density[grid.x_vals > 10.0]) * grid.dx, abs=1e-3)


def test_flux_counts_absorbed_probability():
    size = (-80.0, 80.0)
    absorber = so.ExpAbsorber1d(0.0, -1.0, 5.0, 5.0, *size)
    grid = so.Grid1d(size, 2**12, so.ConstantPotential(0.0), so.Gaussian1d(-20.0, 2.0, 3.0), absorber=absorber)
    flux = Flux(10.0)
    flux(grid)
    so.SplitOperator().propagate(grid, 0.02, 2000, callback=flux)
    assert grid.psi_norm**2 < 0.05
    assert flux.transferred == pytest.approx(1.0, abs=1e-3)


def test_flux_2d_free_packet_passes_line():
    grid = so.Grid2d(
        ((-16, 32), (-12, 12)), (256, 128), so.ConstantPotential(0.0), so.Gaussian2d(-5.0, 1.0, 1.0, px=3.0)
    )
    flux_x, flux_y = Flux(0.0, axis=0), Flux(-6.0, axis=1)
    so.SplitOperator().propagate(grid, 0.02, 250, callback=[flux_x, flux_y])
    assert flux_x.transferred == pytest.approx(np.sum(grid.density[grid.x_vals > 0]) * grid.dV, abs=1e-3)
    assert np.all(flux_x.flux > -1e-12)
    assert flux_y.transferred < 0.0  # the packet spreads downwards through y = -6
