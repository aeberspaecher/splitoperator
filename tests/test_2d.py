import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

import so_schroedinger as so

SIZE = ((-20.0, 20.0), (-16.0, 16.0))


def test_grid2d_layout():
    grid = so.Grid2d(SIZE, (128, 64), so.ConstantPotential(0.0), so.Gaussian2d(0.0, 0.0, 1.0))
    assert grid.psi.shape == (128, 64)
    assert grid.X[1, 0] - grid.X[0, 0] == pytest.approx(grid.dx)
    assert grid.Y[0, 1] - grid.Y[0, 0] == pytest.approx(grid.dy)
    assert grid.extent == pytest.approx((-20.0, 20.0, -16.0, 16.0))
    assert grid.psiNorm == pytest.approx(1.0)


def test_free_gaussian2d_spreads_and_moves():
    sx, sy, px, py = 1.0, 2.0, 1.5, -0.5
    grid = so.Grid2d(SIZE, 128, so.ConstantPotential(0.0), so.Gaussian2d(-5.0, 2.0, sx, sy, px, py))
    so.SplitOperator().propagate(grid, 0.02, 100)
    t = 2.0
    xm, ym = grid.expectationR()
    assert xm == pytest.approx(-5.0 + px*t, abs=1e-6)
    assert ym == pytest.approx(2.0 + py*t, abs=1e-6)
    assert grid.expectationP() == pytest.approx((px, py), abs=1e-6)
    rho = grid.density/np.sum(grid.density)
    wx = np.sqrt(np.sum((grid.X - xm)**2*rho))
    assert wx == pytest.approx(sx/np.sqrt(2)*np.sqrt(1 + t**2/sx**4), rel=1e-6)
    assert grid.psiNorm == pytest.approx(1.0, abs=1e-10)


def test_imaginary_time_ground_state_2d():
    wx, wy = 1.0, 2.0
    grid = so.Grid2d(((-8, 8), (-8, 8)), 64, so.HarmOscPot2d(wx, wy), so.Gaussian2d(1.0, -0.5, 1.5))
    prop = so.SplitOperator(renormalizeEveryNSteps=1)
    prop.propagate(grid, -0.01j, 1500)
    assert prop.groundStateEnergy == pytest.approx((wx + wy)/2, rel=1e-3)


def test_stadium_signed_distance():
    pot = so.StadiumPotential2d(a=2.0, R=1.0, V0=10.0, w=0.01)
    r = (np.array([0.0, 0.0, 3.0, 4.0, 2.0 + np.sqrt(0.5)]), np.array([0.0, 2.0, 0.0, 0.0, np.sqrt(0.5)]))
    assert pot.signedDistance(r) == pytest.approx([-1.0, 1.0, 0.0, 1.0, 0.0])
    V = pot.V(r, 0.0)
    assert np.real(V[0]) == pytest.approx(0.0, abs=1e-12)
    assert np.real(V[1]) == pytest.approx(10.0)


def test_random_potential2d():
    x, y = np.linspace(0, 64, 256, endpoint=False), np.linspace(0, 32, 128, endpoint=False)
    corr = lambda dx, dy: np.exp(-(dx**2 + dy**2)/2)
    pot = so.RandomPotential2d(corr, x, y, 0.3, rng=3)
    X, Y = np.meshgrid(x, y, indexing="ij")
    V = pot.V((X, Y), 0.0)
    assert V.shape == X.shape
    assert np.allclose(V, pot.values)
    assert np.std(np.real(V)) == pytest.approx(0.3)
    # periodic continuation
    assert np.allclose(pot.V((X + 64, Y - 32), 0.0), V)
    assert np.array_equal(so.RandomPotential2d(corr, x, y, 0.3, rng=3).values, pot.values)


def test_boundary_absorber2d_absorbs():
    absorber = so.BoundaryAbsorber2d(SIZE, 4.0, 5.0)
    grid = so.Grid2d(SIZE, 128, so.ConstantPotential(0.0), so.Gaussian2d(0.0, 0.0, 1.5, px=4.0), absorber=absorber)
    assert np.all(np.imag(grid.V) <= 0.0)
    assert np.imag(grid.V)[64, 32] == 0.0
    so.SplitOperator().propagate(grid, 0.02, 400)
    assert grid.psiNorm < 0.1


def test_eigenstate_filter_finds_oscillator_states():
    omega = 1.0
    grid = so.Grid1d((-12.0, 12.0), 256, so.HarmOscPot(omega), so.Gaussian1d(1.0, 1.0, 0.5))
    tMax, dt = 150.0, 0.02
    filt = so.analyzers.EigenstateFilter([0.5, 1.5], tMax)
    prop = so.SplitOperator()
    filt.addData(grid.psi, grid.t, dt)
    for i in range(int(tMax/dt)):
        prop.step(grid, dt)
        filt.addData(grid.psi, grid.t, dt)
    phi0, phi1 = filt.eigenstates(grid.dx)
    x = grid.xVals
    exact0 = np.exp(-x**2/2)/np.pi**0.25
    exact1 = np.sqrt(2)*x*np.exp(-x**2/2)/np.pi**0.25
    assert abs(np.sum(exact0*phi0)*grid.dx) == pytest.approx(1.0, abs=1e-3)
    assert abs(np.sum(exact1*phi1)*grid.dx) == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("mode", ["density", "phase"])
def test_animate2d_renders(mode):
    pot = so.StadiumPotential2d(2.0, 3.0, 20.0, w=0.3)
    grid = so.Grid2d(((-8, 8), (-6, 6)), 64, pot, so.Gaussian2d(0.0, 0.0, 1.0, px=2.0, py=1.0))
    anim = so.animate2d(grid, so.SplitOperator(), dt=0.01, nFrames=4, stepsPerFrame=2, mode=mode)
    html = anim.to_jshtml()
    assert len(html) > 1000
    assert grid.t == pytest.approx(0.06)
