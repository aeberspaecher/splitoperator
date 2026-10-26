import numpy as np
import pytest

import so_schroedinger as so
from so_schroedinger.analyzers import Husimi, Wigner


def gaussian_grid(x0=3.0, sigma=1.3, p0=-2.0, omega=1.0):
    return so.Grid1d((-20.0, 20.0), 256, so.HarmOscPot(omega), so.Gaussian1d(x0, sigma, p0))


def test_wigner_of_a_gaussian():
    x0, sigma, p0 = 3.0, 1.3, -2.0
    grid = gaussian_grid(x0, sigma, p0)
    x, p, W = so.wigner(grid)
    assert W.shape == (256, 512)
    assert x == pytest.approx(grid.x_vals)
    assert p[-1] == pytest.approx(np.pi / grid.dx, rel=0.01)
    exact = np.exp(-((x[:, None] - x0) ** 2) / sigma**2 - sigma**2 * (p[None, :] - p0) ** 2) / np.pi
    assert W == pytest.approx(exact, abs=1e-12)


def test_wigner_marginals_and_negativity():
    grid = so.Grid1d((-15.0, 15.0), 256, so.HarmOscPot(1.0), so.Gaussian1d(0.0, 1.0, 0.0))
    _, states = so.lowest_eigenstates(grid, 2)
    x, p, W = so.wigner(grid, states[1])
    dp = p[1] - p[0]
    assert np.sum(W) * grid.dx * dp == pytest.approx(1.0)
    assert np.sum(W, axis=1) * dp == pytest.approx(np.abs(states[1]) ** 2, abs=1e-10)
    # the first excited oscillator state has W(0, 0) = -1/pi
    assert W[np.argmin(np.abs(x)), np.argmin(np.abs(p))] == pytest.approx(-1 / np.pi, rel=1e-6)


def test_wigner_crop():
    grid = gaussian_grid()
    x, p, W = so.wigner(grid)
    xc, pc, Wc = so.wigner(grid, x_lim=(-1.0, 1.0), p_lim=(0.0, 2.0))
    assert np.all((xc >= -1.0) & (xc <= 1.0)) and np.all((pc >= 0.0) & (pc <= 2.0))
    i, j = np.searchsorted(x, xc[0]), np.searchsorted(p, pc[0])
    assert Wc == pytest.approx(W[i : i + len(xc), j : j + len(pc)], abs=1e-15)


def test_husimi_of_a_coherent_state():
    x0, sigma, p0 = 3.0, 1.3, -2.0
    grid = gaussian_grid(x0, sigma, p0)
    x, p, Q = so.husimi(grid, sigma=sigma)
    assert p == pytest.approx(np.sort(grid.k_vals))
    assert np.sum(Q) * grid.dx * (p[1] - p[0]) == pytest.approx(1.0)
    exact = np.exp(-((x[:, None] - x0) ** 2) / (2 * sigma**2) - sigma**2 * (p[None, :] - p0) ** 2 / 2) / (2 * np.pi)
    assert Q == pytest.approx(exact, abs=1e-12)


def test_husimi_is_smoothed_wigner():
    # a cat state: W has negative fringes, Q is non-negative
    psi0 = so.Gaussian1d(-4.0, 1.0, 0.0) + so.Gaussian1d(4.0, 1.0, 0.0)
    grid = so.Grid1d((-20.0, 20.0), 256, so.HarmOscPot(1.0), psi0)
    grid.renormalize_psi()
    _, _, W = so.wigner(grid)
    _, _, Q = so.husimi(grid, sigma=0.5)
    assert W.min() < -0.2 / np.pi
    assert Q.min() > -1e-15


def test_wigner_rotates_in_the_oscillator():
    # for a quadratic Hamiltonian W moves like a classical density; in the
    # oscillator with omega = 1 it rotates: W(x, p, t) = W0(x cos t - p sin t, p cos t + x sin t)
    x0, sigma, p0 = 3.0, 0.7, 1.0  # squeezed, so the rotation is visible in the shape too
    grid = gaussian_grid(x0, sigma, p0)
    wig = Wigner(x_lim=(-8.0, 8.0), p_lim=(-8.0, 8.0))
    wig(grid)
    so.SplitOperator(order=4).propagate(grid, 0.01, 200, callback=wig, every=100)
    assert wig.t == pytest.approx([0.0, 1.0, 2.0])
    assert wig.W.shape == (3, len(wig.x), len(wig.p))
    X, P = np.meshgrid(wig.x, wig.p, indexing="ij")
    for t, W in zip(wig.t, wig.W, strict=True):
        Xr, Pr = X * np.cos(t) - P * np.sin(t), P * np.cos(t) + X * np.sin(t)
        exact = np.exp(-((Xr - x0) ** 2) / sigma**2 - sigma**2 * (Pr - p0) ** 2) / np.pi
        assert W == pytest.approx(exact, abs=1e-6)


def test_husimi_analyzer_and_1d_check():
    grid = gaussian_grid()
    hus = Husimi(sigma=1.3, p_lim=(-5.0, 5.0))
    hus(grid)
    assert hus.W.shape == (1, 256, len(hus.p))
    grid2d = so.Grid2d(((-5, 5), (-5, 5)), 16, so.ConstantPotential(0.0), so.Gaussian2d(0.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        Wigner()(grid2d)


def test_plot_phase_space():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    grid = gaussian_grid()
    fig, ax = plt.subplots()
    image = so.plotting.plot_phase_space(*so.wigner(grid), ax=ax)
    assert image.get_cmap().name == "magma"
    psi0 = so.Gaussian1d(-4.0, 1.0, 0.0) + so.Gaussian1d(4.0, 1.0, 0.0)
    grid = so.Grid1d((-20.0, 20.0), 256, so.HarmOscPot(1.0), psi0)
    image = so.plotting.plot_phase_space(*so.wigner(grid), ax=ax)
    assert image.get_cmap().name == "RdBu_r"
    assert image.get_clim()[0] == pytest.approx(-image.get_clim()[1])
    plt.close(fig)
