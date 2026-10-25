import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import so_schroedinger as so


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def grid1d():
    return so.Grid1d((-10.0, 10.0), 2**8, so.HarmOscPot(1.0), so.Gaussian1d(1.0, 1.0, 2.0))


def grid2d():
    return so.Grid2d(((-6.0, 6.0), (-4.0, 4.0)), (48, 32), so.HarmOscPot2d(1.0), so.Gaussian2d(1.0, 0.5, 1.0, px=1.0))


def test_plot1d_draws_density_and_potential():
    grid = grid1d()
    fig, ax = plt.subplots()
    line = so.plotting.plot1d(grid, ax=ax)
    assert np.allclose(line.get_ydata(), grid.density)
    assert len(ax.collections) == 1  # the potential
    assert ax.get_xlabel() == "$x$"


@pytest.mark.parametrize("kind", ["density", "abs", "phase", "real"])
def test_plot2d_orientation_and_extent(kind):
    grid = grid2d()
    fig, ax = plt.subplots()
    image = so.plotting.plot2d(grid, kind=kind, ax=ax, potential_levels=3)
    array = np.asarray(image.get_array())
    assert array.shape[:2] == (32, 48)  # rows are y, columns are x
    assert image.get_extent() == pytest.approx(list(grid.extent))
    if kind == "density":
        assert np.allclose(array, grid.density.T)


def test_plot2d_rejects_unknown_kind():
    with pytest.raises(ValueError):
        so.plotting.plot2d(grid2d(), kind="nonsense")


def test_momentum_distribution_is_normalized_and_centered():
    grid = grid1d()
    line = so.plotting.plot_momentum(grid)
    k, rho = line.get_xdata(), line.get_ydata()
    assert np.sum(rho) * (k[1] - k[0]) == pytest.approx(1.0)
    assert k[np.argmax(rho)] == pytest.approx(2.0, abs=k[1] - k[0])
    image = so.plotting.plot_momentum(grid2d(), ax=plt.figure().gca())
    assert np.asarray(image.get_array()).shape == (32, 48)


def test_plot_eigenstates_ladder():
    grid = so.Grid1d((-10.0, 10.0), 2**8, so.HarmOscPot(1.0), so.Gaussian1d(0.0, 1.0, 0.0))
    energies, states = so.lowest_eigenstates(grid, 3, rng=0)
    lines = so.plotting.plot_eigenstates(grid, states, energies, scale=1.0)
    assert len(lines) == 3
    assert np.min(lines[1].get_ydata()) == pytest.approx(energies[1])
