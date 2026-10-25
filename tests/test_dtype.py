import numpy as np
import pytest
from test_propagation import DrivenOscillator

import so_schroedinger as so


def coherent_grid(dtype, potential=None, absorber=None):
    potential = so.HarmOscPot(1.0) if potential is None else potential
    return so.Grid1d((-20.0, 20.0), 2**10, potential, so.CoherentState(3.0, 1.0, 1.0), absorber=absorber, dtype=dtype)


def test_single_precision_agrees_with_double():
    results = []
    for dtype in (np.complex128, np.complex64):
        grid = coherent_grid(dtype)
        so.SplitOperator().propagate(grid, 0.01, 628)
        results.append((grid.expectation_x(), grid.energy(), grid.psi_norm))
    (x128, E128, _), (x64, E64, norm64) = results
    assert x64 == pytest.approx(x128, rel=1e-5)
    assert E64 == pytest.approx(E128, rel=1e-5)
    assert norm64 == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("order", [2, 4])
def test_single_precision_stays_single(order):
    absorber = so.ExpAbsorber1d(0.0, -2.0, 5.0, 5.0, -20.0, 20.0)
    grid = coherent_grid(np.complex64, DrivenOscillator(1.0, 1.0, 2.5), absorber)
    assert grid.psi.dtype == grid.V.dtype == np.complex64
    so.SplitOperator(renormalize_every=7, order=order).propagate(grid, 0.01, 50, callback=lambda g: None, every=3)
    assert grid.psi.dtype == grid.V.dtype == np.complex64


def test_single_precision_in_2d():
    psi0 = so.Gaussian2d(3.0, 0.0, 1.0, 1.0, 0.0, 1.0)
    size = ((-10.0, 10.0), (-10.0, 10.0))
    grids = [so.Grid2d(size, 128, so.HarmOscPot2d(1.0), psi0, dtype=d) for d in ("complex128", "complex64")]
    for grid in grids:
        so.SplitOperator().propagate(grid, 0.01, 200)
    assert grids[1].psi.dtype == np.complex64
    assert grids[1].expectation_r() == pytest.approx(grids[0].expectation_r(), abs=1e-5)


def test_single_precision_eigenstates_converge():
    grid = so.Grid1d((-15.0, 15.0), 2**9, so.HarmOscPot(1.0), so.Gaussian1d(0.0, 1.0, 0.0), dtype=np.complex64)
    energies, states = so.lowest_eigenstates(grid, 3, rng=0, max_steps=5000)
    assert energies == pytest.approx([0.5, 1.5, 2.5], abs=1e-5)
    assert states[0].dtype == np.complex64


def test_only_complex_dtypes_are_accepted():
    with pytest.raises(ValueError, match="complex64 or complex128"):
        coherent_grid(np.float32)
