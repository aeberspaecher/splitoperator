import numpy as np
import pytest

import so_schroedinger as so

x = np.linspace(-10, 10, 201)

ALL_POTENTIALS = [
    so.ConstantPotential(1.0),
    so.BoxPotential(0.0, 2.0, 3.0),
    so.SmoothedBoxPotential(0.0, 2.0, 3.0),
    so.WellPotential1d(0.0, 4.0, 0.0, 10.0),
    so.WellPotential1d(0.0, 4.0, 0.0, 10.0, d=1.0, Vext=1.0),
    so.SquareWell1d(1.0, 2.0, 4.0, 0.0),
    so.EckartPotential1d(0.0, 1.0, 1.0),
    so.GaussBarrier1d(0.0, 1.0, 1.0),
    so.Ramp1d(0.0, 1.0, 0.5, True),
    so.MovingRamp1d(0.0, 1.0, 0.5, True, "sin", 1.0, 1.0, 0.0),
    so.HarmOscPot(1.0),
    so.InterpolatingPotential1d([(-10, 5), (0, 0), (10, 5)]),
    so.RandomPotential1d(lambda r: np.exp(-r**2/2), x, 1.0, rng=0),
    so.ExpAbsorber1d(-1.0, -1.0, 2.0, 2.0, -10.0, 10.0),
]


@pytest.mark.parametrize("pot", ALL_POTENTIALS, ids=lambda p: type(p).__name__)
def test_vectorized_matches_pointwise(pot):
    V = pot.V(x, 0.3)
    assert V.shape == x.shape
    assert np.iscomplexobj(V)
    assert np.allclose(V, [pot.V(xi, 0.3) for xi in x])


def test_square_well_is_continuous_off_center():
    pot = so.SquareWell1d(x0=3.0, omega=2.0, width=4.0, V0=1.0)
    inner, outer = pot.V(np.array([4.999999, 5.000001]), 0.0)
    assert inner == pytest.approx(outer, abs=1e-4)


def test_superposition_adds_and_tracks_time_dependence():
    a, b = so.HarmOscPot(1.0), so.MovingRamp1d(0.0, 1.0, 0.5, True, "sin", 1.0, 1.0, 0.0)
    s = a + b
    assert s.isTimeDependent
    assert np.allclose(s.V(x, 0.1), a.V(x, 0.1) + b.V(x, 0.1))
    assert not (a + so.ConstantPotential(0)).isTimeDependent


def test_random_potential_is_reproducible_and_scaled():
    corr = lambda r: np.exp(-r**2/2)
    V1 = so.RandomPotential1d(corr, x, 2.0, rng=42).V(x, 0.0)
    V2 = so.RandomPotential1d(corr, x, 2.0, rng=42).V(x, 0.0)
    assert np.array_equal(V1, V2)
    assert np.max(np.abs(V1)) == pytest.approx(2.0)


def test_gaussian_random_field_has_requested_correlation():
    ell, dx, n = 2.0, 0.25, 2**14
    field = so.gaussian_random_field(n, dx, lambda r: np.exp(-r**2/(2*ell**2)), rng=1)
    lag = int(ell/dx)
    corr = np.mean(field*np.roll(field, lag))
    assert corr == pytest.approx(np.exp(-0.5), abs=0.05)
