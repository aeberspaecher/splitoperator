import matplotlib

matplotlib.use("Agg")


import numpy as np
import pytest

import so_schroedinger as so


def make_grid(potential):
    return so.Grid1d((-20.0, 20.0), 2**9, potential, so.Gaussian1d(-5.0, 1.0, 2.0))


def test_animation_renders_and_is_repeatable():
    grid = make_grid(so.GaussBarrier1d(0.0, 1.0, 0.5))
    anim = so.animate1d(grid, so.SplitOperator(), dt=0.01, n_frames=5, steps_per_frame=3, show_re_im=True)
    assert abs(grid.t - 0.01 * 3 * 4) < 1e-12
    html1 = anim.to_jshtml()
    html2 = anim.to_jshtml()
    assert "<img" in html1 or "animation" in html1.lower()
    assert len(html1) == len(html2)


def test_time_dependent_potential_and_gif(tmp_path):
    pot = so.MovingRamp1d(5.0, 2.0, 0.5, True, "sin", 1.0, 2.0, 0.0)
    grid = make_grid(pot)
    observables = so.analyzers.Observables()
    anim = so.animate1d(grid, so.SplitOperator(), dt=0.01, n_frames=4, analyzers=[observables])
    out = tmp_path / "movie.gif"
    anim.save(out, writer="pillow", fps=10)
    assert out.stat().st_size > 0
    assert len(observables) == 4
    assert observables.t[-1] == grid.t


def test_live_animation_propagates_while_drawing():
    grid = make_grid(so.GaussBarrier1d(0.0, 1.0, 0.5))
    observables = so.analyzers.Observables()
    anim = so.animate1d(
        grid, so.SplitOperator(), dt=0.01, n_frames=5, steps_per_frame=3, analyzers=[observables], live=True
    )
    assert grid.t == 0.0  # nothing propagated yet
    html = anim.to_jshtml()
    assert "<img" in html or "animation" in html.lower()
    assert abs(grid.t - 0.01 * 3 * 4) < 1e-12
    assert len(observables) == 5
    with pytest.raises(RuntimeError):
        anim._func(0)  # drawing it again from the start


def test_live_matches_recorded_2d():
    def run(live):
        size = ((-6.0, 6.0), (-6.0, 6.0))
        grid = so.Grid2d(size, 48, so.HarmOscPot2d(1.0), so.Gaussian2d(1.0, 0.0, 1.0, px=1.0))
        anim = so.animate2d(grid, so.SplitOperator(), dt=0.05, n_frames=4, steps_per_frame=2, live=live)
        anim.to_jshtml()
        return grid

    recorded, live = run(False), run(True)
    assert live.t == recorded.t
    assert np.allclose(live.psi, recorded.psi)
