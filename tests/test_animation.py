import matplotlib

matplotlib.use("Agg")


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
    observables = so.analyzers.Observables(grid.x_vals, grid.psi, grid.V)
    anim = so.animate1d(grid, so.SplitOperator(), dt=0.01, n_frames=4, analyzers=[observables])
    out = tmp_path / "movie.gif"
    anim.save(out, writer="pillow", fps=10)
    assert out.stat().st_size > 0
    assert len(observables._t_samples) == 4
