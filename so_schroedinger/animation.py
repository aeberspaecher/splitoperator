"""Animations of time-propagated wavepackets.

The functions here return :class:`matplotlib.animation.FuncAnimation`
objects. In a Jupyter notebook, display them with
``IPython.display.HTML(anim.to_jshtml())``; save them to a file with
``anim.save("movie.mp4")`` (needs ffmpeg) or ``anim.save("movie.gif")``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.colors import hsv_to_rgb

__all__ = ["animate1d", "animate2d", "complex_to_rgb"]


def animate1d(
    grid,
    propagator,
    dt,
    n_frames,
    steps_per_frame=1,
    scale_pot=None,
    show_re_im=False,
    show_initial=True,
    analyzers=None,
    interval=40,
    ax=None,
    ylim=None,
    close_figure=True,
    live=False,
):
    r"""Animate the propagation of a wavepacket on a 1d grid.

    Each frame is ``steps_per_frame`` time steps after the previous one. By
    default the propagation is done up front and ``grid`` holds the final
    state afterwards; the returned animation can then be rendered or saved
    several times with identical results. With ``live=True`` the grid is
    propagated while the animation is drawn instead, see below.

    Parameters
    ----------
    grid : Grid1d
        The grid to propagate.
    propagator : Propagator
        E.g. a :class:`SplitOperator` instance.
    dt : complex
        Time step.
    n_frames : int
        Number of frames.
    steps_per_frame : int, optional
        Time steps per frame.
    scale_pot : float, optional
        Factor applied to the potential for plotting. By default, the
        potential is scaled such that its largest absolute (real) value
        matches the maximum of $\vert\psi_0\vert^2$.
    show_re_im : bool, optional
        Also plot $\mathrm{Re}\,\psi$ and $\mathrm{Im}\,\psi$.
    show_initial : bool, optional
        Plot $\vert\psi_0\vert^2$ in the background.
    analyzers : list of Analyzer, optional
        Each analyzer is called with the grid at every frame.
    interval : int, optional
        Delay between frames in milliseconds.
    ax : matplotlib Axes, optional
        Axes to draw into. A new figure is created if omitted.
    ylim : tuple of float, optional
        y axis limits.
    close_figure : bool, optional
        Close the figure so that notebooks do not show an extra static
        image. Pass False to show the animation with ``plt.show()``.
    live : bool, optional
        Propagate frame by frame while the animation is drawn instead of up
        front. The animation then starts at once in an interactive window
        (``plt.show()`` or ``%matplotlib widget``) and no frames are stored,
        which helps for long runs on large grids. Such an animation can be
        drawn only once, and analyzers are called as the frames are drawn.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
    else:
        fig = ax.figure

    x = grid.x_vals
    rho0 = grid.density.copy()
    frames = _Frames(grid, propagator, dt, n_frames, steps_per_frame, analyzers, lambda g: g.psi.copy(), live)

    rho_max = np.max(rho0)
    if scale_pot is None:
        V_max = np.max(np.abs(np.real(grid.V)))
        scale_pot = rho_max / V_max if V_max > 0 else 1.0

    (pot_real,) = ax.plot(x, scale_pot * np.real(grid.V), lw=1, c="tab:blue", label=r"$V$ (scaled)")
    (pot_imag,) = ax.plot(x, scale_pot * np.imag(grid.V), lw=0.8, ls="--", c="tab:blue")
    pot_fill = [ax.fill_between(x, scale_pot * np.real(grid.V), 0.0, color="tab:blue", alpha=0.2)]
    if show_initial:
        ax.plot(x, rho0, c="gray", lw=1, alpha=0.7, label=r"$\vert\psi_0\vert^2$")
    if show_re_im:
        (re_line,) = ax.plot(x, np.real(grid.psi), c="tab:green", lw=0.8, label=r"$\mathrm{Re}\,\psi$")
        (im_line,) = ax.plot(x, np.imag(grid.psi), c="tab:orange", lw=0.8, label=r"$\mathrm{Im}\,\psi$")
    (rho_line,) = ax.plot(x, rho0, c="tab:red", lw=1.5, label=r"$\vert\psi(t)\vert^2$")
    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, va="top")

    ax.set_xlim(x[0], x[-1])
    if ylim is None:
        lower = -np.max(np.abs(grid.psi)) if show_re_im else min(0.0, np.min(scale_pot * np.real(grid.V)))
        ylim = (lower - 0.05 * rho_max, 1.2 * max(rho_max, np.max(np.abs(grid.psi)) if show_re_im else rho_max))
    ax.set_ylim(*ylim)
    ax.set_xlabel("$x$")
    ax.legend(loc="upper right")

    def update(i):
        t, psi, V = frames[i]
        rho = np.abs(psi) ** 2
        rho_line.set_ydata(rho)
        artists = [rho_line, time_text]
        if show_re_im:
            re_line.set_ydata(np.real(psi))
            im_line.set_ydata(np.imag(psi))
            artists += [re_line, im_line]
        if V is not None:
            pot_real.set_ydata(scale_pot * np.real(V))
            pot_imag.set_ydata(scale_pot * np.imag(V))
            pot_fill[0].remove()
            pot_fill[0] = ax.fill_between(x, scale_pot * np.real(V), 0.0, color="tab:blue", alpha=0.2)
            artists += [pot_real, pot_imag, pot_fill[0]]
        norm_sq = np.sum(rho) * grid.dx
        time_text.set_text(f"$t = {t:.3f}$, $\\Vert\\psi\\Vert^2 = {norm_sq:.4f}$")
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False, repeat=False)
    if close_figure:
        plt.close(fig)  # avoid a duplicate static figure in notebooks
    return anim


class _Frames:
    """The frames ``(t, data, V)`` of an animation, V only if it changes.

    Without ``live``, all frames are computed and stored up front. With
    ``live``, frame i is computed when it is first asked for; frames must
    then be asked for in order, as an animation draws them.
    """

    def __init__(self, grid, propagator, dt, n_frames, steps_per_frame, analyzers, snapshot, live):
        self._grid, self._propagator, self._dt = grid, propagator, dt
        self._n, self._steps = int(n_frames), int(steps_per_frame)
        self._analyzers, self._snapshot, self._live = analyzers or [], snapshot, live
        self._index, self._current = 0, self._take()
        self._stored = None if live else [self._current] + [self._advance() for _ in range(1, self._n)]

    def _take(self):
        for analyzer in self._analyzers:
            analyzer(self._grid)
        grid = self._grid
        return (grid.t, self._snapshot(grid), grid.V.copy() if grid.update_pot else None)

    def _advance(self):
        self._propagator.propagate(self._grid, self._dt, self._steps)
        self._index += 1
        self._current = self._take()
        return self._current

    def __len__(self):
        return self._n

    def __getitem__(self, i):
        if self._stored is not None:
            return self._stored[i]
        if i == self._index + 1:
            self._advance()
        elif i != self._index:
            raise RuntimeError("A live animation can be drawn only once; use live=False to draw it again.")
        return self._current


def complex_to_rgb(psi, vmax=None):
    """Color a complex array: hue shows the phase, brightness the modulus.

    Parameters
    ----------
    psi : complex array
    vmax : float, optional
        Modulus mapped to full brightness; defaults to max(|psi|).
    """

    amp = np.abs(psi)
    vmax = np.max(amp) if vmax is None else vmax
    hsv = np.empty(np.shape(psi) + (3,))
    hsv[..., 0] = (np.angle(psi) / (2 * np.pi)) % 1.0
    hsv[..., 1] = 1.0
    hsv[..., 2] = np.clip(amp / vmax, 0.0, 1.0)
    return hsv_to_rgb(hsv)


def animate2d(
    grid,
    propagator,
    dt,
    n_frames,
    steps_per_frame=1,
    mode="density",
    potential_levels=5,
    cmap="magma",
    vmax=None,
    analyzers=None,
    interval=40,
    ax=None,
    close_figure=True,
    live=False,
):
    r"""Animate the propagation of a wavepacket on a 2d grid.

    As in :func:`animate1d`, the propagation is done up front and ``grid``
    holds the final state afterwards, unless ``live=True``.

    Parameters
    ----------
    grid : Grid2d
        The grid to propagate.
    propagator : Propagator
        E.g. a :class:`SplitOperator` instance.
    dt : complex
        Time step.
    n_frames : int
        Number of frames.
    steps_per_frame : int, optional
        Time steps per frame.
    mode : {"density", "phase"}, optional
        Show $\vert\psi\vert^2$ with a colormap, or $\psi$ with the phase as
        hue and $\vert\psi\vert$ as brightness.
    potential_levels : int or sequence, optional
        Contour levels of the real part of the potential drawn on top; 0 or
        None for no contours.
    cmap : str, optional
        Colormap for mode "density".
    vmax : float, optional
        Value for full color (density) or brightness (modulus); defaults to
        the maximum of the initial state.
    analyzers : list of Analyzer, optional
        Each analyzer is called with the grid at every frame.
    interval : int, optional
        Delay between frames in milliseconds.
    ax : matplotlib Axes, optional
        Axes to draw into. A new figure is created if omitted.
    close_figure : bool, optional
        Close the figure so that notebooks do not show an extra static image.
    live : bool, optional
        Propagate while the animation is drawn, see :func:`animate1d`.

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
    """

    if mode not in ("density", "phase"):
        raise ValueError("mode must be 'density' or 'phase'")
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6 * grid.y_extent / grid.x_extent))
    else:
        fig = ax.figure

    def snapshot(g):
        return g.density.astype(np.float32) if mode == "density" else g.psi.astype(np.complex64)

    frames = _Frames(grid, propagator, dt, n_frames, steps_per_frame, analyzers, snapshot, live)

    def image(data):
        if mode == "density":
            return data.T
        return complex_to_rgb(data.T, vmax)

    if vmax is None:
        vmax = np.max(np.abs(frames[0][1]))
    im = ax.imshow(
        image(frames[0][1]),
        origin="lower",
        extent=grid.extent,
        cmap=cmap,
        vmin=0.0,
        vmax=vmax,
        interpolation="bilinear",
    )
    contours = [None]

    def draw_contours(V):
        if potential_levels:
            if contours[0] is not None:
                contours[0].remove()
            contours[0] = ax.contour(
                grid.x_vals, grid.y_vals, V.T, levels=potential_levels, colors="white", linewidths=0.6, alpha=0.6
            )

    draw_contours(np.real(grid.V) if frames[0][2] is None else np.real(frames[0][2]))
    time_text = ax.text(0.02, 0.97, "", transform=ax.transAxes, va="top", color="white")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    def update(i):
        t, data, V = frames[i]
        im.set_data(image(data))
        if V is not None:
            draw_contours(np.real(V))
        time_text.set_text(f"$t = {t:.2f}$")
        return [im, time_text]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False, repeat=False)
    if close_figure:
        plt.close(fig)
    return anim
