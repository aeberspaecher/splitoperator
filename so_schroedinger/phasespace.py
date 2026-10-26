r"""Phase space representations of 1d wavefunctions.

The Wigner function

$$W(x, p) = \frac{1}{\pi}\int \psi^*(x + y)\,\psi(x - y)\,e^{2ipy}\,\mathrm{d}y$$

is real, its marginals are the position and momentum densities, and for
quadratic Hamiltonians it moves exactly like a classical phase space
density. Interference shows up as negative values. The Husimi function

$$Q(x, p) = \frac{1}{2\pi}\,\vert\langle x, p\vert\psi\rangle\vert^2$$

is the overlap with coherent states $\langle x'|x, p\rangle \propto
e^{-(x' - x)^2/(2\sigma^2) + ipx'}$. It is the Wigner function smoothed by a
Gaussian (widths $\sigma/\sqrt{2}$ in $x$ and $1/(\sqrt{2}\sigma)$ in $p$),
which makes it non-negative but blurs the interference fringes. Both are
normalized to $\int\!\!\int W\,\mathrm{d}x\,\mathrm{d}p = 1$ for $\Vert\psi\Vert = 1$.

Both functions take a ``Grid1d`` and return ``x, p, W`` with ``W[i, j]`` at
``(x[i], p[j])``. ``so.analyzers.Wigner`` and ``so.analyzers.Husimi``
record them during a propagation, and ``so.plotting.plot_phase_space()``
shows them.
"""

import numpy as np
from scipy.signal import resample

# list of exported names:
__all__ = ["wigner", "husimi"]


def _crop(values, lim):
    """Indices of ``values`` within ``lim = (lo, hi)``, all if ``lim`` is None."""

    if lim is None:
        return np.arange(len(values))
    return np.flatnonzero((values >= lim[0]) & (values <= lim[1]))


def wigner(grid, psi=None, x_lim=None, p_lim=None):
    r"""Wigner function $W(x, p)$ of a wavefunction on a 1d grid.

    $\psi$ is taken to vanish outside the grid, i.e. the integral runs only
    over $y$ for which $x \pm y$ lie on the grid; on a periodic grid the
    copies of $\psi$ would otherwise add "ghost" images. To reach the full momentum
    range $|p| < \pi/\Delta x$ of the grid, $\psi$ is first interpolated to
    twice the resolution (exact for band-limited $\psi$). The momenta are
    spaced by $\pi/L$, half the momentum spacing of the grid.

    Parameters
    ----------
    grid : Grid1d
    psi : array, optional
        Wavefunction; defaults to ``grid.psi``.
    x_lim, p_lim : tuple of float, optional
        Only return the part with ``x_lim[0] <= x <= x_lim[1]`` (and the same
        for ``p``); restricting ``x`` also saves time.

    Returns
    -------
    x : array
        Positions, a subset of ``grid.x_vals``.
    p : array
        Momenta, ascending.
    W : array
        ``W[i, j]`` $= W(x_i, p_j)$, real.
    """

    psi = np.asarray(grid.psi if psi is None else psi, dtype=complex)
    n = len(psi)
    m = 2 * n  # points of the fine grid
    fine = resample(psi, m)
    dy = grid.dx / 2
    i_x = _crop(grid.x_vals, x_lim)
    p = np.pi * np.fft.fftshift(np.fft.fftfreq(m, 1.0 / m)) / grid.x_extent
    i_p = _crop(p, p_lim)
    shift = np.fft.ifftshift(np.arange(m) - m // 2)  # y / dy in the order of the DFT
    j = 2 * i_x[:, None]  # original points are every second fine point
    plus, minus = j + shift, j - shift
    inside = (plus >= 0) & (plus < m) & (minus >= 0) & (minus < m)
    corr = np.where(inside, np.conj(fine[plus % m]) * fine[minus % m], 0.0)
    # sum_m corr[m] exp(2 i p_l m dy) with p_l = pi l / L is an inverse DFT
    W = np.real(np.fft.fftshift(np.fft.ifft(corr, axis=1), axes=1)) * m * dy / np.pi
    return grid.x_vals[i_x], p[i_p], W[:, i_p]


def husimi(grid, psi=None, sigma=1.0, x_lim=None, p_lim=None):
    r"""Husimi function $Q(x, p)$ of a wavefunction on a 1d grid.

    The coherent states have the position width $\sigma$ (that of
    ``so.Gaussian1d(x, sigma, p)``; for an oscillator of frequency
    $\omega$, $\sigma = 1/\sqrt{\omega}$ makes them its coherent states).
    The momenta are those of the grid, ``grid.k_vals``, sorted.

    Parameters
    ----------
    grid : Grid1d
    psi : array, optional
        Wavefunction; defaults to ``grid.psi``.
    sigma : float, optional
        Position width of the coherent states.
    x_lim, p_lim : tuple of float, optional
        Only return the part with ``x_lim[0] <= x <= x_lim[1]`` (and the same
        for ``p``); restricting ``x`` also saves time.

    Returns
    -------
    x : array
        Positions, a subset of ``grid.x_vals``.
    p : array
        Momenta, ascending.
    Q : array
        ``Q[i, j]`` $= Q(x_i, p_j)$, non-negative.
    """

    psi = np.asarray(grid.psi if psi is None else psi, dtype=complex)
    x, L = grid.x_vals, grid.x_extent
    i_x = _crop(x, x_lim)
    p = np.fft.fftshift(grid.k_vals)
    i_p = _crop(p, p_lim)
    d = x[None, :] - x[i_x, None]
    d = (d + L / 2) % L - L / 2  # periodic distance to the coherent state center
    window = np.exp(-(d**2) / (2 * sigma**2)) / (np.pi * sigma**2) ** 0.25
    # <x, p|psi> = int window(x' - x) e^{-i p x'} psi(x') dx', up to a phase
    overlap = np.fft.fftshift(np.fft.fft(window * psi[None, :], axis=1), axes=1) * grid.dx
    Q = np.abs(overlap) ** 2 / (2 * np.pi)
    return x[i_x], p[i_p], Q[:, i_p]
