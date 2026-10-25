r"""One-line plots of wavefunctions, potentials and eigenstates.

Every function draws into ``ax`` (the current axes if omitted), labels the
axes and returns what it drew, so the plot can be refined afterwards:

>>> so.plotting.plot1d(grid)  # |psi|^2 with the potential in the background
>>> so.plotting.plot2d(grid, kind="phase", potential_levels=3)
>>> so.plotting.plot_eigenstates(grid, states, energies)
"""

import matplotlib.pyplot as plt
import numpy as np

from .animation import complex_to_rgb

# list of exported names:
__all__ = ["plot1d", "plot2d", "plot_momentum", "plot_eigenstates"]

_KINDS = ("density", "abs", "phase", "real")


def _field(data, kind, vmax=None):
    """Turn complex or real data into what ``imshow`` or ``plot`` shows."""

    if kind == "density":
        return np.abs(data) ** 2
    if kind == "abs":
        return np.abs(data)
    if kind == "phase":
        return complex_to_rgb(data, vmax)
    if kind == "real":
        return np.real(data)
    raise ValueError(f"kind must be one of {_KINDS}")


def plot1d(grid, psi=None, ax=None, potential=True, scale_pot=None, **kwargs):
    r"""Plot $\vert\psi\vert^2$ on a 1d grid, with the (scaled) real part of the
    potential as a shaded background.

    Parameters
    ----------
    grid : Grid1d
    psi : array, optional
        Wavefunction; defaults to ``grid.psi``.
    ax : matplotlib Axes, optional
    potential : bool, optional
        Draw the potential.
    scale_pot : float, optional
        Factor applied to the potential; by default its largest absolute value
        is scaled to the maximum of $\vert\psi\vert^2$.
    **kwargs
        Passed on to ``ax.plot`` for the density.

    Returns
    -------
    line : matplotlib Line2D
    """

    ax = plt.gca() if ax is None else ax
    rho = _field(grid.psi if psi is None else psi, "density")
    if potential:
        V = np.real(grid.V)
        if scale_pot is None:
            V_max = np.max(np.abs(V))
            scale_pot = np.max(rho) / V_max if V_max > 0 else 1.0
        ax.fill_between(grid.x_vals, scale_pot * V, 0.0, color="gray", alpha=0.25, lw=0)
    (line,) = ax.plot(grid.x_vals, rho, **kwargs)
    ax.set(xlim=(grid.x_vals[0], grid.x_vals[0] + grid.x_extent), xlabel="$x$", ylabel=r"$\vert\psi\vert^2$")
    return line


def plot2d(grid, data=None, kind="density", ax=None, potential_levels=0, vmax=None, **kwargs):
    r"""Show a field on a 2d grid as an image with the right orientation and
    extent.

    Parameters
    ----------
    grid : Grid2d
    data : array, optional
        Array indexed as ``data[i_x, i_y]``; defaults to ``grid.psi``.
    kind : {"density", "abs", "phase", "real"}, optional
        Show $\vert\psi\vert^2$, $\vert\psi\vert$, $\psi$ with the phase as hue
        and $\vert\psi\vert$ as brightness, or the real part of ``data`` (e.g.
        a potential).
    ax : matplotlib Axes, optional
    potential_levels : int or sequence, optional
        Contour levels of the real part of the potential drawn on top; 0 or
        None for none.
    vmax : float, optional
        Value for full color (or full brightness for "phase").
    **kwargs
        Passed on to ``ax.imshow``, e.g. ``cmap``.

    Returns
    -------
    image : matplotlib AxesImage
    """

    ax = plt.gca() if ax is None else ax
    data = grid.psi if data is None else data
    field = _field(data, kind, vmax)
    if kind != "phase":
        kwargs.setdefault("vmax", vmax)
        kwargs.setdefault("cmap", "RdBu_r" if kind == "real" else "magma")
    image = ax.imshow(np.swapaxes(field, 0, 1), origin="lower", extent=grid.extent, **kwargs)
    if potential_levels:
        V = np.real(grid.V)
        ax.contour(grid.x_vals, grid.y_vals, V.T, levels=potential_levels, colors="white", linewidths=0.6, alpha=0.6)
    ax.set(xlabel="$x$", ylabel="$y$")
    return image


def plot_momentum(grid, psi=None, ax=None, **kwargs):
    r"""Plot the momentum distribution $\vert\tilde\psi(k)\vert^2$, normalized
    to $\int\vert\tilde\psi\vert^2\,dk = 1$.

    In 1d this is a line plot, in 2d an image over $(k_x, k_y)$.

    Parameters
    ----------
    grid : Grid1d or Grid2d
    psi : array, optional
        Wavefunction; defaults to ``grid.psi``.
    ax : matplotlib Axes, optional
    **kwargs
        Passed on to ``ax.plot`` (1d) or ``ax.imshow`` (2d).

    Returns
    -------
    line or image
    """

    ax = plt.gca() if ax is None else ax
    rho_k = np.fft.fftshift(np.abs(np.fft.fftn(grid.psi if psi is None else psi)) ** 2)
    if grid.dim == 1:
        k = np.fft.fftshift(grid.k_vals)
        dk = k[1] - k[0]
        (line,) = ax.plot(k, rho_k / (np.sum(rho_k) * dk), **kwargs)
        ax.set(xlabel="$k$", ylabel=r"$\vert\tilde\psi(k)\vert^2$")
        return line
    kx, ky = np.fft.fftshift(grid.kx_vals), np.fft.fftshift(grid.ky_vals)
    dkx, dky = kx[1] - kx[0], ky[1] - ky[0]
    extent = (kx[0] - dkx / 2, kx[-1] + dkx / 2, ky[0] - dky / 2, ky[-1] + dky / 2)
    kwargs.setdefault("cmap", "magma")
    image = ax.imshow((rho_k / (np.sum(rho_k) * dkx * dky)).T, origin="lower", extent=extent, **kwargs)
    ax.set(xlabel="$k_x$", ylabel="$k_y$")
    return image


def plot_eigenstates(grid, states, energies, ax=None, scale=None, potential=True):
    r"""Plot 1d eigenstates as $\vert\varphi_n\vert^2 + E_n$ over the potential,
    the usual "ladder" picture.

    Parameters
    ----------
    grid : Grid1d
    states : sequence of arrays
    energies : sequence of float
    ax : matplotlib Axes, optional
    scale : float, optional
        Factor applied to the densities; by default the largest density is
        scaled to 0.8 times the smallest level spacing.
    potential : bool, optional
        Draw the real part of the potential.

    Returns
    -------
    lines : list of matplotlib Line2D
    """

    ax = plt.gca() if ax is None else ax
    densities = [_field(phi, "density") for phi in states]
    energies = np.asarray(energies, dtype=float)
    if scale is None:
        spacing = np.min(np.diff(np.sort(energies))) if len(energies) > 1 else 1.0
        scale = 0.8 * spacing / max(np.max(rho) for rho in densities)
    lines = []
    for n, (rho, E) in enumerate(zip(densities, energies, strict=True)):
        lines += ax.plot(grid.x_vals, scale * rho + E, label=f"$n = {n}$")
    if potential:
        ax.plot(grid.x_vals, np.real(grid.V), "k", lw=0.8)
    ax.set(xlabel="$x$", ylabel=r"$\vert\varphi_n\vert^2 + E_n$")
    return lines
