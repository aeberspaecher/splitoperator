r"""Analyze wavepacket dynamics during time propagation.

An analyzer collects data from a grid at the times it is called with that
grid. Every analyzer is callable as ``analyzer(grid)``, so it can be passed
directly as ``callback`` to ``SplitOperator.propagate()`` or in the
``analyzers`` list of the animation functions:

>>> obs = Observables()
>>> obs(grid)  # record the initial state
>>> prop.propagate(grid, dt, n_steps, callback=obs, every=10)
>>> obs.t, obs.position

All analyzers work on 1d and 2d grids. Time integrals use the trapezoidal
rule on the sample times, which need not be equidistant (except for
``AutoCorrelation.spectrum()``). Analyzers copy what they need, so the grid
may change after each call. None of them plots or prints anything.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from scipy.signal import get_window

# list of exported names:
__all__ = [
    "Analyzer",
    "Observables",
    "AutoCorrelation",
    "EigenstateFilter",
    "EigenfunctionBuilder",
    "IntegratedDensity",
    "Flux",
]


class Analyzer:
    """Base class: collects data from a grid each time it is called."""

    def add_data(self, grid):
        """Take data from the grid at its current time ``grid.t``."""

        raise NotImplementedError

    def __call__(self, grid):
        self.add_data(grid)


class _Trapezoid:
    """Trapezoidal weights for samples that arrive one at a time.

    The weight of a sample, $(t_{i+1} - t_{i-1})/2$, is known only once the
    next sample time is known, so each sample is held back by one call.
    """

    def __init__(self):
        self._t_before = None  # time of the sample before the pending one
        self._pending = None  # (t, item)

    def push(self, t, item):
        """Add a sample; return the previous ``(t, item, weight)`` or None."""

        done = None
        if self._pending is not None:
            t_pend, item_pend = self._pending
            t_before = t_pend if self._t_before is None else self._t_before
            done = (t_pend, item_pend, (t - t_before) / 2.0)
            self._t_before = t_pend
        self._pending = (t, item)
        return done

    def last(self):
        """The held back sample with its weight as last sample, or None."""

        if self._pending is None or self._t_before is None:
            return None
        t_pend, item_pend = self._pending
        return (t_pend, item_pend, (t_pend - self._t_before) / 2.0)


# -----------------------------------------------------------------------
class Observables(Analyzer):
    r"""Record norm and expectation values.

    After a series of calls, the attributes hold arrays with one entry per
    call:

    - ``t``: times
    - ``norm``: $\Vert\psi\Vert$ (decreases with absorbers)
    - ``position``: $\langle x\rangle$, in 2d $(\langle x\rangle, \langle y\rangle)$
    - ``momentum``: $\langle p\rangle$, in 2d $(\langle p_x\rangle, \langle p_y\rangle)$
    - ``width``: $\Delta x$, in 2d $(\Delta x, \Delta y)$

    In 2d, ``position``, ``momentum`` and ``width`` have shape ``(n, 2)``.
    """

    def __init__(self):
        self._data = {"t": [], "norm": [], "position": [], "momentum": [], "width": []}

    def add_data(self, grid):
        self._data["t"].append(grid.t)
        self._data["norm"].append(grid.psi_norm)
        if grid.dim == 1:
            self._data["position"].append(grid.expectation_x())
            self._data["width"].append(grid.width_x())
        else:
            self._data["position"].append(grid.expectation_r())
            self._data["width"].append(grid.width_r())
        self._data["momentum"].append(grid.expectation_p())

    def __len__(self):
        return len(self._data["t"])

    @property
    def t(self):
        """Sample times."""
        return np.array(self._data["t"])

    @property
    def norm(self):
        r"""$\Vert\psi\Vert$."""
        return np.array(self._data["norm"])

    @property
    def position(self):
        r"""$\langle x\rangle$, in 2d $(\langle x\rangle, \langle y\rangle)$."""
        return np.array(self._data["position"])

    @property
    def momentum(self):
        r"""$\langle p\rangle$, in 2d $(\langle p_x\rangle, \langle p_y\rangle)$."""
        return np.array(self._data["momentum"])

    @property
    def width(self):
        r"""$\Delta x$, in 2d $(\Delta x, \Delta y)$."""
        return np.array(self._data["width"])


class AutoCorrelation(Analyzer):
    r"""Record the autocorrelation function $C(t) = \langle\psi_0|\psi(t)\rangle$
    and compute the spectrum from it.

    Parameters
    ----------
    reference : array, optional
        $\psi_0$. Defaults to the wavefunction of the first call.
    """

    def __init__(self, reference=None):
        self._reference = None if reference is None else np.array(reference, dtype=complex)
        self._t = []
        self._C = []

    def add_data(self, grid):
        if self._reference is None:
            self._reference = grid.psi.copy()
        self._t.append(grid.t)
        self._C.append(np.sum(np.conj(self._reference) * grid.psi) * grid.dV)

    @property
    def t(self):
        """Sample times."""
        return np.array(self._t)

    @property
    def C(self):
        r"""$C(t)$ at the sample times."""
        return np.array(self._C)

    def spectrum(self, window="hann"):
        r"""Spectrum $|\tilde C(E)| = \left|\int_0^T w(t)\,e^{iEt}\,C(t)\,\mathrm{d}t\right|$.

        Expanding $\psi_0$ in eigenstates, $C(t) = \sum_n |c_n|^2 e^{-iE_n t}$,
        so the spectrum has peaks at the eigenenergies $E_n$, weighted by
        $|c_n|^2$; cmp. Feit et al., J. Comp. Phys. 47, 412 (1982). The
        resolution is about $2\pi/T$, the largest energy $\pi/\Delta t$.

        Parameters
        ----------
        window : str or tuple, optional
            Window $w(t)$, as understood by ``scipy.signal.get_window()``.
            Only its second half is used, which decays from 1 at $t = 0$ to 0
            at $t = T$ (since $C(-t) = C(t)^*$). Defaults to Hann.

        Returns
        -------
        E : array
            Energies, ascending.
        amplitude : array
            $|\tilde C(E)|$.
        """

        t = self.t
        n = len(t)
        if n < 2:
            raise ValueError("at least two samples are needed for a spectrum")
        dt = t[1] - t[0]
        if not np.allclose(np.diff(t), dt, rtol=1e-6, atol=0.0):
            raise ValueError("the spectrum needs equidistant sample times")
        w = get_window(window, 2 * n, fftbins=False)[n:]
        E = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(n, dt))
        amplitude = np.abs(np.fft.fftshift(np.fft.ifft(self.C * w))) * n * dt
        return E, amplitude


class EigenstateFilter(Analyzer):
    r"""Build eigenstates for given energies on the fly, without storing the
    wavefunction at every time step.

    For each energy $E$ the filter accumulates
    $\varphi_E = \int_0^{T} w(t)\, e^{iEt}\,\psi(t)\,\mathrm{d}t$,
    with half a Hann window $w(t) = (1 + \cos(\pi t/T))/2$ that decays from 1
    at $t = 0$ to 0 at $t = T$. Times are counted from the first call.
    Cmp. Feit et al., J. Comp. Phys. 47, 412 (1982).

    The energy resolution is about $2\pi/T$; states closer in energy are
    mixed.

    Parameters
    ----------
    energies : float or array-like
        Energies to filter for.
    t_max : float
        Filter time $T$. Data added later than $T$ after the first call is
        ignored.
    """

    def __init__(self, energies, t_max):
        self.energies = np.atleast_1d(np.asarray(energies, dtype=float))
        self.t_max = float(t_max)
        self._t0 = None
        self._dV = None
        self._phi = None
        self._trapezoid = _Trapezoid()

    def add_data(self, grid):
        self._add(grid.t, grid.psi, grid.dV)

    def _add(self, t, psi, dV):
        if self._t0 is None:
            self._t0 = t
            self._dV = dV
            self._phi = np.zeros((len(self.energies),) + np.shape(psi), dtype=complex)
        t = t - self._t0
        if t > self.t_max * (1 + 1e-9):
            return
        done = self._trapezoid.push(t, np.array(psi, dtype=complex))
        if done is not None:
            self._accumulate(self._phi, *done)

    def _accumulate(self, phi, t, psi, weight):
        w = 0.5 * (1 + np.cos(np.pi * t / self.t_max)) * weight
        for i, E in enumerate(self.energies):
            phi[i] += w * np.exp(1j * E * t) * psi

    def eigenstates(self):
        r"""Return the filtered states, normalized to $\Vert\varphi_E\Vert = 1$.

        Returns
        -------
        phi : complex array
            ``phi[i]`` belongs to ``energies[i]``.
        """

        if self._phi is None:
            raise ValueError("no data has been added yet")
        phi = self._phi.copy()
        last = self._trapezoid.last()
        if last is not None:
            self._accumulate(phi, *last)
        norms = np.sqrt(np.sum(np.abs(phi) ** 2, axis=tuple(range(1, phi.ndim))) * self._dV)
        return phi / norms.reshape((-1,) + (1,) * (phi.ndim - 1))


class EigenfunctionBuilder(Analyzer):
    r"""Store $\psi(t)$ to build eigenstates for energies chosen after the
    propagation, e.g. from the peaks of ``AutoCorrelation.spectrum()``.

    This needs memory for all stored wavefunctions; use ``EigenstateFilter``
    if the energies are known in advance. The filter is the same, with $T$
    the time between the first and the last call.
    """

    def __init__(self):
        self._t = []
        self._psi = []
        self._dV = None

    def add_data(self, grid):
        self._t.append(grid.t)
        self._psi.append(grid.psi.copy())
        self._dV = grid.dV

    def eigenstates(self, energies):
        """Filtered states for the given energies, see ``EigenstateFilter``.

        Parameters
        ----------
        energies : float or array-like

        Returns
        -------
        phi : complex array
            ``phi[i]`` belongs to ``energies[i]``.
        """

        if len(self._t) < 2:
            raise ValueError("at least two samples are needed")
        filt = EigenstateFilter(energies, self._t[-1] - self._t[0])
        for t, psi in zip(self._t, self._psi, strict=True):
            filt._add(t, psi, self._dV)
        return filt.eigenstates()


class IntegratedDensity(Analyzer):
    r"""Time integrated density $\int |\psi(\mathbf{r}, t)|^2\,\mathrm{d}t$.

    Shows where a wave has been, e.g. the branches of branched flow.
    """

    def __init__(self):
        self._acc = None
        self._trapezoid = _Trapezoid()

    def add_data(self, grid):
        if self._acc is None:
            self._acc = np.zeros(np.shape(grid.psi))
        done = self._trapezoid.push(grid.t, grid.density)
        if done is not None:
            self._acc += done[2] * done[1]

    @property
    def density(self):
        """The integrated density so far."""
        if self._acc is None:
            raise ValueError("no data has been added yet")
        last = self._trapezoid.last()
        return self._acc if last is None else self._acc + last[2] * last[1]


class Flux(Analyzer):
    r"""Probability flux through a point (1d) or a line (2d).

    The current density is $j = \mathrm{Im}(\psi^*\nabla\psi)$ (atomic
    units, $m = 1$), with the derivative taken spectrally. In 1d the flux is
    $j(x_0, t)$, in 2d the flux through the line $x = x_0$ (``axis=0``) or
    $y = y_0$ (``axis=1``). The time integral of the flux is the probability
    that has passed, e.g. the transmission through a barrier. Unlike the
    probability to the right of the barrier, it counts the transmitted part
    even after an absorber has swallowed it.

    The flux at a single point is sensitive to high momentum components,
    e.g. from discontinuous potentials. Call the analyzer at every time step
    and use a time step that resolves the relevant momenta; otherwise the
    time integral can be off by several percent.

    Parameters
    ----------
    position : float
        $x_0$ (or $y_0$). The nearest grid point (line) is used.
    axis : int, optional
        In 2d, the axis normal to the line.
    """

    def __init__(self, position, axis=0):
        self.position = float(position)
        self.axis = int(axis)
        self._t = []
        self._flux = []

    def add_data(self, grid):
        axis = 0 if grid.dim == 1 else self.axis
        if grid.dim == 1:
            coords, k, d = grid.x_vals, grid.k_vals, None
        elif axis == 0:
            coords, k, d = grid.x_vals, grid.kx_vals, grid.dy
        else:
            coords, k, d = grid.y_vals, grid.ky_vals, grid.dx
        i = np.argmin(np.abs(coords - self.position))
        psi = grid.psi
        k_shape = [1] * psi.ndim
        k_shape[axis] = -1
        dpsi = np.fft.ifft(1j * k.reshape(k_shape) * np.fft.fft(psi, axis=axis), axis=axis)
        j = np.imag(np.conj(np.take(psi, i, axis=axis)) * np.take(dpsi, i, axis=axis))
        self._t.append(grid.t)
        self._flux.append(j if d is None else np.sum(j) * d)

    @property
    def t(self):
        """Sample times."""
        return np.array(self._t)

    @property
    def flux(self):
        """Flux at the sample times; positive in the direction of the axis."""
        return np.array(self._flux)

    @property
    def transferred(self):
        """Probability that has passed so far, $\\int j\\,\\mathrm{d}t$."""
        return trapezoid(self.flux, self.t) if len(self._t) > 1 else 0.0

    def cumulative(self):
        """Probability that has passed, at each sample time."""
        return cumulative_trapezoid(self.flux, self.t, initial=0.0)
