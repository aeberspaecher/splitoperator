"""Analyzing tools for wavepacket dynamics."""

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from scipy.integrate import simpson
from scipy.signal.windows import blackman, hamming, hann

# FIXME: so far, all analyzers store their psi data individually.
# this eats a lot of memory.

# list of exported names:
__all__ = ["AutoCorrelation", "Observables", "EigenfunctionBuilder", "EigenstateFilter"]


class Analyzer:
    """Analyze wavefunctions and compute quantities derived from
    wavefunctions.

    This class defines a method allows adding a wavefunction and a potential
    at a given time. Everything that happens with this data is a matter of
    subclasses.
    """

    def __init__(self, x_vals, psi0, V0):
        """Create and initialize an analyzer object.
        Information about the grid will be infered from the given data.

        Paramters
        ---------
        x_vals : array
            The grid.
        psi0 : array
            Initial wavefunction.
        V0 : array
            Initial potential.
        """

        self._x_grid = x_vals
        self._psi0 = psi0
        self._V0 = V0
        self._grid_samples = np.shape(V0)  # dimension(s) of grid array
        self._x_extent = np.zeros(len(self._grid_samples))
        self._dx = np.zeros(len(self._grid_samples))
        self._is1d = np.size(self._grid_samples) == 1

        # compute grid spacing and extent
        if self._is1d:
            self._x_extent[0] = x_vals[-1] - x_vals[0]
            self._dx[0] = x_vals[1] - x_vals[0]
        else:
            self._x_extent[0] = x_vals[0, -1] - x_vals[0, 0]
            self._x_extent[1] = x_vals[1, -1] - x_vals[1, 0]
            self._dx[0] = x_vals[0, 1] - x_vals[0, 0]
            self._dx[1] = x_vals[1, 1] - x_vals[1, 0]

        # TODO: what's the difference between dx = (x_max - x_min)/N and dx = x_1 - x_0?

        self._t_samples = []
        self._psi_samples = []
        self._V_samples = []

    def add_data(self, psi, V, t):
        r"""Add data at a given instant.

        Parameters
        ----------
        psi : array
            Discrete wavefunction $\psi(r, t)$.
        V : array
            Discrete potential $V(r, t)$
        t : float
            Time.
        """

        self._psi_samples.append(psi)
        self._V_samples.append(V)
        self._t_samples.append(np.abs(t))  # absolute value fixes this for imaginary time evolution

    def process_data(self):
        """Compute what has to be computed.
        Overwritten by derived classes.
        """

        raise NotImplementedError("Do not call this method of the *base* class.")


# -----------------------------------------------------------------------
# derived classes:
class AutoCorrelation(Analyzer):
    """Compute autocorrelation data and extract the spectrum."""

    def process_data(self):
        """Compute overlap data computed from the wavefunctions given."""

        self._overlap = []
        for psi in self._psi_samples:
            if self._is1d:
                overlap = simpson(
                    self._psi0 * np.conj(psi), dx=self._dx[0]
                )  # simpson() works with complex-valued samples
                self._overlap.append(overlap)
            else:
                raise NotImplementedError("2d overlap integrals are not implemented yet!")  # TODO: implement

    def plot_corr(self):
        """Plot correlation function $C(t)$."""

        plt.plot(self._t_samples, np.abs(self._overlap), c="b", lw=2, ls="-")
        plt.title(r"$\vert C(t)\vert$")
        plt.show()

    def plot_spectrum(self, window="hamming"):
        """Plot Fourier transform of $C(t)$.

        As the time series is not periodic in any way, windowing is done before
        the Fourier transform is computed.

        Parameters
        ----------
        window : string, optional
            "hann", "hamming" or "blackman" for Hann, Hamming or Blackman
            window functions, respevtively. Defaults to the Hamming window.
        """

        # TODO: what about complex frequencies?

        if window == "hann":
            win_func = hann
        elif window == "hamming":
            win_func = hamming
        elif window == "blackman":
            win_func = blackman
        else:
            raise ValueError("Illegal window option in plot_spectrum()!")

        dt = self._t_samples[1] - self._t_samples[0]  # assume equal dt for all steps
        n_time = len(self._t_samples)
        freqs = 2 * np.pi * fftshift(fftfreq(n_time, dt))
        windowed_signal = np.asarray(self._overlap) * win_func(
            n_time
        )  # windowing -> make function periodic on time interval
        c_omega = np.abs(fftshift(fft(windowed_signal))) ** 2
        plt.plot(freqs, c_omega, c="b", lw=2, ls="-")
        plt.title(r"$\vert\tilde{C}(\omega)\vert^2$")
        plt.show()


class Observables(Analyzer):
    """Compute observables from data."""

    def process_data(self, pos_exp=True, mom_exp=True):
        """Compute position and momentum expectation values."""

        # TODO: fix momenta. something is wrong here (units?)

        self._pos_exp = []
        self._mom_exp = []

        if mom_exp and self._is1d:
            k_vals = 2 * np.pi * fftshift(fftfreq(self._grid_samples[0], self._dx[0]))  # *2pi means 1/lambda --> k = p

        for psi in self._psi_samples:
            if self._is1d:
                if pos_exp:
                    x_exp = simpson(self._x_grid * np.abs(psi) ** 2, dx=self._dx[0])
                    self._pos_exp.append(x_exp)
                if mom_exp:  # compute in momentum space (where the momentum operator is just a multiplication)
                    Psi = fftshift(fft(psi)) / np.sqrt(self._grid_samples[0])
                    # ensure correct *physical* norm of k-space wave function:
                    Psi_norm = simpson(abs(Psi) ** 2, x=k_vals)
                    Psi /= np.sqrt(Psi_norm)
                    p_exp = simpson(k_vals * np.abs(Psi) ** 2, dx=abs(k_vals[1] - k_vals[0]))
                    self._mom_exp.append(p_exp)
                    # TODO: maybe we can leave out the fftshift() call?
            else:
                raise NotImplementedError("2d expectation values are not implemented yet!")  # TODO: implement

    def plot_data(self):
        """Plot available data."""

        if len(self._pos_exp) > 0 and len(self._mom_exp) > 0:
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(self._t_samples, self._pos_exp, ls="-", c="b", lw=2)
            ax2.plot(self._t_samples, self._mom_exp, ls="--", c="r", lw=2)
            ax1.set_xlabel("$t$")
            ax1.set_ylabel(r"$\langle x\rangle$", color="b")
            ax2.set_ylabel(r"$\langle p\rangle$", color="r")
            for tl in ax1.get_yticklabels():
                tl.set_color("b")
            for tl in ax2.get_yticklabels():
                tl.set_color("r")

        if len(self._pos_exp) > 0 and len(self._mom_exp) == 0:  # plot only position
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(self._t_samples, self._pos_exp, ls="-", c="b", lw=2)
            ax1.set_xlabel("$t$")
            ax1.set_ylabel(r"$\langle x\rangle$")

        if len(self._pos_exp) == 0 and len(self._mom_exp) > 0:  # plot only momentum
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(self._t_samples, self._mom_exp, ls="--", c="r", lw=2)
            ax1.set_xlabel("$t$")
            ax1.set_ylabel(r"$\langle p\rangle$")

        plt.show()


class EigenfunctionBuilder(Analyzer):
    """Construct an eigenstate with given energy from time propagation data.

    For the construction of eigenstates, compare
    Feit et al., J. Comp. Phys. 47, 412 (1982) and the excellent textbook
    "An introduction to time dependent Quantum Mechanics" by D. Tannor.
    """

    def process_data(self):
        """Nothing to do here. Use compute_eigenfunction() instead."""

        pass

    def compute_eigenfunction(self, E, window="hamming"):
        r"""Build the eigenfunction that belongs to the energy $E$.

        Parameters
        ----------
        E : complex
            Use a phase $c = \exp(i E_n t)$ to build the eigenstate the belongs
            to the energy given by $E$.
        window : string, optional
            "hann", "hamming" or "blackman" for Hann, Hamming or Blackman
            window functions, respevtively. Defaults to the Hamming window.

        Returns
        -------
        psi_n : complex array
            Eigenfunction estimate for the energy given.
        """

        if window == "hann":
            win_func = hann
        elif window == "hamming":
            win_func = hamming
        elif window == "blackman":
            win_func = blackman
        else:
            raise ValueError("Illegal window option in process_data()!")

        if not self._is1d:
            raise NotImplementedError("Two dimensional computations are not implemented yet!")  # TODO: implement 2d

        integrands = np.zeros([len(self._psi_samples[0]), len(self._t_samples)], dtype=complex)
        window_array = win_func(len(self._t_samples))
        t_samples = np.asarray(self._t_samples)
        dt = t_samples[1] - t_samples[0]
        psi_samples = np.asarray(self._psi_samples)

        for i in range(len(self._t_samples)):  # build exp(i E_n t_n) psi_{x_n, t_n} array
            integrands[:, i] = np.exp(1j * E * t_samples[i]) * psi_samples[i] * window_array[i]

        return simpson(integrands, dx=dt)  # simpson() integrates the last axis, here: the *time* axis

    def plot_eigenfunction(self, E):
        """Plot eigenfunction for energy E.

        Parameters
        ----------
        E : complex
            Energy E.
        """

        psi_eigen = self.compute_eigenfunction(E)

        plt.plot(self._x_grid, np.abs(psi_eigen), lw=2, c="b", ls="-", label=rf"$\vert\psi_E(x)\vert$ for $E={E}$")
        plt.plot(
            self._x_grid, np.real(psi_eigen), lw=1.5, c="red", ls="--", label=rf"$\mathrm{{Re}}\,\psi_E(x)$ for $E={E}$"
        )
        plt.plot(
            self._x_grid,
            np.imag(psi_eigen),
            lw=1.5,
            c="gray",
            ls="--",
            label=rf"$\mathrm{{Im}}\,\psi_E(x)$ for $E={E}$",
        )
        plt.legend()
        plt.show()


class EigenstateFilter:
    r"""Build eigenstates for given energies on the fly, without storing the
    wavefunction at every time step (needed in 2d).

    For each energy $E$ the filter accumulates
    $\varphi_E = \int_0^{T} w(t)\, e^{iEt}\,\psi(t)\,\mathrm{d}t$,
    with a window function $w$ that decays from 1 at $t = 0$ to 0 at
    $t = T$. Cmp. Feit et al., J. Comp. Phys. 47, 412 (1982).
    """

    def __init__(self, energies, t_max):
        """Create the filter.

        Parameters
        ----------
        energies : array-like
            Energies to filter for.
        t_max : float
            Total propagation time, needed for the window function.
        """

        self.energies = np.atleast_1d(np.asarray(energies, dtype=float))
        self.t_max = float(t_max)
        self._phi = None

    def add_data(self, psi, t, dt):
        """Add the wavefunction at time t; dt is the time step between calls."""

        t = abs(t)
        if t > self.t_max:
            return
        if self._phi is None:
            self._phi = np.zeros((len(self.energies),) + np.shape(psi), dtype=complex)
        w = 0.5 * (1 + np.cos(np.pi * t / self.t_max)) * abs(dt)  # half of a Hann window
        for i, E in enumerate(self.energies):
            self._phi[i] += w * np.exp(1j * E * t) * psi

    def eigenstates(self, dV=1.0):
        """Return the filtered states, normalized for the volume element dV."""

        norms = np.sqrt(np.sum(np.abs(self._phi) ** 2, axis=tuple(range(1, self._phi.ndim))) * dV)
        return self._phi / norms.reshape((-1,) + (1,) * (self._phi.ndim - 1))
