#-*- coding:utf-8 -*-

"""Analyzing tools for wavepacket dynamics.
"""

import numpy as np
from numpy.fft import fft, fftfreq, fftshift
from scipy.integrate import simpson
from scipy.signal.windows import hamming, hann, blackman
import matplotlib.pyplot as plt

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

    def __init__(self, xVals, psi0, V0):
        """Create and initialize an analyzer object.
        Information about the grid will be infered from the given data.

        Paramters
        ---------
        xVals : array
            The grid.
        psi0 : array
            Initial wavefunction.
        V0 : array
            Initial potential.
        """

        self._xGrid = xVals
        self._psi0 = psi0
        self._V0 = V0
        self._gridSamples = np.shape(V0)  # dimension(s) of grid array
        self._xExtent = np.zeros(len(self._gridSamples))
        self._dx = np.zeros(len(self._gridSamples))
        self._is1d = (np.size(self._gridSamples) == 1)

        # compute grid spacing and extent
        if(self._is1d):
            self._xExtent[0] = xVals[-1] - xVals[0]
            self._dx[0] = xVals[1] - xVals[0]
        else:
            self._xExtent[0] = xVals[0, -1] - xVals[0, 0]
            self._xExtent[1] = xVals[1, -1] - xVals[1, 0]
            self._dx[0] = xVals[0, 1] - xVals[0, 0]
            self._dx[1] = xVals[1, 1] - xVals[1, 0]

        # TODO: what's the difference between dx = (x_max - x_min)/N and dx = x_1 - x_0?

        self._tSamples = []
        self._psiSamples = []
        self._Vsamples = []

    def addData(self, psi, V, t):
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

        self._psiSamples.append(psi)
        self._Vsamples.append(V)
        self._tSamples.append(np.abs(t))  # absolute value fixes this for imaginary time evolution

    def processData(self):
        """Compute what has to be computed.
        Overwritten by derived classes.
        """

        raise NotImplementedError("Do not call this method of the *base* class.")


#-----------------------------------------------------------------------
# derived classes:
class AutoCorrelation(Analyzer):
    """Compute autocorrelation data and extract the spectrum.
    """

    def processData(self):
        """Compute overlap data computed from the wavefunctions given.
        """

        self._overlap = []
        for psi in self._psiSamples:
            if(self._is1d):
                overlap = simpson(self._psi0*np.conj(psi), dx=self._dx[0])  # simpson() works with complex-valued samples
                self._overlap.append(overlap)
            else:
                raise NotImplementedError("2d overlap integrals are not implemented yet!")  # TODO: implement

    def plotCorr(self):
        """Plot correlation function $C(t)$.
        """

        plt.plot(self._tSamples, np.abs(self._overlap), c="b", lw=2, ls="-")
        plt.title(r"$\vert C(t)\vert$")
        plt.show()

    def plotSpectrum(self, window="hamming"):
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

        if(window=="hann"):
            winFunc = hann
        elif(window=="hamming"):
            winFunc = hamming
        elif(window=="blackman"):
            winFunc = blackman
        else:
            raise ValueError("Illegal window option in plotSpectrum()!")

        dt = self._tSamples[1] - self._tSamples[0]  # assume equal dt for all steps
        Ntime = len(self._tSamples)
        freqs = 2*np.pi*fftshift(fftfreq(Ntime, dt))
        windowedSignal = np.asarray(self._overlap)*winFunc(Ntime)  # windowing -> make function periodic on time interval
        cOmega = np.abs(fftshift(fft(windowedSignal)))**2
        plt.plot(freqs, cOmega, c="b", lw=2, ls="-")
        plt.title(r"$\vert\tilde{C}(\omega)\vert^2$")
        plt.show()


class Observables(Analyzer):
    """Compute observables from data.
    """

    def processData(self, posExp=True, momExp=True):
        """Compute position and momentum expectation values.
        """

         # TODO: fix momenta. something is wrong here (units?)

        self._posExp = []; self._momExp = []

        if(momExp and self._is1d): kVals = 2*np.pi*fftshift(fftfreq(self._gridSamples[0],
                                                            self._dx[0]))  # *2pi means 1/lambda --> k = p

        for psi in self._psiSamples:
            if(self._is1d):
                if(posExp):
                    xExp = simpson(self._xGrid*np.abs(psi)**2, dx=self._dx[0])
                    self._posExp.append(xExp)
                if(momExp):  # compute in momentum space (where the momentum operator is just a multiplication)
                    Psi = fftshift(fft(psi))/np.sqrt(self._gridSamples[0])
                    # ensure correct *physical* norm of k-space wave function:
                    PsiNorm = simpson(abs(Psi)**2, x=kVals)
                    Psi /= np.sqrt(PsiNorm)
                    #print("|psi| = %s, |Psi| = %s"%(simpson(abs(psi)**2, x=self._xGrid), simpson(abs(Psi)**2, x=pVals)))
                    pExp = simpson(kVals*np.abs(Psi)**2, dx=abs(kVals[1]-kVals[0]))
                    self._momExp.append(pExp)
                    # TODO: maybe we can leave out the fftshift() call?
            else:
                raise NotImplementedError("2d expectation values are not implemented yet!")  # TODO: implement

    def plotData(self):
        """Plot available data.
        """

        if(len(self._posExp) > 0 and len(self._momExp) > 0):
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twinx()
            ax1.plot(self._tSamples, self._posExp, ls="-", c="b", lw=2)
            ax2.plot(self._tSamples, self._momExp, ls="--", c="r", lw=2)
            ax1.set_xlabel("$t$")
            ax1.set_ylabel(r"$\langle x\rangle$", color="b")
            ax2.set_ylabel(r"$\langle p\rangle$", color="r")
            for tl in ax1.get_yticklabels():
                tl.set_color('b')
            for tl in ax2.get_yticklabels():
                tl.set_color('r')


        if(len(self._posExp) > 0 and len(self._momExp) == 0):  # plot only position
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(self._tSamples, self._posExp, ls="-", c="b", lw=2)
            ax1.set_xlabel("$t$")
            ax1.set_ylabel(r"$\langle x\rangle$")

        if(len(self._posExp) == 0 and len(self._momExp) > 0):  # plot only momentum
            fig = plt.figure()
            ax1 = fig.add_subplot(111)
            ax1.plot(self._tSamples, self._momExp, ls="--", c="r", lw=2)
            ax1.set_xlabel("$t$")
            ax1.set_ylabel(r"$\langle p\rangle$")

        plt.show()


class EigenfunctionBuilder(Analyzer):
    """Construct an eigenstate with given energy from time propagation data.

    For the construction of eigenstates, compare
    Feit et al., J. Comp. Phys. 47, 412 (1982) and the excellent textbook
    "An introduction to time dependent Quantum Mechanics" by D. Tannor.
    """

    def processData(self):
        """Nothing to do here. Use computeEigenfunction() instead.
        """

        pass

    def computeEigenfunction(self, E, window="hamming"):
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

        if(window == "hann"):
            winFunc = hann
        elif(window == "hamming"):
            winFunc = hamming
        elif(window == "blackman"):
            winFunc = blackman
        else:
            raise ValueError("Illegal window option in processData()!")

        if(not self._is1d):
            raise NotImplementedError("Two dimensional computations are not implemented yet!")  # TODO: implement 2d

        integrands = np.zeros([len(self._psiSamples[0]), len(self._tSamples)], dtype=complex)
        windowArray = winFunc(len(self._tSamples))
        tSamples = np.asarray(self._tSamples)
        dt = tSamples[1] - tSamples[0]
        psiSamples = np.asarray(self._psiSamples)

        for i in range(len(self._tSamples)):  # build exp(i E_n t_n) psi_{x_n, t_n} array
            integrands[:, i] = np.exp(1j*E*tSamples[i])*psiSamples[i]*windowArray[i]

        return simpson(integrands, dx=dt)  # simpson() integrates the last axis, here: the *time* axis

    def plotEigenfunction(self, E):
        """Plot eigenfunction for energy E.

        Parameters
        ----------
        E : complex
            Energy E.
        """

        psiEigen = self.computeEigenfunction(E)

        plt.plot(self._xGrid, np.abs(psiEigen), lw=2, c="b", ls="-", label=r"$\vert\psi_E(x)\vert$ for $E=%s$"%E)
        plt.plot(self._xGrid, np.real(psiEigen), lw=1.5, c="red", ls="--", label=r"$\mathrm{Re}\,\psi_E(x)$ for $E=%s$"%E)
        plt.plot(self._xGrid, np.imag(psiEigen), lw=1.5, c="gray", ls="--", label=r"$\mathrm{Im}\,\psi_E(x)$ for $E=%s$"%E)
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

    def __init__(self, energies, tMax):
        """Create the filter.

        Parameters
        ----------
        energies : array-like
            Energies to filter for.
        tMax : float
            Total propagation time, needed for the window function.
        """

        self.energies = np.atleast_1d(np.asarray(energies, dtype=float))
        self.tMax = float(tMax)
        self._phi = None

    def addData(self, psi, t, dt):
        """Add the wavefunction at time t; dt is the time step between calls."""

        t = abs(t)
        if(t > self.tMax):
            return
        if(self._phi is None):
            self._phi = np.zeros((len(self.energies),) + np.shape(psi), dtype=complex)
        w = 0.5*(1 + np.cos(np.pi*t/self.tMax))*abs(dt)  # half of a Hann window
        for i, E in enumerate(self.energies):
            self._phi[i] += w*np.exp(1j*E*t)*psi

    def eigenstates(self, dV=1.0):
        """Return the filtered states, normalized for the volume element dV."""

        norms = np.sqrt(np.sum(np.abs(self._phi)**2, axis=tuple(range(1, self._phi.ndim)))*dV)
        return self._phi/norms.reshape((-1,) + (1,)*(self._phi.ndim - 1))
