"""Initial wavepackets for time propagation.

Wavepackets are evaluated through ``psi(x)``, where ``x`` may be a scalar or a
NumPy array of positions.
"""

import logging

import numpy as np

__all__ = ["WavePacket", "WavepacketSuperposition", "Gaussian1d", "Gaussian1dEnergy", "CoherentState"]

log = logging.getLogger(__name__)


class WavePacket:
    """Base class for initial wavepackets.

    Derived classes implement ``psi(r)``, which must accept array-valued
    positions.
    """

    def psi(self, r):
        raise NotImplementedError("Do not call this in the base class!")

    def __add__(self, other):
        return WavepacketSuperposition([self, other])


class WavepacketSuperposition(WavePacket):
    """Create a superposition of several wave packets.
    """

    def __init__(self, psis):
        """Create a superposition of several wavepackets.

        Parameters
        ----------
        psis : list
            List of WavePacket objects to be added.
        """

        self._psis = list(psis)

    def psi(self, r):
        return sum(p.psi(r) for p in self._psis) + 0.0j


class Gaussian1d(WavePacket):
    """A Gaussian wavepacket in one dimension created from momentum and
    position space width.
    """

    def __init__(self, x0, sigma, p0):
        r"""Create a Gaussian wavepacket
        $\psi(x) = N \exp(-(x-x_0)^2/(2\sigma^2) + i p_0 (x-x_0))$.

        Parameters
        ----------
        x0 : float
            Center position.
        sigma : float
            Position space width.
        p0 : float
            Momentum of corresponding classical particle.
        """

        self.x0 = float(x0)
        self.sigma = float(sigma)
        self.p0 = float(p0)
        log.info("Created a Gaussian wavepacket with kinetic energy of T = p_0^2/2 = %s", self.p0**2/2)

    @property
    def sigma(self):
        """Standard deviation."""
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = float(value)
        self.N = np.sqrt(1.0/(np.sqrt(np.pi)*self._sigma))

    def psi(self, x):
        x = np.asarray(x)
        return self.N*np.exp(-(x - self.x0)**2/(2.0*self._sigma**2) + 1j*self.p0*(x - self.x0))


class Gaussian1dEnergy(Gaussian1d):
    """A Gaussian wavepacket in one dimension created from energy parameters.
    """

    def __init__(self, x0, E0, deltaE, direction=+1):
        """Create a Gaussian wavepacket from energy parameters.

        Parameters
        ----------
        x0 : float
            Center position.
        E0 : float
            Center energy.
        deltaE : float
            Full width half maximum of the Gaussian in energy space. Using
            this and E0, sigma_x will be derived.
        direction : float
            Positive for propagation to the right, negative for propagation to
            the left.

        Notes
        -----
        The narrower the Gaussian in energy space (~ momentum space), the
        broader the Gaussian will be in position space. However, choosing a
        broad energy distribution will lead to high dispersion.

        Journal reference for formulas:
        Hussain and Roberts, Phys. Rev. A 63, 012703 (2000)
        """

        E0, deltaE = float(E0), float(deltaE)
        sigma = 2*np.sqrt(2*np.log(2)*E0)/deltaE
        p0 = np.sign(direction)*np.sqrt(2*E0)
        super().__init__(x0, sigma, p0)
        log.info("Created a Gaussian wavepacket with energy E = %s, FWHM deltaE = %s", E0, deltaE)


class CoherentState(Gaussian1d):
    """A Gaussian wavepacket that does not spread in harmonic potentials.
    """

    def __init__(self, x0, omega, p0):
        r"""Create a coherent state for an oscillator of frequency $\omega$. The
        momentum can also be specified.

        Parameters
        ----------
        x0 : float
            Center position.
        omega : float
            Harmonic oscillator frequency (as in $V=\omega^2 x^2 / 2$).
        p0 : float
            Momentum of corresponding classical particle.
        """

        super().__init__(x0, np.sqrt(1/float(omega)), p0)
        log.info("Created a coherent state for omega = %s", float(omega))
