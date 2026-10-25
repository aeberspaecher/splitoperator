#-*- coding:utf-8 -*-

#cython: profile=False
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

"""Initial wavepackets for time propagation.
"""

import numpy as np
cimport numpy as np
np.import_array()

__all__ = ["WavepacketSuperposition", "Gaussian1d", "Gaussian1dEnergy",
           "CoherentState"]


cdef class WavePacket(object):
    """Base class for initial wavepackets.

    Dervived classes need to take care that _psi(r, t) is set.
    """

    cpdef complex psi(self, r):
        raise NotImplementedError("Do not call this in the base class!")


cdef class WavepacketSuperposition(WavePacket):
    """Create a superposition of several wave packets.
    """

    def __init__(self, psis):
        """Create a superposition of several wavepackets.

        Parameters
        ----------
        pots : list
            List of WavePacket objects to be added.
        """

        self._N = len(psis)
        self._psis = psis

    cpdef complex psi(self, r):
        psi = 0.0j
        for i in range(self._N):
            psi += self._psis[i]._psi(r)

        return psi


cdef class Gaussian1d(WavePacket):
    """A Gaussian wavepacket in one dimension created from momentum and
    position space width.
    """

    property x0:
        """Center position."""

        def __get__(self):
            return self._x0
        def __set__(self, value):
            self._x0 = value

    property sigma:
        """Standard deviation."""

        def __get__(self):
            return self._sigma
        def __set__(self, value):
            self._sigma = value
            self._recomputeN()

    property N:
        """Normalization factor."""

        def __get__(self):
            return self._N
        def __set__(self, value):
            self._N = value

    property p0:
        """Center momentum."""

        def __get__(self):
            return self._p0

        def __set__(self, value):
            self._p0 = value

    cpdef _recomputeN(self):
        """Recompute normalization factor."""

        self._N = np.sqrt((1.0/(np.sqrt(np.pi)*self.sigma)))

    def __init__(self, x0, sigma, p0):
        """Create a Gaussian wavepacket.

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
        print("Created a Gaussian wavepacket with kinetic energy of T = p_0^2/2 = %s"%(self._p0**2/2))

    cpdef complex psi(self, x):
       return self._N * np.exp(-(x-self._x0)**2/(2.0*self._sigma**2) + 1j*self._p0*(x-self._x0))


cdef class Gaussian1dEnergy(Gaussian1d):
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
            Postive for propagtion to the right, negative for propagtion to
            the left.

        Notes
        -----
        The narrower the Gaussian in energy space (~ momentum space), the
        broader the Gaussian will be in position space. However, choosing a
        broad energy distribution will lead to high dispersion.

        Journal reference for formulas:
        Hussain and Roberts, Phys. Rev. A 63, 012703 (2000)
        """

        # convert energy quantities to "geometric" quantities and initialize
        # using the base class' constructor
        x0 = float(x0)
        E0 = float(E0)
        deltaE = float(deltaE)
        sigma = 2*np.sqrt(2*np.log(2)*E0)/deltaE
        p0 = np.sign(direction)*np.sqrt(2*E0)
        Gaussian1d.__init__(self, x0, sigma, p0) # TODO: fix!
        print("Created a Gaussian wavepacket with energy E = %s, FHWM deltaE = %s"%(E0, deltaE))


cdef class CoherentState(Gaussian1d):
    """A Gaussian wavepacket that does not spread in harmonic potentials.
    """

    # TODO: doc...

    def __init__(self, x0, omega, p0):
        """Create a coherent state for an oscillator of frequency $omega$. The
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

        # convert energy quantities to "geometric" quantities and initialize
        # using the base class' constructor
        x0 = float(x0)
        sigma = np.sqrt(1/float(omega))

        Gaussian1d.__init__(self, x0, sigma, p0)

        print("Created a coherent state for omega = %s"%(float(omega)))

    # TODO: learn about thermal Gaussian wavepackets
