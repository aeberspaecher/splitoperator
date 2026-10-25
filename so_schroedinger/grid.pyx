#-*- coding:utf-8 -*-

#cython: profile=True
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

import cython

import numpy as np
cimport numpy as np
from scipy.integrate import simps
from numpy.fft import fftfreq

from cpython cimport bool

from potentials cimport *

np.import_array()

# list of exported names:
__all__ = ["Grid", "Grid1d"]


#-----------------------------------------------------------------------
# Grids:
cdef class Grid(object):
    """A grid object holds discrete representations of the wavepacket and the
    potential at a given time t. It also knows about the initital
    wavepacket. """

    property updatePot:
        """Is updating the potential necessary?"""

        def __get__(self):
            return self._updatePot

        def __set__(self, value):
            self._updatePot = value

    property t:
        """Current time $t$."""

        def __get__(self):
            return self._t

        def __set__(self, value):
            self._t = <double> value

    property gridSamples:
        """Number of grid samples."""

        def __get__(self):
            if(self._d == 1):
                return self._gridsamples[0]
            else:
                return self._gridsamples[0], self._gridsamples[1]

    # derived classes shall also implement the properties psi, V
    # however, as the Cython details of storage, casting and returning are a
    # matter of dimension, these properties are not implemented in the base
    # class

    def __init__(self, dim, size, gridsamples, potential, psi0, absorber=None):
        """Create a grid.

        Parameters
        ----------
        dim : int
            Spatial dimension.
        size : array-like
            Extent of grid dimensions.
        gridsamples : array-like
            Number of discrete samples for each dimension.
        potential : Potential object
            The potential used in potential discretization.
        psi0 : WavePacket object
            Initial wavepacket object.
        absorber : Absorber object, optional
            Absorbing boundary object.
        """

        # catch integer values for gridsamples:
#        if(type(gridsamples) == int):
#            gridsamples = [gridsamples]

#        self._d = dim
#        self._size = size
#        self._gridsamples[0] = gridsamples[0]
#        if(self._d == 2):
#            self._gridsamples[1] = gridsamples[1]
#        self._potential = potential
        self._psi0 = psi0.psi
#        if(absorber is not None):
#            self._absorber = absorber
#        else:
#            self._absorber = None
#        self._t = 0.0
        self._updatePot = potential.isTimeDependent

        return

    cpdef suggestTimeStep(self, safety=3.0):
        """Suggested a time step for computation from a consideration of
        potential landscape elevation. The highest difference in total energy
        that is possible is $\Delta E = \Delta V$, which corresponds to
        $\delta\omega = \Delta V$ (atomic units!) or $\delta t = 2\pi/\Delta
        V$.

        Do not use this for tiny or vanishing potentials as e.g. in
        free-particle propagation.

        Parameters
        ----------
        safety : float, optional
            Divide computed dt by this number for safer propagtion.

        Returns
        -------
        dt : float
            Suggested time step.
        """

        return 2*np.pi/(safety*(np.max(np.real(self.V)) - np.min(np.real(self.V))))

    # TODO: implement data output - either here or in a seperate class


cdef class Grid1d(Grid):
    """One dimensional grid.
    """

    property psi:
        r"""Current wavefunction $\psi(t)$."""

        def __get__(self):
            return np.asarray(self._psi)

        def __set__(self, value):
            self._psi = value

    property V:
        """Current potential $V(t)$."""

        def __get__(self):
            return np.asarray(self._pot)

    property x_extent:
        """Size of x dimension."""

        def __get__(self):
            return self._x_extent

    property xVals:
        """Discrete space variable."""
        def __get__(self):
            return np.asarray(self._xVals)

    property psiNorm:
        """Norm $\vert\vert\psi\vert\vert$."""

        def __get__(self):
            cdef double norm = np.sqrt(simps(np.abs(self.psi)**2, dx=self._dx))
            return norm

    def __init__(self, size, gridsamples, potential, psi0, absorber=None):
        cdef int i
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] potTemp = np.zeros(gridsamples, dtype=np.complex)
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] absorberTemp = np.zeros(gridsamples, dtype=np.complex)

        if(type(gridsamples) == int):
            gridsamples = [gridsamples]

        Grid.__init__(self, 1, size, gridsamples, potential, psi0, absorber)
        self._gridsamples[0] = int(gridsamples[0])
        self._potential = potential
        print gridsamples
        self._xVals = np.linspace(size[0], size[1], gridsamples[0], endpoint=True)
        self._x_extent = float(size[1]) - float(size[0])
        self._dx = self._xVals[1] - self._xVals[0]

        # create discrete wavefunction and potential
        self._psi = np.zeros(gridsamples, dtype=np.complex)
        self._pot = np.zeros(gridsamples, dtype=np.complex)
        self._absorber0 = np.zeros(gridsamples, dtype=np.complex)

        # initialize potential and wavefunction
        for i in range(gridsamples[0]):
            potTemp[i] = self._potential.V(self._xVals[i], 0.0)
            if(self._absorber is not None):
                absorberTemp[i] = self._absorber.V(self._xVals[i], 0.0)
            self._psi[i] = self._psi0(self._xVals[i])

        self._pot = potTemp + absorberTemp
        self._absorber0 = absorberTemp

        print("Maximum potential (real) value on grid V_max = %s"%np.max(np.real(self._pot)))
        print("Minimum potential (real) value on grid V_min = %s"%np.min(np.real(self._pot)))
        print("Maximum kinetic energy supported by grid T_max = %s"%np.max(fftfreq(self._gridsamples[0], self._x_extent/self._gridsamples[0])**2/2.0))

    @cython.initializedcheck(False)
    cpdef potentialUpdate(self, double t):
        """Update potential for time $t$.
        """

        cdef int i

        for i in range(self._gridsamples[0]):
            self._pot[i] = self._potential.V(self._xVals[i], t)
            if(self._absorber is not None):
                self._pot[i] = self._pot[i] + self._absorber0[i]

    cpdef renormalizePsi(self):
        """Renormalize $\psi$ such that $||\psi|| = 1$.
        """

        cdef np.ndarray[ndim=1, dtype=np.complex128_t] psiTemp = self.psi

        psiTemp /= self.psiNorm
        self._psi = psiTemp

    cpdef makePsiPeriodic(self):
        """Try avoid Gibbs' phenomenon by manually making the wavefunction
        periodic.
        """

        cdef np.complex128_t psiVal

        psiVal = (self._psi[0] + self._psi[self._gridsamples[0]-1])/2.0
        self._psi[0] = psiVal
        self._psi[self._gridsamples[0]-1] = psiVal
