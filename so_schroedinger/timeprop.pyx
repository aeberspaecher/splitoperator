#-*- coding:utf-8 -*-

#cython: profile=True
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

"""Simple quantum time propagation in 1d/2d.

All physical quantities are in (Hartree) atomic units
(numerical values: hbar = e = m = k_B = 1; c=1/alpha), cmp. also
http://en.wikipedia.org/wiki/Atomic_units or
http://en.wikipedia.org/wiki/Natural_units#Atomic_units
"""

import numpy as np
cimport numpy as np
from anfft import *
from numpy.fft import fftfreq

from grid cimport *
from potentials cimport *

from libc.math cimport sqrt

# list of exported symbols:
__all__ = ["SplitOperator"]


#-----------------------------------------------------------------------
cdef class Propagator(object):
    """Base class for wavepacket time-propagtion.
    """

    cpdef step(self, Grid1d grid, complex dt):
        """To be implemented in subclasses.
        """

        raise NotImplementedError


cdef class SplitOperator(Propagator):
    """Split operator time propagation.
    """

    # TODO: implement large-interval propagation (saving half-step computations)
    # TODO: distinguish 1d/2d - and implement 2d

    cdef int _stepcount, _doRenormalize

    def __init__(self, renormalizeEveryNSteps=0):
        """Create a split operator propagator.

        Parameters
        ----------
        renormalizeEveryNSteps : int, optional
            Renormalize the wavefuntion after this many steps have been made.
            Use 0 if you do not want renormalization.
        """

        Propagator.__init__(self)
        self._stepcount = 0
        self._doRenormalize = int(renormalizeEveryNSteps)

    cpdef step(self, Grid1d grid, complex dt):
        """Advance time by one step dt.

        Parmaters
        ---------
        grid : Grid object
            The grid that holds all information needed to advance time.
        dt : complex
            Time step.
        """

        self._stepcount += 1

        if(grid.updatePot):
            grid.potentialUpdate(grid.t)
            # TODO: would this also be the right place to implement Gross-Pitaevskii-like nonlinearites?

        # use symmetrized version of the time propagator (error is O(dt^3)
        # instead of O(dt^2) for asymmetric variant):

        self._applyPotOperatorPart(grid, dt/2.0)
        self._applyKinOperatorPart(grid, dt)
        self._applyPotOperatorPart(grid, dt/2.0)

        grid.t = grid.t + np.abs(dt)  # the abs() call is need for imaginary time evolution

        # estimate ground state energy in case of imaginary time evolution,
        # formula from Koeberle, Dissertation (2011), p. 36 (derived by assuming
        # the evolution has converged to the ground state and then acting
        # with the imaginary time evolution operator on the again and then
        # computing the norm of this evolved ground state -- the loss of norm
        # here relates to ground state energy)
        if(np.imag(dt) < 0.0):
            print("Current ground state energy estimate E_0 = %s"%(np.log(grid.psiNorm)/np.imag(dt)))

        if(self._doRenormalize):
            if(self._stepcount % self._doRenormalize == 0):
                grid.renormalizePsi()

    cpdef _applyPotOperatorPart(self, Grid1d grid, complex dt):
        """Apply potential energy operator part of the propagator in position
        space (where the position operator is diagonal).
        """

        grid.psi = grid.psi*np.exp(-1j*grid.V*dt)
        # TODO: is this NumPy operation (involves array creation through the psi property) faster than a loop over the memoryview?

    cpdef _applyKinOperatorPart(self, Grid1d grid, complex dt):
        """Apply kinetic energy operator part of the propagation.
        Use momentum representation in which the momentum operator is diagonal.

        Fourier transform -> Apply kinetic energy operator part
        -> inverse Fourier transform.
        """

        cdef np.ndarray[ndim=1, dtype=np.complex128_t] Psi = fft(grid.psi)/sqrt(grid.gridSamples[0])  # transform to k-space
        cdef np.ndarray[ndim=1, dtype=double] kVals = 2*np.pi*fftfreq(grid.gridSamples[0], grid.x_extent/grid.gridSamples[0])
        Psi *= np.exp(-1j*kVals**2*dt/2.0)
        grid.psi = ifft(Psi)*sqrt(grid.gridSamples[0])  # go back to position space
        # TODO: can the fftfreq() call be done only once? do we need to store the grid then?
