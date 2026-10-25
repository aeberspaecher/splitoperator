#-*- coding:utf-8 -*-

#cython: profile=True
#cython: boundscheck=False
#cython: cdivision=True
#cython: wraparound=False

import cython

import numpy as np
cimport numpy as np

from potentials cimport *

from cpython cimport bool


#-----------------------------------------------------------------------
# Grids:
cdef class Grid(object):

    # number of dimensions, number of grid samples (same in each dimension):
    cdef int _d
    cdef int[2] _gridsamples
    # size list, potential func, initial wave packet object
    cdef object _size, _psi0
    cdef double _t  # current time
    cdef bool _updatePot  # does the potential have to be updated at each time step?
    cdef Potential1d _potential, _absorber # TODO: this 1d stuff will break 2d!

    cpdef suggestTimeStep(self, safety=?)


cdef class Grid1d(Grid):

    cdef object _x_extent
    # discretized absorber potential, wavefunction, potential
    cdef np.complex128_t[:] _absorber0, _psi, _pot
    cdef double[:] _xVals
    cdef double _dx

    cpdef potentialUpdate(self, double t)
    cpdef renormalizePsi(self)
    cpdef makePsiPeriodic(self)
