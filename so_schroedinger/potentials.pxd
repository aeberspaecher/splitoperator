#-*- coding:utf-8 -*-

#cython: profile=False
#cython: boundscheck=False
#cython: cdivision=True

import numpy as np
cimport numpy as np

np.import_array()
from cpython cimport bool

#-----------------------------------------------------------------------
# Potentials
cdef class Potential(object):

    cdef bool _isTimeDependent

cdef class Potential1d(Potential):

    cpdef complex V(self, double x, double t)

cdef class PotentialSuperposition1d(Potential1d):

    cdef int _N
    cdef list _pots

    cpdef complex V(self, double r, double t)


cdef class ConstantPotential(Potential1d):

    cdef double _V0

    cpdef complex V(self, double r, double t)

cdef class BoxPotential(Potential1d):

    cdef double _pos, _width, _height

    cpdef complex V(self, double x, double t)


cdef class SmoothedBoxPotential(Potential1d):

    cdef _pos, _width, _height
    cdef int _exp

    cpdef complex V(self, double x, double t)


cdef class WellPotential1d(Potential1d):

    cdef double _x0, _width, _V0, _Vwall, _d, _Vext

    cpdef complex V(self, double r, double t)


cdef class SquareWell1d(Potential1d):

    cdef double _x0, _omega, _width, _V0, _parabSign, _V1

    cpdef complex V(self, double x, double t)


cdef class MicrocyvityRadialPotential(Potential1d):

    cdef double _R, _n, _k, _mSquare

    cpdef complex V(self, double x, double t)


cdef class EckartPotential1d(Potential1d):

    cdef double _x0, _V0, _alpha

    cpdef complex V(self, double x, double t)


cdef class GaussBarrier1d(Potential1d):

    cdef double _x0, _V0, _sigmaSquare

    cpdef complex V(self, double x, double t)


cdef class Ramp1d(Potential1d):

    cdef double _x0, _V0, _alpha, _sign

    cpdef complex V(self, double x, double t)


cdef class MovingRamp1d(Potential1d):

    cdef double _x0, _V0, _alpha, _omega, _A, _phi0, _sign
    cdef object _motionType

    cpdef complex V(self, double x, double t)


cdef class HarmOscPot(Potential1d):

    cdef double _omega0

    cpdef complex V(self, double x, double t)


cdef class InterpolatingPotential1d(Potential1d):

    cdef object _x, _y, _interpolPoly

    cpdef complex V(self, double x, double t)


cdef class RandomPotential1d(Potential1d):

    cdef object _Vinterpolating

    cpdef complex V(self, double x, double t)


cdef class ExpAbsorber1d(Potential1d):

    cdef double _Areal, _Aimag, _lreal, _limag, _x0, _x1, _alpha_r, _Aleft, _Aright

    cpdef complex V(self, double x, double t)
