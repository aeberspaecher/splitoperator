
__all__ = ["WavepacketSuperposition", "Gaussian1d", "Gaussian1dEnergy",
           "CoherentState"]

#-----------------------------------------------------------------------
# initial wavefunctions
cdef class WavePacket(object):

    cpdef complex psi(self, r)


cdef class WavepacketSuperposition(WavePacket):

       cpdef complex psi(self, r)


cdef class Gaussian1d(WavePacket):

    cdef double _x0, _sigma, _p0, _N
    cdef object _psi

    cpdef _recomputeN(self)

    cpdef complex psi(self, x)


cdef class Gaussian1dEnergy(Gaussian1d):

    pass

cdef class CoherentState(Gaussian1d):

    pass
