#-*- coding:utf-8 -*-

#cython: profile=False
#cython: boundscheck=False
#cython: cdivision=True

from __future__ import division
import numpy as np
cimport numpy as np
from scipy.interpolate import BarycentricInterpolator, interp1d
from anfft import fft, ifft
from numpy.fft import fftshift, ifftshift

np.import_array()
from libc.math cimport sin, cos, log, exp, abs, cosh, tanh, pow
from cpython cimport bool

cdef double pi = 3.1415926535897932384626433

# list of exported names:
__all__ = ["Potential1d", "PotentialSuperposition1d", "ConstantPotential", "BoxPotential", "SmoothedBoxPotential",
           "WellPotential1d", "SquareWell1d", "MicrocyvityRadialPotential", "EckartPotential1d",
           "GaussBarrier1d", "Ramp1d", "MovingRamp1d", "HarmOscPot", "InterpolatingPotential1d",
           "RandomPotential1d", "ExpAbsorber1d"]


#-----------------------------------------------------------------------
# Potentials
cdef class Potential(object):
    r"""Base class for potentials. All potentials should inherit from this
    class.

    All derived classes must call the constructor Potential.__init__() *or*
    set the property isTimeDependent themselves. Furthermore, the dervied
    classes either have to set the member function _V(r, t) (read by the base
    class' V(r, t) or overwrite V(r, t) directly.
    """

    property isTimeDependent:
        """True if the potential has a time dependence."""

        def __get__(self):
            return self._isTimeDependent

        def __set__(self, isTimeDep):
            self._isTimeDependent = isTimeDep

    def __init__(self):
        self._isTimeDependent = True  # unless overwritten by subclass

    def V(self, r, t):
        """Potential V(r,t).
        """

        raise NotImplementedError("Call V() in dervived classes!")

cdef class Potential1d(Potential):
    """Potentials in 1d.
    """

    cpdef complex V(self, double x, double t):
        raise NotImplementedError("Call V() in dervived classes!")

cdef class PotentialSuperposition1d(Potential1d):
    """Create a superposition of several potentials.
    """

    def __init__(self, pots):
        r"""Create a superposition of several potentials.

        Parameters
        ----------
        pots : list
            List of potentials to be added.
        """

        Potential.__init__(self)

        self._N = len(pots)
        self._pots = pots
        isTimeDep = False
        for i in range(self._N):
            isTimeDep = isTimeDep or self._pots[i].isTimeDependent
        self._isTimeDependent = isTimeDep

    cpdef complex V(self, double r, double t):
        cdef int i
        cdef complex pot = 0.0 + 0.0j
        for i in range(self._N):
            pot += self._pots[i].V(r, t)

        return pot


cdef class ConstantPotential(Potential1d):
    """Constant Potential for free particle propagation.
    """

    def __init__(self, V0):
        """Create zero potential.
        """

        self._V0 = V0
        Potential.__init__(self)
        self._isTimeDependent = False

    cpdef complex V(self, double r, double t):
        return self._V0


cdef class BoxPotential(Potential1d):
    """A box potential of given position, width and height.
    """

    def __init__(self, pos, width, height):
        """Create a box potential in one dimension.

        Parameters
        ----------
        pos : float
            Center position.
        width : float
            Width of box.
        height : float
            Height of the box (may be negative).
        """

        Potential.__init__(self)
        self._pos, self._width, self._height = float(pos), float(width), float(height)
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._height if (abs(x - self._pos) <= self._width) else 0.0


cdef class SmoothedBoxPotential(Potential1d):
    r"""A smoothed box potential of given position, width and height.
    The potential is V = H*exp(-(|x_0 - x|/b)^N).
    """

    def __init__(self, pos, width, height, exponent=18):
        """Create smoothed box potential in 1d.

        Parameters
        ----------
        pos : float
            Center position.
        width : float
            Width of box.
        height : float
            Height of the box (may be negative).
        exponent : integer
            Integer exponent that determines the slope of the box' side walls.
        """

        Potential.__init__(self)
        self._pos, self._width, self._height = float(pos), float(width), float(height)
        self._exp = int(exponent)
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._height*exp(-pow((abs(self._pos-x)/self._width), self._exp))


cdef class WellPotential1d(Potential1d):
    """A quantum well potential.
    """

    def __init__(self, pos, width, V0, Vwall, d=-1.0, Vext=0.0):
        r"""Create a (symmetric) quantum well potential of given position,
        width, well potential, wall "height", and, optionally, wall thickness.

        Parameters
        ----------
        pos : float
            center coordinate of well
        width : float
            width of well
        V0 : float
            Potential inside the well.
        Vwall : float
            Wall potential
        d : float, optional
            Wall thickness. Negative values correspond to inifinte thickness.
        Vext : float, optional
            Potential beyond the walls of the well. Relevant only for finite
            wall thickness.
        """

        self._x0, self._width = float(pos), float(width)
        self._V0, self._Vwall, self._Vext = float(V0), float(Vwall), float(Vext)
        self._d = float(d)
        self._isTimeDependent = False

    cpdef complex V(self, double r, double t):
        """Well potential at r, t.
        """

        if(abs(r - self._x0) <= self._width/2.0):
            return self._V0
        elif(self._d > 0.0):  # finite walls
            if(abs(r - self._x0) <= self._width/2.0 + self._d):  # inside the wall
                return self._Vwall
            else:  # outside wall
                return self._Vext
        else:
            return self._Vwall


cdef class SquareWell1d(Potential1d):
    """A "square well" (parabola continued by constant potentials to each side).
    Harmonic potential inside, flat potential outside.
    """

    # TODO: improve docstring above

    def __init__(self, x0, omega, width, V0, well=True):
        r"""Create a "square well" potential.

        Parameters
        ----------
        x0 : float
            Center position.
        omega : float
            Harmonic oscillator frequency.
        width : float
            Width a the well.
        V0 : float
            Potential at the crest of the parabola. The outside potential is
            determined by the width. It is chosen such that that V is continous.
        well : boolean
            If True, the outide potential is higher than the crest potential,
            thus, a well is formed. If False, a bump potential is created.

        Note
        ----
        If you want an oscillator frequency that gives you a specific V_outside
        for given width, x0 and V0, use this formula:
        omega = sqrt(2(V_1 - V_0)/sign*(x_0 + d/2)^2).
        """

        Potential.__init__(self)
        self._x0, self._omega, self._width = float(x0), float(omega), float(width)
        self._V0, self._parabSign = float(V0), +1.0 if well else -1.0
        self._V1 = self._parabSign*0.5*self._omega**2*(x0+width/2.0)**2 + self._V0
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        if(abs(x-self._x0) <= self._width/2.0):
            return 0.5*self._parabSign*self._omega**2*(x-self._x0)**2 + self._V0
        else:
            return self._V1


cdef class MicrocyvityRadialPotential(Potential1d):
    """The radial potential as used in microcavity optics of the circle.
    """

    def __init__(self, R, n, k, m):
        r"""Create a microcavity style radial potential $V = k^2(1 - n^2(r))
        + m^2/r^2$. A refractive index of $n$ is assumed inside the cavity,
        whereas air ($n=1$) is assumed on the outside.

        Parameters
        ----------
        R : float
            Cavity radius.
        n : float
            Refractive index of semiconductor.
        k : float
            Wavenumber (acts like a shape parameter).
        m : int
            Rotational mode number.
        """

        Potential.__init__(self)
        self._R, self._n, self._k = float(R), float(n), float(k)
        self._mSquare = float(int(m))**2
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        if(x < self._R):
            return pow(self._k, 2)*(1.0 - pow(self._n, 2)) + self._mSquare/x**2
        else:
            return self._mSquare/x**2


cdef class EckartPotential1d(Potential1d):
    """The Eckart potential barrier.
    """

    def __init__(self, x0, V0, alpha):
        r"""Create an Eckart potential $V = V_0 \sech(\alpha (x-x_0)^2)$.

        Parameters
        ----------
        x0 : float
            Center position of the Gaussian bump.
        V0 : float
            Height of the bump.
        alpha : float
            Shape parameter.

        Notes
        -----
        Reference: Eckart, Phys. Rev. 35, 1303 (1930).
        """

        self._x0, self._V0, self._alpha = float(x0), float(V0), float(alpha)
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._V0/cosh(self._alpha*pow(x-self._x0, 2))


cdef class GaussBarrier1d(Potential1d):
    r"""A Gaussian bump in $1d$.
    """

    def __init__(self, x0, V0, sigma):
        r"""Create a Gaussian bump $V(x) = V_0 \exp(-(x-x_0)^2/2\sigma^2)$.

        Parameters
        ----------
        x0 : float
            Center position of the Gaussian bump.
        V0 : float
            Height of the bump.
        sigma : float
            Width or standard deviation of the potential bump.
        """

        self._x0, self._V0, self._sigmaSquare = float(x0), float(V0), float(sigma)**2
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._V0*exp(-pow(x-self._x0, 2)/(2*self._sigmaSquare))


cdef class Ramp1d(Potential1d):
    """A ramp potential.
    """

    def __init__(self, x0, V0, alpha, uphill):
        r"""Create a ramp potential $V = V_0/2 (1 + \tanh((x-x_0)/(2\alpha)))$.

        Parameters
        ----------
        x0 : float
            Center position of the ramp.
        V0 : float
            Height of the ramp.
        alpha : float
            Shape parameter. The smaller the value, the steeper the ramp.
        uphill : boolean
            If True, the potential is uphill. Is False, it goes "downhill".

        Notes
        -----
        A Bohmian investigation of this potential can be found in
        Poirier, J. Chem. Phys. 124, 034116 (2006).
        """

        self._x0, self._V0, self._alpha = float(x0), float(V0), float(alpha)
        self._sign = +1.0 if uphill else -1.0
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._V0*(1 + tanh(self._sign*(x-self._x0)/(2*self._alpha)))


cdef class MovingRamp1d(Potential1d):
    """A moving ramp potential.
    """

    def __init__(self, x0, V0, alpha, uphill,
                 motionType, frequency, motionAmpl,
                 phase):
        r"""Create a moving ramp potential $V = V_0/2 (1 +
        \tanh((x-x_0(t))/(2\alpha)))$.

        Use this to construct a fermi accelerator.

        Parameters
        ----------
        x0 : float
            Center position of the ramp.
        V0 : float
            Height of the ramp.
        alpha : float
            Shape parameter. The smaller the value, the steeper the ramp.
        uphill : boolean
            If True, the potential is uphill. Is False, it goes "downhill".
        motionType : string
            "sin" for $sin(\omega t)$ time dependence, "sawtooh" for
            sawtooth-like time dependence.
        frequency : float
            Frequency of oscillation.
        motionAmpl : float
            Amplitude of ramp position oscillation.
        phase : float
            Frequency of oscillation.
        """

        if(motionType != "sin"):
            raise NotImplementedError  # TODO: implement --> below!

        self._x0, self._V0, self._alpha = float(x0), float(V0), float(alpha)
        self._motionType = motionType
        self._omega, self._A, self._phi0 = float(frequency), float(motionAmpl), float(phase)
        self._sign = +1.0 if uphill else -1.0
        self._isTimeDependent = True

    cpdef complex V(self, double x, double t):

        cdef double x0 = self._x0 + self._A*sin(self._omega*t + self._phi0)
        return self._V0*(1 + tanh(self._sign*(x-x0)/(2*self._alpha)))


cdef class HarmOscPot(Potential1d):
    r"""Harmonic oscillator potential V = 1/2 omega_0 x^2.
    """

    def __init__(self, omega0):
        r"""Create harmonic oscillator  $V = \omega_0 x^2/2$ potential in 1d.

        Parameters
        ----------
        omega0 : float
            Harmonic oscillator frequency.
        """

        Potential.__init__(self)
        self._omega0 = float(omega0)
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return 0.5*self._omega0**2*x**2


cdef class InterpolatingPotential1d(Potential1d):
    """Interpolating polynomial potential defined by a set of points.
    """

    def __init__(self, points):
        r"""Create oscillator  $V=\omega_0x^2/2 + \omega_1x^4/4$ potential
        in 1d.

        Parameters
        ----------
        points : list
            List of tuples (x_i, y_i).
        """

        Potential.__init__(self)
        self._x = []
        self._y = []
        for point in points:
            self._x.append(point[0]); self._y.append(point[1]);
        self._interpolPoly = BarycentricInterpolator(self._x, self._y)
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._interpolPoly(x)


cdef class RandomPotential1d(Potential1d):
    r"""Create a random potential from a given autocorrelation function.
    """

    def __init__(self, corrFunc, x, V0):
        """Create random potential.

        Parameters
        ----------
        corrFunc : object
            Function object that can be evaluated for vector valued x
            (with x the grid values).
        x : array
            The grid values.
        """

        cdef int N
        cdef np.ndarray[ndim=1, dtype=np.complex128_t] tildeV

        Potential.__init__(self)
        self._isTimeDependent = False

        # Use the fact that the correlation function is the Fourier transform
        # of the power spectrum of the potential: for a given correlation
        # function, a random realization of a potential with the given
        # correlation can be generated by taking the Fourier transform of the
        # correlation function, randomizing the phases and transforming back.
        # This randomisation does not change anything in the *power* spectrum
        # (and thus also not in the correlation function), but will change
        # the potential in position space.

        N = len(x)
        corr = corrFunc(x)  # sample correlation function for the grid values
        spectralDensity = fftshift(fft(corr)/np.sqrt(N))  # use Wiener-Khinchin theorem
        tildeV = np.sqrt(spectralDensity)  # |FT| of the potential
        # randomize phases:
        randPhases = np.exp(1j*2*np.pi*np.random.rand(N/2))
        for i in range(N/2):
            tildeV[i] *= randPhases[i]
            tildeV[N/2+i] *= np.conj(randPhases[i])
        pot = np.abs(ifft(ifftshift(tildeV))*np.sqrt(N))
        print pot
        # normalize potential max(|V(x)|) = 1:
        pot /= np.max(np.abs(pot))
        self._Vinterpolating = interp1d(x, V0*pot)  # maximum amplitude is V0

    cpdef complex V(self, double x, double t):
        return self._Vinterpolating(x)


# absorbing potentials:
cdef class ExpAbsorber1d(Potential1d):
    """Absorning complex potential.

    A use complex potential with negative real part for better absorption
    (the negative real part will increase kinetic energy/decrease the
    de-Broglie wavelength which makes absorption easier).

    The potential allows placement of a real potential step for particle
    acceleration.

    Compare e.g. Hussain and Roberts, Phys. Rev. A 63, 012703 (2000).
    """

    def __init__(self, Areal, Aimag, lreal, limag, x0, x1, absorbLeft=True,
                 absorbRight=True):
        """Create absorbing complex potential in 1d.

        Parameters
        ----------
        Areal, Aimag : float
            Amplitude for complex potential.
        lreal : float
            Accelerating length of potential.
        limag  : float
            Absorbing length of potential.
        x0, x1 : float
            Outer points of spatial grid.
        absorbLeft, absorbRight : bool, optional
            Absorb on the left respectively right boundary of the grid.
        """

        Potential.__init__(self)
        self._isTimeDependent = False
        self._lreal, self._limag = float(lreal), float(limag)
        self._x0, self._x1 = float(x0), float(x1)
        self._Areal, self._Aimag = float(Areal), float(Aimag)
        self._alpha_r = log(999000)/self._lreal
        self._Aleft = 1.0 if absorbLeft else 0.0
        self._Aright = 1.0 if absorbRight else 0.0
        self._isTimeDependent = False

    cpdef complex V(self, double x, double t):
        return self._Areal*(1.0 - self._Aleft/(1.0 + exp(-self._alpha_r*(x - (self._x0+self._lreal))))  \
                            + self._Aright/(1.0 + exp(-self._alpha_r*(x - (self._x1-self._lreal)))) ) \
               + 1j*self._Aimag*(self._Aleft*exp(-abs(x-self._x0)/self._limag) + self._Aright*exp(-abs(x-self._x1)/self._limag))
