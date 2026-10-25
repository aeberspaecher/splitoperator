"""Potentials for wavepacket time propagation.

All potentials are evaluated through ``V(x, t)``, where ``x`` may be a scalar
or a NumPy array of positions. The return value is complex: an imaginary part
acts as an absorber.
"""

import numpy as np
from scipy.interpolate import BarycentricInterpolator

# list of exported names:
__all__ = ["Potential", "Potential1d", "PotentialSuperposition", "PotentialSuperposition1d", "ConstantPotential", "BoxPotential",
           "SmoothedBoxPotential", "WellPotential1d", "SquareWell1d", "MicrocavityRadialPotential",
           "MicrocyvityRadialPotential", "EckartPotential1d", "GaussBarrier1d", "Ramp1d", "MovingRamp1d",
           "HarmOscPot", "InterpolatingPotential1d", "RandomPotential1d", "ExpAbsorber1d",
           "gaussian_random_field", "Potential2d", "HarmOscPot2d", "GaussBump2d", "StadiumPotential2d",
           "RandomPotential2d", "BoundaryAbsorber2d"]


#-----------------------------------------------------------------------
# Potentials
class Potential:
    r"""Base class for potentials. All potentials should inherit from this
    class.

    Derived classes set the attribute ``isTimeDependent`` and implement
    ``V(r, t)``, which must accept array-valued positions.
    """

    isTimeDependent = True  # unless overwritten by subclass

    def V(self, r, t):
        """Potential V(r,t).
        """

        raise NotImplementedError("Call V() in derived classes!")

    def __add__(self, other):
        return PotentialSuperposition1d([self, other])


class Potential1d(Potential):
    """Potentials in 1d.
    """


class PotentialSuperposition1d(Potential1d):
    """Create a superposition of several potentials.
    """

    def __init__(self, pots):
        r"""Create a superposition of several potentials.

        Parameters
        ----------
        pots : list
            List of potentials to be added.
        """

        self._pots = list(pots)
        self.isTimeDependent = any(pot.isTimeDependent for pot in self._pots)

    def V(self, r, t):
        return sum(pot.V(r, t) for pot in self._pots) + 0.0j


PotentialSuperposition = PotentialSuperposition1d  # works in any dimension


class ConstantPotential(Potential1d):
    """Constant Potential for free particle propagation.
    """

    isTimeDependent = False

    def __init__(self, V0):
        """Create constant potential V0.
        """

        self._V0 = V0

    def V(self, r, t):
        shape = np.shape(r[0]) if isinstance(r, tuple) else np.shape(r)
        return np.full(shape, self._V0, dtype=complex)


class BoxPotential(Potential1d):
    """A box potential of given position, width and height.
    """

    isTimeDependent = False

    def __init__(self, pos, width, height):
        """Create a box potential in one dimension.

        Parameters
        ----------
        pos : float
            Center position.
        width : float
            Half width of box.
        height : float
            Height of the box (may be negative).
        """

        self._pos, self._width, self._height = float(pos), float(width), float(height)

    def V(self, x, t):
        return np.where(np.abs(np.asarray(x) - self._pos) <= self._width, self._height, 0.0) + 0.0j


class SmoothedBoxPotential(Potential1d):
    r"""A smoothed box potential of given position, width and height.
    The potential is V = H*exp(-(|x_0 - x|/b)^N).
    """

    isTimeDependent = False

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

        self._pos, self._width, self._height = float(pos), float(width), float(height)
        self._exp = int(exponent)

    def V(self, x, t):
        return self._height*np.exp(-(np.abs(self._pos - np.asarray(x))/self._width)**self._exp) + 0.0j


class WellPotential1d(Potential1d):
    """A quantum well potential.
    """

    isTimeDependent = False

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
            Wall thickness. Negative values correspond to infinite thickness.
        Vext : float, optional
            Potential beyond the walls of the well. Relevant only for finite
            wall thickness.
        """

        self._x0, self._width = float(pos), float(width)
        self._V0, self._Vwall, self._Vext = float(V0), float(Vwall), float(Vext)
        self._d = float(d)

    def V(self, r, t):
        """Well potential at r, t.
        """

        dist = np.abs(np.asarray(r) - self._x0)
        outside = self._Vext if self._d > 0.0 else self._Vwall
        wallEdge = self._width/2.0 + (self._d if self._d > 0.0 else np.inf)
        return np.where(dist <= self._width/2.0, self._V0,
                        np.where(dist <= wallEdge, self._Vwall, outside)) + 0.0j


class SquareWell1d(Potential1d):
    """A "square well": a parabola inside a region of given width, continued
    by constant potentials to each side.
    """

    isTimeDependent = False

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
            determined by the width. It is chosen such that that V is continuous.
        well : boolean
            If True, the outside potential is higher than the crest potential,
            thus, a well is formed. If False, a bump potential is created.

        Note
        ----
        If you want an oscillator frequency that gives you a specific V_outside
        for given width and V0, use this formula:
        omega = sqrt(2(V_1 - V_0)/(sign*(d/2)^2)).
        """

        self._x0, self._omega, self._width = float(x0), float(omega), float(width)
        self._V0, self._parabSign = float(V0), +1.0 if well else -1.0
        self._V1 = self._parabSign*0.5*self._omega**2*(self._width/2.0)**2 + self._V0

    def V(self, x, t):
        dx = np.asarray(x) - self._x0
        return np.where(np.abs(dx) <= self._width/2.0,
                        0.5*self._parabSign*self._omega**2*dx**2 + self._V0, self._V1) + 0.0j


class MicrocavityRadialPotential(Potential1d):
    """The radial potential as used in microcavity optics of the circle.
    """

    isTimeDependent = False

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

        self._R, self._n, self._k = float(R), float(n), float(k)
        self._mSquare = float(int(m))**2

    def V(self, x, t):
        x = np.asarray(x, dtype=float)
        return np.where(x < self._R, self._k**2*(1.0 - self._n**2), 0.0) + self._mSquare/x**2 + 0.0j


MicrocyvityRadialPotential = MicrocavityRadialPotential  # old (misspelled) name


class EckartPotential1d(Potential1d):
    """The Eckart potential barrier.
    """

    isTimeDependent = False

    def __init__(self, x0, V0, alpha):
        r"""Create an Eckart potential $V = V_0 \sech(\alpha (x-x_0)^2)$.

        Parameters
        ----------
        x0 : float
            Center position of the bump.
        V0 : float
            Height of the bump.
        alpha : float
            Shape parameter.

        Notes
        -----
        Reference: Eckart, Phys. Rev. 35, 1303 (1930).
        """

        self._x0, self._V0, self._alpha = float(x0), float(V0), float(alpha)

    def V(self, x, t):
        return self._V0/np.cosh(self._alpha*(np.asarray(x) - self._x0)**2) + 0.0j


class GaussBarrier1d(Potential1d):
    r"""A Gaussian bump in $1d$.
    """

    isTimeDependent = False

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

    def V(self, x, t):
        return self._V0*np.exp(-(np.asarray(x) - self._x0)**2/(2*self._sigmaSquare)) + 0.0j


class Ramp1d(Potential1d):
    """A ramp potential.
    """

    isTimeDependent = False

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

    def V(self, x, t):
        return self._V0/2*(1 + np.tanh(self._sign*(np.asarray(x) - self._x0)/(2*self._alpha))) + 0.0j


class MovingRamp1d(Potential1d):
    """A moving ramp potential.
    """

    isTimeDependent = True

    def __init__(self, x0, V0, alpha, uphill, motionType, frequency, motionAmpl, phase):
        r"""Create a moving ramp potential $V = V_0/2 (1 +
        \tanh((x-x_0(t))/(2\alpha)))$.

        Use this to construct a Fermi accelerator.

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
            "sin" for $sin(\omega t)$ time dependence (only option so far).
        frequency : float
            Frequency of oscillation.
        motionAmpl : float
            Amplitude of ramp position oscillation.
        phase : float
            Phase of oscillation.
        """

        if(motionType != "sin"):
            raise NotImplementedError("Only motionType='sin' is implemented.")

        self._x0, self._V0, self._alpha = float(x0), float(V0), float(alpha)
        self._motionType = motionType
        self._omega, self._A, self._phi0 = float(frequency), float(motionAmpl), float(phase)
        self._sign = +1.0 if uphill else -1.0

    def V(self, x, t):
        x0 = self._x0 + self._A*np.sin(self._omega*t + self._phi0)
        return self._V0/2*(1 + np.tanh(self._sign*(np.asarray(x) - x0)/(2*self._alpha))) + 0.0j


class HarmOscPot(Potential1d):
    r"""Harmonic oscillator potential $V = \omega_0^2 x^2/2$.
    """

    isTimeDependent = False

    def __init__(self, omega0, x0=0.0):
        r"""Create harmonic oscillator potential $V = \omega_0^2 (x-x_0)^2/2$
        in 1d.

        Parameters
        ----------
        omega0 : float
            Harmonic oscillator frequency.
        x0 : float, optional
            Center position.
        """

        self._omega0, self._x0 = float(omega0), float(x0)

    def V(self, x, t):
        return 0.5*self._omega0**2*(np.asarray(x) - self._x0)**2 + 0.0j


class InterpolatingPotential1d(Potential1d):
    """Interpolating polynomial potential defined by a set of points.
    """

    isTimeDependent = False

    def __init__(self, points):
        r"""Create a polynomial potential that interpolates the given points.

        Parameters
        ----------
        points : list
            List of tuples (x_i, y_i).
        """

        x, y = zip(*points)
        self._interpolPoly = BarycentricInterpolator(x, y)

    def V(self, x, t):
        return self._interpolPoly(x) + 0.0j


def gaussian_random_field(shape, spacing, corrFunc, rng=None):
    r"""Draw a real Gaussian random field with a given autocorrelation
    function on a periodic grid.

    By the Wiener-Khinchin theorem, the power spectrum of the field is the
    Fourier transform of its autocorrelation function. White noise is filtered
    with the square root of that spectrum, which leaves the desired
    correlations.

    Parameters
    ----------
    shape : tuple of int
        Number of grid points per dimension.
    spacing : tuple of float
        Grid spacing per dimension.
    corrFunc : callable
        Correlation function ``C(*lags)``, called with one array of lags per
        dimension.
    rng : numpy.random.Generator or int, optional
        Random number generator or seed.

    Returns
    -------
    field : array
        Field with zero mean and unit standard deviation.
    """

    rng = np.random.default_rng(rng)
    shape = tuple(int(n) for n in np.atleast_1d(shape))
    spacing = tuple(float(d) for d in np.atleast_1d(spacing))
    # periodic lags 0, d, 2d, ..., -2d, -d in FFT ordering:
    lags = np.meshgrid(*[np.fft.fftfreq(n, 1.0/n)*d for n, d in zip(shape, spacing)], indexing="ij")
    spectrum = np.clip(np.real(np.fft.fftn(corrFunc(*lags))), 0.0, None)
    noise = rng.standard_normal(shape)
    field = np.real(np.fft.ifftn(np.sqrt(spectrum)*np.fft.fftn(noise)))
    field -= field.mean()
    return field/field.std()


def _normalized(field, normalize):
    if(normalize == "max"):
        return field/np.max(np.abs(field))
    elif(normalize == "std"):
        return field/np.std(field)
    raise ValueError("normalize must be 'max' or 'std'")


class RandomPotential1d(Potential1d):
    r"""Create a random potential from a given autocorrelation function.
    """

    isTimeDependent = False

    def __init__(self, corrFunc, x, V0, rng=None, normalize="max"):
        """Create random potential.

        Parameters
        ----------
        corrFunc : callable
            Correlation function C(dx); must accept arrays.
        x : array
            The (equidistant) grid values.
        V0 : float
            Amplitude of the potential, see ``normalize``.
        rng : numpy.random.Generator or int, optional
            Random number generator or seed for reproducible potentials.
        normalize : {"max", "std"}, optional
            Scale the potential such that its maximum absolute value
            ("max") or its standard deviation ("std") equals V0.
        """

        self._x = np.asarray(x, dtype=float)
        field = gaussian_random_field(len(self._x), self._x[1] - self._x[0], corrFunc, rng)
        self._values = V0*_normalized(field, normalize)

    def V(self, x, t):
        return np.interp(x, self._x, self._values) + 0.0j


# absorbing potentials:
class ExpAbsorber1d(Potential1d):
    """Absorbing complex potential.

    A complex potential with negative real part absorbs better (the negative
    real part increases kinetic energy and decreases the de Broglie
    wavelength, which makes absorption easier). Use a negative imaginary part
    for absorption.

    Compare e.g. Hussain and Roberts, Phys. Rev. A 63, 012703 (2000).
    """

    isTimeDependent = False

    def __init__(self, Areal, Aimag, lreal, limag, x0, x1, absorbLeft=True, absorbRight=True):
        """Create absorbing complex potential in 1d.

        Parameters
        ----------
        Areal, Aimag : float
            Amplitudes of real and imaginary part.
        lreal : float
            Accelerating length of potential.
        limag  : float
            Absorbing length of potential.
        x0, x1 : float
            Outer points of spatial grid.
        absorbLeft, absorbRight : bool, optional
            Absorb on the left respectively right boundary of the grid.
        """

        self._lreal, self._limag = float(lreal), float(limag)
        self._x0, self._x1 = float(x0), float(x1)
        self._Areal, self._Aimag = float(Areal), float(Aimag)
        self._alpha_r = np.log(999000)/self._lreal
        self._Aleft = 1.0 if absorbLeft else 0.0
        self._Aright = 1.0 if absorbRight else 0.0

    def V(self, x, t):
        x = np.asarray(x, dtype=float)
        real = self._Areal*(self._Aleft/(1.0 + np.exp(self._alpha_r*(x - (self._x0 + self._lreal))))
                            + self._Aright/(1.0 + np.exp(-self._alpha_r*(x - (self._x1 - self._lreal)))))
        imag = self._Aimag*(self._Aleft*np.exp(-np.abs(x - self._x0)/self._limag)
                            + self._Aright*np.exp(-np.abs(x - self._x1)/self._limag))
        return real + 1j*imag


#-----------------------------------------------------------------------
# 2d potentials: evaluated as V(r, t) with r = (X, Y)
class Potential2d(Potential):
    """Potentials in 2d. ``V(r, t)`` takes ``r = (X, Y)``.
    """


class HarmOscPot2d(Potential2d):
    r"""Anisotropic harmonic oscillator $V = (\omega_x^2 (x-x_0)^2 + \omega_y^2 (y-y_0)^2)/2$.
    """

    isTimeDependent = False

    def __init__(self, omegax, omegay=None, x0=0.0, y0=0.0):
        self._wx = float(omegax)
        self._wy = float(omegax if omegay is None else omegay)
        self._x0, self._y0 = float(x0), float(y0)

    def V(self, r, t):
        x, y = r
        return 0.5*(self._wx**2*(x - self._x0)**2 + self._wy**2*(y - self._y0)**2) + 0.0j


class GaussBump2d(Potential2d):
    r"""Gaussian bump $V = V_0 \exp(-((x-x_0)^2 + (y-y_0)^2)/2\sigma^2)$.
    """

    isTimeDependent = False

    def __init__(self, x0, y0, V0, sigma):
        self._x0, self._y0, self._V0, self._sigma = float(x0), float(y0), float(V0), float(sigma)

    def V(self, r, t):
        x, y = r
        return self._V0*np.exp(-((x - self._x0)**2 + (y - self._y0)**2)/(2*self._sigma**2)) + 0.0j


class StadiumPotential2d(Potential2d):
    r"""Bunimovich stadium: a rectangle of length $2a$ and height $2R$ capped
    by two half circles of radius $R$, centered at $(x_0, y_0)$.

    The potential is zero inside and $V_0$ outside, with walls smoothed over a
    width $w$: $V = V_0/2\,(1 + \tanh(d/w))$, where $d$ is the signed distance
    to the boundary (negative inside). With $a = 0$ the stadium is a circle,
    whose classical dynamics is integrable, while every stadium with $a > 0$
    is chaotic.
    """

    isTimeDependent = False

    def __init__(self, a, R, V0, w=0.1, x0=0.0, y0=0.0):
        """Create a stadium potential.

        Parameters
        ----------
        a : float
            Half length of the straight segments.
        R : float
            Radius of the half circles.
        V0 : float
            Potential outside.
        w : float, optional
            Width of the smoothed walls.
        x0, y0 : float, optional
            Center.
        """

        self._a, self._R, self._V0, self._w = float(a), float(R), float(V0), float(w)
        self._x0, self._y0 = float(x0), float(y0)

    def signedDistance(self, r):
        """Signed distance to the stadium boundary (negative inside)."""
        x, y = r
        x, y = np.abs(np.asarray(x) - self._x0), np.asarray(y) - self._y0
        dx = np.maximum(x - self._a, 0.0)
        return np.sqrt(dx**2 + y**2) - self._R

    def V(self, r, t):
        return 0.5*self._V0*(1 + np.tanh(self.signedDistance(r)/self._w)) + 0.0j


class RandomPotential2d(Potential2d):
    r"""Gaussian random potential with a given autocorrelation function.

    The potential is drawn on a periodic grid and interpolated (with periodic
    continuation) when evaluated elsewhere. Weak, smooth random potentials
    cause *branched flow*: a wave entering such a medium splits into
    narrow branches, cmp. Topinka et al., Nature 410, 183 (2001) and
    Heller, Kaplan, Dahlen, Phys. Today 74(4), 44 (2021).
    """

    isTimeDependent = False

    def __init__(self, corrFunc, x, y, V0, rng=None, normalize="std"):
        """Create a random potential.

        Parameters
        ----------
        corrFunc : callable
            Correlation function C(dx, dy); must accept arrays.
        x, y : array
            Equidistant grid values along x and y.
        V0 : float
            Amplitude of the potential, see ``normalize``.
        rng : numpy.random.Generator or int, optional
            Random number generator or seed.
        normalize : {"std", "max"}, optional
            Scale the potential such that its standard deviation ("std") or
            maximum absolute value ("max") equals V0.
        """

        from scipy.interpolate import RegularGridInterpolator

        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        dx, dy = x[1] - x[0], y[1] - y[0]
        field = gaussian_random_field((len(x), len(y)), (dx, dy), corrFunc, rng)
        self.values = V0*_normalized(field, normalize)
        self._x0, self._y0 = x[0], y[0]
        self._Lx, self._Ly = len(x)*dx, len(y)*dy
        # periodic continuation: append the first row and column at the end
        ext = np.pad(self.values, ((0, 1), (0, 1)), mode="wrap")
        self._interp = RegularGridInterpolator((np.append(x, x[0] + self._Lx), np.append(y, y[0] + self._Ly)), ext)

    def V(self, r, t):
        x, y = r
        x = self._x0 + np.mod(np.asarray(x) - self._x0, self._Lx)
        y = self._y0 + np.mod(np.asarray(y) - self._y0, self._Ly)
        return self._interp(np.stack([x, y], axis=-1)) + 0.0j


class BoundaryAbsorber2d(Potential2d):
    r"""Absorbing layer along the edges of a rectangular grid.

    Inside a layer of width $l$ along each edge, the potential is
    $-i\eta\,((l - d)/l)^2$ with $d$ the distance to the edge. The quadratic
    onset keeps reflections small.
    """

    isTimeDependent = False

    def __init__(self, size, width, strength):
        """Create an absorbing boundary layer.

        Parameters
        ----------
        size : tuple
            Grid boundaries ((x_min, x_max), (y_min, y_max)).
        width : float
            Width l of the absorbing layer.
        strength : float
            Maximum absorption rate eta (positive).
        """

        (self._x0, self._x1), (self._y0, self._y1) = size
        self._l, self._eta = float(width), float(strength)

    def V(self, r, t):
        x, y = r
        d = np.minimum.reduce([np.asarray(x) - self._x0, self._x1 - np.asarray(x),
                               np.asarray(y) - self._y0, self._y1 - np.asarray(y)])
        ramp = np.clip((self._l - d)/self._l, 0.0, None)
        return -1j*self._eta*ramp**2
