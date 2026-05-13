"""
This module provides the functions for the one-loop finite
temperature corrections to a potential in QFT. The two basic
functions are:

    Jb(x) = int[0->inf] dy +y^2 log( 1 - exp(-sqrt(x^2 + y^2)) )

    Jf(x) = int[0->inf] dy -y^2 log( 1 + exp(-sqrt(x^2 + y^2)) )

Call them by:

    Jb(x, approx='high', deriv=0, n = 8)

Here, approx can either be 'exact', 'spline', 'high', or 'low'.
``Exact`` calculates the integral numerically, while ``high``
and ``low`` calculate the high and low x expansions of J to order n.
Specify the derivative with the ``deriv`` parameter.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import os
from typing import Callable

import numpy
from scipy import integrate, interpolate
from scipy import special
from scipy.special import factorial as fac

pi = numpy.pi
euler_gamma = 0.577215661901532
log, exp, sqrt = numpy.log, numpy.exp, numpy.sqrt
array = numpy.array

spline_data_path = os.path.dirname(__file__) + "/tab_data"


# The following are the exact integrals:

def _Jf_exact(x : complex | float) -> complex:
    """Evaluate the fermionic thermal integral for a single scalar argument.

    Parameters
    ----------
    x : complex | float
        Dimensionless mass-to-temperature ratio. Complex values are handled via
        analytic continuation as described in the original CosmoTransitions
        implementation.

    Returns
    -------
    complex
        Exact value of the integral ``J_f(x)``.
    """
    f = lambda y: -y*y*log(1+exp(-sqrt(y*y+x*x)))
    if(x.imag == 0):
        x = abs(x)
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: -y*y*log(2*abs(numpy.cos(sqrt(abs(x*x)-y*y)/2)))
        return (
            integrate.quad(f1,0,abs(x))[0] +
            integrate.quad(f,abs(x),numpy.inf)[0]
        )


def _Jf_exact2(theta : float) -> float:
    """Evaluate the fermionic thermal integral as a function of ``theta = x^2``.

    Parameters
    ----------
    theta : float
        Squared mass-to-temperature ratio. Negative values are supported through
        analytic continuation.

    Returns
    -------
    float
        Exact value of the integral ``J_f(x)`` with ``x^2 = theta``.
    """
    # Note that this is a function of theta so that you can get negative values
    f = lambda y: -y*y*log(1+exp(-sqrt(y*y+theta))).real
    if theta >= 0:
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: -y*y*log(2*abs(numpy.cos(sqrt(-theta-y*y)/2)))
        return (
            integrate.quad(f, abs(theta)**.5, numpy.inf)[0] +
            integrate.quad(f1, 0, abs(theta)**.5)[0]
        )


def _Jb_exact(x : complex | float) -> complex:
    """Evaluate the bosonic thermal integral for a single scalar argument.

    Parameters
    ----------
    x : complex | float
        Dimensionless mass-to-temperature ratio. Complex values are handled via
        analytic continuation.

    Returns
    -------
    complex
        Exact value of the integral ``J_b(x)``.
    """
    f = lambda y: y*y*log(1-exp(-sqrt(y*y+x*x)))
    if(x.imag == 0):
        x = abs(x)
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: y*y*log(2*abs(numpy.sin(sqrt(abs(x*x)-y*y)/2)))
        return (
            integrate.quad(f1,0,abs(x))[0] +
            integrate.quad(f,abs(x),numpy.inf)[0]
        )


def _Jb_exact2(theta : float) -> float:
    """Evaluate the bosonic thermal integral as a function of ``theta = x^2``.

    Parameters
    ----------
    theta : float
        Squared mass-to-temperature ratio. Negative values are supported through
        analytic continuation.

    Returns
    -------
    float
        Exact value of the integral ``J_b(x)`` with ``x^2 = theta``.
    """
    # Note that this is a function of theta so that you can get negative values
    f = lambda y: y*y*log(1-exp(-sqrt(y*y+theta))).real
    if theta >= 0:
        return integrate.quad(f, 0, numpy.inf)[0]
    else:
        f1 = lambda y: y*y*log(2*abs(numpy.sin(sqrt(-theta-y*y)/2)))
        return (
            integrate.quad(f, abs(theta)**.5, numpy.inf)[0] +
            integrate.quad(f1, 0, abs(theta)**.5)[0]
        )


def _dJf_exact(x : complex | float) -> complex:
    """Derivative of the fermionic thermal integral with respect to ``x``."""
    f = lambda y: y*y*(exp(sqrt(y*y+x*x))+1)**-1*x/sqrt(y*y+x*x)
    return integrate.quad(f, 0, numpy.inf)[0]


def _dJb_exact(x : complex | float) -> complex:
    """Derivative of the bosonic thermal integral with respect to ``x``."""
    f = lambda y: y*y*(exp(sqrt(y*y+x*x))-1)**-1*x/sqrt(y*y+x*x)
    return integrate.quad(f, 0, numpy.inf)[0]


def arrayFunc(f : Callable[[float], float], x : float | numpy.ndarray, typ=float) -> numpy.ndarray | float:
    """Vectorise a scalar integrand helper so it accepts scalar or 1D arrays.

    Parameters
    ----------
    f : Callable[[float], float]
        Function that accepts and returns scalar values.
    x : float | array-like
        Input argument. Arrays are handled element-wise, scalars are forwarded
        directly.
    typ : type, optional
        Data type of the output array. Defaults to ``float``.

    Returns
    -------
    numpy.ndarray | float
        Array of evaluated values or the scalar return of ``f(x)``.
    """
    i = 0
    try:
        n = len(x)
    except:
        return f(x)  # x isn't an array
    s = numpy.empty(n, typ)
    while(i < n):
        try:
            s[i] = f(x[i])
        except:
            s[i] = numpy.NaN
        i += 1
    return s


def Jf_exact(x : complex | float) -> complex | float:
    """Jf calculated directly from the integral."""
    return arrayFunc(_Jf_exact, x, complex)


def Jf_exact2(theta : float) -> float:
    """Jf calculated directly from the integral; input is theta = x^2."""
    return arrayFunc(_Jf_exact2, theta)


def Jb_exact(x : complex | float) -> complex | float:
    """Jb calculated directly from the integral."""
    return arrayFunc(_Jb_exact, x)


def Jb_exact2(theta : float) -> float:
    """Jb calculated directly from the integral; input is theta = x^2."""
    return arrayFunc(_Jb_exact2, theta)


def dJf_exact(x : complex | float) -> complex | float:
    """dJf/dx calculated directly from the integral."""
    return arrayFunc(_dJf_exact, x)


def dJb_exact(x : complex | float) -> complex | float:
    """dJb/dx calculated directly from the integral."""
    return arrayFunc(_dJb_exact, x)


# Spline fitting, Jf
_xfmin = -6.82200203  # -11.2403168
_xfmax = 1.35e3
_Jf_dat_path = spline_data_path+"/finiteT_f.dat.txt"
if os.path.exists(_Jf_dat_path):
    _xf, _yf = numpy.loadtxt(_Jf_dat_path).T
else:
    # x = |xmin|*sinh(y), where y in linear
    # (so that we're not overpopulating the uniteresting region)
    _xf = numpy.linspace(numpy.arcsinh(-1.3*20),
                         numpy.arcsinh(-20*_xfmax/_xfmin), 1000)
    _xf = abs(_xfmin)*numpy.sinh(_xf)/20
    _yf = Jf_exact2(_xf)
    numpy.savetxt(_Jf_dat_path, numpy.array([_xf, _yf]).T)


_tckf = interpolate.splrep(_xf, _yf)
def Jf_spline(X,n=0):
    """Jf interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)
    x = X.ravel()
    y = interpolate.splev(x,_tckf, der=n).ravel()
    y[x < _xfmin] = interpolate.splev(_xfmin,_tckf, der=n)
    y[x > _xfmax] = 0
    return y.reshape(X.shape)


# Spline fitting, Jb
_xbmin = -2999.9999999999995 # This is new, see BSMPTv3 appendix
# We're setting the lower acceptable bound as the point where it's a minimum
# This guarantees that it's a monatonically increasing function, and the first
# deriv is continuous.
_xbmax = 1.41e3
_Jb_dat_path = spline_data_path+"/finiteT_b.dat.txt"
if os.path.exists(_Jb_dat_path):
    _xb, _yb = numpy.loadtxt(_Jb_dat_path).T
else:
    # x = |xmin|*sinh(y), where y in linear
    # (so that we're not overpopulating the uniteresting region)
    _xb = numpy.linspace(numpy.arcsinh(-1.3*20),
                         numpy.arcsinh(-20*_xbmax/_xbmin), 1000)
    _xb = abs(_xbmin)*numpy.sinh(_xb)/20
    _yb = Jb_exact2(_xb)
    numpy.savetxt(_Jb_dat_path, numpy.array([_xb, _yb]).T)


_tckb = interpolate.splrep(_xb, _yb)
def Jb_spline(X : float | numpy.ndarray, n : int = 0) -> float | numpy.ndarray:
    """Jb interpolated from a saved spline. Input is (m/T)^2."""
    X = numpy.array(X, copy=False)
    x = X.ravel()
    y = interpolate.splev(x,_tckb, der=n).ravel()
    y[x < _xbmin] = interpolate.splev(_xbmin,_tckb, der=n)
    y[x > _xbmax] = 0
    return y.reshape(X.shape)


# Now for the low x expansion (require that n <= 50)
a,b,c,d = -pi**4/45, pi*pi/12, -pi/6, -1/32.
logab = 1.5 - 2*euler_gamma + 2*log(4*pi)
l = numpy.arange(50)+1
g = (-2*pi**3.5 * (-1)**l*(1+special.zetac(2*l+1)) *
     special.gamma(l+.5)/(fac(l+2)*(2*pi)**(2*l+4)))
lowCoef_b = (a,b,c,d,logab,l,g)
del (a,b,c,d,logab,l,g)  # clean up name space

a,b,d = -7*pi**4/360, pi*pi/24, 1/32.
logaf = 1.5 - 2*euler_gamma + 2*log(pi)
l = numpy.arange(50)+1
g = (.25*pi**3.5 * (-1)**l*(1+special.zetac(2*l+1)) *
     special.gamma(l+.5)*(1-.5**(2*l+1))/(fac(l+2)*pi**(2*l+4)))
lowCoef_f = (a,b,d,logaf,l,g)
del (a,b,d,logaf,l,g)  # clean up name space


def Jb_low(x : complex | float, n : int = 20) -> complex | float:
    """Jb calculated using the low-x (high-T) expansion."""
    (a,b,c,d,logab,l,g) = lowCoef_b
    y = a + x*x*(b + x*(c + d*x*(numpy.nan_to_num(log(x*x)) - logab)))
    i = 1
    while i <= n:
        y += g[i-1]*x**(2*i+4)
        i += 1
    return y


def Jf_low(x : complex | float, n : int = 20) -> complex | float:
    """Jf calculated using the low-x (high-T) expansion."""
    (a,b,d,logaf,l,g) = lowCoef_f
    y = a + x*x*(b + d*x*x*(numpy.nan_to_num(log(x*x)) - logaf))
    i = 1
    while i <= n:
        y += g[i-1]*x**(2*i+4)
        i += 1
    return y


# The next few functions are all for the high-T approximation

def x2K2(k : int, x : float | numpy.ndarray) -> float | numpy.ndarray:
    """Return ``-x^2 K_2(k x) / k^2`` with analytic behaviour enforced at the origin,
    where ``K_2`` is the modified Bessel function of the second kind.

    Parameters
    ----------
    k : int
        Positive integer
    x : float | numpy.ndarray
        Dimensionless argument. Arrays are handled element-wise.

    Returns
    -------
    float | numpy.ndarray
        Evaluated expression with the ``x -> 0`` limit patched in.
    """
    y = -x*x*special.kn(2, k*x)/(k*k)
    if(isinstance(x, numpy.ndarray)):
        y[x == 0] = numpy.ones(len(y[x == 0]))*-2.0/k**4
    elif(x == 0):
        return -2.0/k**4
    return y


def dx2K2(k : int, x : float | numpy.ndarray) -> float | numpy.ndarray:
    """Return the first derivative of :func:`x2K2` with respect to ``x``."""
    y = abs(x)
    return numpy.nan_to_num(x*y*special.kn(1,k*y)/k)


def d2x2K2(k : int, x : float | numpy.ndarray) -> float | numpy.ndarray:
    """Return the second derivative of :func:`x2K2` with respect to ``x``."""
    x = abs(x)
    y = numpy.nan_to_num(x*(special.kn(1,k*x)/k - x*special.kn(0,k*x)))
    if(isinstance(x, numpy.ndarray)):
        y[x == 0] = numpy.ones(len(y[x == 0]))*1.0/k**2
    elif(x == 0):
        return 1.0/k**2
    return y


def d3x2K2(k : int, x : float | numpy.ndarray) -> float | numpy.ndarray:
    """Return the third derivative of :func:`x2K2` with respect to ``x``."""
    y = abs(x)
    return numpy.nan_to_num(x*(y*k*special.kn(1,k*y) - 3*special.kn(0,k*y)))


def Jb_high(x : float | numpy.ndarray, deriv=0, n=8) -> float | numpy.ndarray:
    """Jb calculated using the high-x (low-T) expansion."""
    K = (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]
    y, k = 0.0, 1
    while k <= n:
        y += K(k,x)
        k += 1
    return y


def Jf_high(x : float | numpy.ndarray, deriv=0, n=8) -> float | numpy.ndarray:
    """Jf calculated using the high-x (low-T) expansion."""
    K = (x2K2, dx2K2, d2x2K2, d3x2K2)[deriv]
    y, k, i = 0.0, 1, 1
    while k <= n:
        y += i*K(k,x)
        i *= -1
        k += 1
    return y


# And here are the final functions:
# Note that if approx = 'spline', the function called is
# J(theta) (x^2 -> theta so you can get negative mass squared)
def Jb(x : float | numpy.ndarray, approx : str ='high', deriv : int =0, n : int =8) -> float | numpy.ndarray:
    """
    A shorthand for calling one of the Jb functions above.

    Parameters
    ----------
    approx : str, optional
        One of 'exact', 'high', 'low', or 'spline'.
    deriv : int, optional
        The order of the derivative (0 for no derivative).
        Must be <= (1, 3, 0, 3) for approx = (exact, high, low, spline).
    n : int, optional
        Number of terms to use in the low and high-T approximations.
    """
    if(approx == 'exact'):
        if(deriv == 0):
            return Jb_exact(x)
        elif(deriv == 1):
            return dJb_exact(x)
        else:
            raise ValueError("For approx=='exact', deriv must be 0 or 1.")
    elif(approx == 'spline'):
        return Jb_spline(x, deriv)
    elif(approx == 'low'):
        if(n > 100):
            raise ValueError("Must have n <= 100")
        if(deriv == 0):
            return Jb_low(x,n)
        else:
            raise ValueError("For approx=='low', deriv must be 0.")
    elif(approx == 'high'):
        if(deriv > 3):
            raise ValueError("For approx=='high', deriv must be 3 or less.")
        else:
            return Jb_high(x, deriv, n)
    raise ValueError("Unexpected value for 'approx'.")


def Jf(x : float | numpy.ndarray, approx : str ='high', deriv : int =0, n : int =8) -> float | numpy.ndarray:
    """
    A shorthand for calling one of the Jf functions above.

    Parameters
    ----------
    approx : str, optional
        One of 'exact', 'high', 'low', or 'spline'.
    deriv : int, optional
        The order of the derivative (0 for no derivative).
        Must be <= (1, 3, 0, 3) for approx = (exact, high, low, spline).
    n : int, optional
        Number of terms to use in the low and high-T approximations.
    """
    if(approx == 'exact'):
        if(deriv == 0):
            return Jf_exact(x)
        elif(deriv == 1):
            return dJf_exact(x)
        else:
            raise ValueError("For approx=='exact', deriv must be 0 or 1.")
    elif(approx == 'spline'):
        return Jf_spline(x, deriv)
    elif(approx == 'low'):
        if(n > 100):
            raise ValueError("Must have n <= 100")
        if(deriv == 0):
            return Jf_low(x,n)
        else:
            raise ValueError("For approx=='low', deriv must be 0.")
    elif(approx == 'high'):
        if(deriv > 3):
            raise ValueError("For approx=='high', deriv must be 3 or less.")
        else:
            return Jf_high(x, deriv, n)
    raise ValueError("Unexpected value for 'approx'.")
