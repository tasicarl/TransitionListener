"""Module containing routines to calculate thermodynamic quantities
like the numberdensity, energy density, pressure and entropy density.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import numpy as np

from scipy import interpolate
from scipy import integrate
from scipy import special
from scipy.signal import savgol_filter

from .particles import BaseParticle

import os
myPath = os.path.abspath(os.path.dirname(__file__)) + "/"

# Interpolate numberdensities
def calcNumberdensityIntegrals():
    """Calculate geff for numberdensity"""
    def integf(u,x): return u * np.sqrt(u**2 - x**2) / (np.exp(u) + 1)
    def integb(u,x): return u * np.sqrt(u**2 - x**2) / (np.exp(u) - 1)

    n = 1000
    x_range = np.logspace(-7, np.log10(7e2), n)
    Ib = np.zeros(n)
    If = np.zeros(n)
    for i, x in enumerate(x_range):
        Ib[i] = integrate.quad(integb, x + 1e-50, np.inf, args=(x,))[0]
        If[i] = integrate.quad(integf, x + 1e-50, np.inf, args=(x,))[0]

    Ib = Ib / (2*special.zeta(3))
    If = If / (2*special.zeta(3))
    data = np.array([x_range, Ib, If]).T

    np.savetxt(myPath + "tab_data/numberdensities.dat", data, header="# x, Ib, If")


def calcEnergydensityIntegrals():
    """Tabulate geff for the energydensity"""
    def integf(u, x): return u**2 * np.sqrt(u**2 - x**2) / (np.exp(u) + 1)
    def integb(u, x): return u**2 * np.sqrt(u**2 - x**2) / (np.exp(u) - 1)

    n = 1000
    x_range = np.logspace(-7, np.log10(7e2), n)
    Ib = np.zeros(n)
    If = np.zeros(n)
    for i, x in enumerate(x_range):
        Ib[i] = integrate.quad(integb, x + 1e-50, np.inf, args=(x,))[0]
        If[i] = integrate.quad(integf, x + 1e-50, np.inf, args=(x,))[0]

    Ib = Ib * 15/np.pi**4
    If = If * 15/np.pi**4
    data = np.array([x_range, Ib, If]).T

    np.savetxt(myPath + "tab_data/energydensities.dat", data, header="# x, Ienergy_b, Ienergy_f")


def calcPressureIntegrals():
    """Pre-compute spline approximations of the pressure integrals."""
    def integf(u,x): return np.sqrt(u**2 - x**2)**3 / (np.exp(u) + 1)
    def integb(u,x): return np.sqrt(u**2 - x**2)**3 / (np.exp(u) - 1)

    n = 1000
    x_range = np.logspace(-7, np.log10(7e2), n)
    Ib = np.zeros(n)
    If = np.zeros(n)
    for i,x in enumerate(x_range):
        Ib[i] = integrate.quad(integb, x + 1e-50, np.inf, args=(x,))[0]
        If[i] = integrate.quad(integf, x + 1e-50, np.inf, args=(x,))[0]

    Ib = Ib * 15/np.pi**4 
    If = If * 15/np.pi**4 
    data = np.array([x_range, Ib, If]).T
        
    np.savetxt(myPath + "tab_data/pressures.dat", data, header="# x, Ipressure_b, Ipressure_f")


try:
    data_ndensity = np.loadtxt(myPath + "tab_data/numberdensities.dat")
    data_edensity = np.loadtxt(myPath + "tab_data/energydensities.dat")
    data_pressure = np.loadtxt(myPath + "tab_data/pressures.dat")
except:
    calcNumberdensityIntegrals()
    calcEnergydensityIntegrals()
    calcPressureIntegrals()
    data_ndensity = np.loadtxt(myPath + "tab_data/numberdensities.dat")
    data_edensity = np.loadtxt(myPath + "tab_data/energydensities.dat")
    data_pressure = np.loadtxt(myPath + "tab_data/pressures.dat")

data_geffSM = np.genfromtxt(myPath + "tab_data/geff_SM.dat")
Ie_geffSM = interpolate.CubicSpline(
    np.log10(data_geffSM[:, 0]),
    data_geffSM[:, 1],
    bc_type="clamped"
)
# Add filter to smooth for derivative
Ie_geffSM_smooth = interpolate.CubicSpline(
    np.log10(data_geffSM[:, 0]),
    savgol_filter(data_geffSM[:, 1], 81, 3),
    bc_type="clamped"
)

Is_geffSM = interpolate.CubicSpline(
    np.log10(data_geffSM[:, 0]),
    data_geffSM[:, 3],
    bc_type="clamped"
)
Is_geffSM_smooth = interpolate.CubicSpline(
    np.log10(data_geffSM[:, 0]),
    savgol_filter(data_geffSM[:, 3], 81, 3),
    bc_type="clamped"
)


Inb = interpolate.interp1d(data_ndensity[:,0], data_ndensity[:,1], kind="cubic")
Inf = interpolate.interp1d(data_ndensity[:,0], data_ndensity[:,2], kind="cubic")
Ieb = interpolate.interp1d(data_edensity[:,0], data_edensity[:,1], kind="cubic", fill_value=(1, 0), bounds_error=False)
Ief = interpolate.interp1d(
    data_edensity[:, 0],
    data_edensity[:, 2],
    kind="cubic",
    fill_value=(0.875, 0),
    bounds_error=False,
)
Ipb = interpolate.interp1d(data_pressure[:,0], data_pressure[:,1], kind="cubic")
Ipf = interpolate.interp1d(data_pressure[:,0], data_pressure[:,2], kind="cubic")

# ==================================================
# Fit function for the SM effective DOFs
# K. Saikawa and S. Shirai (2018), arxiv:1803.01038
# ==================================================

def geff_rho_h(T: float):
    r"""High-temperature fit for the SM energy-density degrees of freedom.

    The fit approximates :math:`g_{\mathrm{eff},\rho}^{\mathrm{SM}}(T)` defined
    by

    .. math::
       \rho_{\mathrm{SM}}(T)
       = \frac{\pi^2}{30}\, g_{\mathrm{eff},\rho}^{\mathrm{SM}}(T)\, T^4.

    It uses the rational approximation of Saikawa and Shirai
    (arXiv:1803.01038) above the QCD crossover.
    """
    a = np.array([1.0, 1.117240, 3.12672e-01,
                  -4.68049e-02, -2.65004e-02, -1.19760e-03,
                  1.82812e-04,  1.36436e-04,  8.55051e-05,
                  1.22840e-05,  3.82259e-07,  -6.87035e-09])
    b = np.array([1.43382e-02,  1.37559e-02,  2.92108e-03,
                  -5.38533e-04, -1.62496e-04, -2.87906e-05,
                  -3.84278e-06, 2.78776e-06,  7.40342e-07,
                  1.17210e-07,  3.72499e-09,  -6.74107e-11])

    t = np.log(T)
    x = 1
    temp1 = 0
    temp2 = 0
    for i in range(12):
        temp1 += a[i] * x
        temp2 += b[i] * x
        x *= t
    return temp1/temp2

def ratio(T: float):
    r"""Return the fitted ratio :math:`g_{\mathrm{eff},\rho}/g_{\mathrm{eff},s}`.

    This auxiliary fit is combined with :func:`geff_rho_h` to reconstruct the
    entropy degrees of freedom according to

    .. math::
       g_{\mathrm{eff},s}(T)
       = \frac{g_{\mathrm{eff},\rho}(T)}
              {g_{\mathrm{eff},\rho}(T)/g_{\mathrm{eff},s}(T)}.
              
    This function implements the ratio in the denominator.
    """
    a = np.array([1.,  6.07869e-01,  -1.54485e-01,
                  -2.24034e-01, -2.82147e-02, 2.90620e-02,
                  6.86778e-03,  -1.05e-03, -1.69104e-04,
                  1.06301e-05,  1.69528e-06,  -9.33311e-08])

    b = np.array([7.07388E+01,  9.18011E+01,  3.31892E+01,
                  -1.39779E+00, -1.52558E+00, -1.97857e-02,
                  -1.60146e-01, 8.22615e-05,  2.02651e-02,
                  -1.82134e-05, 7.83943e-05,  7.13518e-05])

    t = np.log(T)
    x = 1
    temp1 = 0
    temp2 = 0
    for i in range(12):
        temp1 += a[i] * x
        temp2 += b[i] * x
        x *= t

    return temp1 / temp2 + 1


def geff_s_h(T: float):
    r"""High-temperature fit for the SM entropy degrees of freedom.

    The quantity is defined by

    .. math::
       s_{\mathrm{SM}}(T)
       = \frac{2\pi^2}{45}\, g_{\mathrm{eff},s}^{\mathrm{SM}}(T)\, T^3.
    """
    return 1 / ratio(T) * geff_rho_h(T)


def fr(x: float):
    """Low-temperature fermionic energy-density kernel used in the tabulated SM fit."""
    return np.exp(-1.04855 * x) * \
        (1 + 1.03757 * x + 0.508630 * x * x + 0.0893988 * x * x * x)

def br(x: float):
    """Low-temperature bosonic energy-density kernel used in the tabulated SM fit."""
    return np.exp(-1.03149 * x) * \
        (1 + 1.03317 * x + 0.398264 * x * x + 0.0648056 * x * x * x)

def fs(x: float):
    """Low-temperature fermionic entropy-density kernel used in the tabulated SM fit."""
    return np.exp(-1.04190 * x) * \
        (1. + 1.03400 * x + 0.456426 * x * x + 0.0595248 * x * x * x)


def bs(x: float):
    """Low-temperature bosonic entropy-density kernel used in the tabulated SM fit."""
    return np.exp(-1.03365 * x) * \
        (1. + 1.03397 * x + 0.342548 * x * x + 0.0506182 * x * x * x)

def Sfit(x: float):
    """Auxiliary suppression factor appearing in the low-temperature SM fits."""
    return 1. + 7. / 4. * np.exp(-1.0419 * x) * \
        (1. + 1.034 * x + 0.456426 * x * x + 0.0595249 * x * x * x)

def geff_rho_l(T, p1 = 1, p2 = 1):
    r"""Low-temperature fit for :math:`g_{\mathrm{eff},\rho}^{\mathrm{SM}}`.

    Below the QCD crossover the fit is written as a sum of hadronic and leptonic
    threshold functions and still satisfies

    .. math::
       \rho_{\mathrm{SM}}(T)
       = \frac{\pi^2}{30}\, g_{\mathrm{eff},\rho}^{\mathrm{SM}}(T)\, T^4.
    """
    me = 511e-6; mmu = 0.1056; mpi0 = 0.135; mpip = 0.140; m1 = 0.5
    m2 = 0.77; m3 = 1.2; m4 = 2.0
    return 2.030 + 1.353 * np.power(Sfit(me / T), 4. / 3.) + 3.495 * fr(me / T) + \
        3.446 * fr(mmu / T) + 1.05 * br(mpi0 / T) + 2.08 * br(mpip / T) + \
        4.165 * br(m1 / T) + 30.55 * br(m2 / T) + 89.4 * br(p2 * m3 / T) + \
        8209 * br(p2 * m4 / T)

def geff_s_l(T: float, p1=1, p2=1):
    r"""Low-temperature fit for :math:`g_{\mathrm{eff},s}^{\mathrm{SM}}`.

    The fit is defined through

    .. math::
       s_{\mathrm{SM}}(T)
       = \frac{2\pi^2}{45}\, g_{\mathrm{eff},s}^{\mathrm{SM}}(T)\, T^3.
    """
    me = 511e-6; mmu = 0.1056; mpi0 = 0.135; mpip = 0.140; m1 = 0.5
    m2 = 0.77; m3 = 1.2; m4 = 2.0
    return 2.008 + 1.923 * np.power(Sfit(me / T), 1.) + 3.442 * fs(me / T) + \
        3.468 * fs(mmu / T) + 1.034 * bs(mpi0 / T) + 2.068 * bs(mpip / T) + \
        4.160 * bs(m1 / T) + 30.55 * bs(m2 / T) + 90 * bs(p2 * m3 / T) + \
        6209 * br(p2 * m4 / T)


def _e_geffSM_fit(T: float, CF: float):
    """Dispatch between the low- and high-temperature SM energy-density fits."""
    if (T < 0.12):
        return geff_rho_l(T)
    return geff_rho_h(T)
    
e_geffSM_fit = np.vectorize(_e_geffSM_fit)

def _s_geffSM_fit(T: float, CF: float):
    """Dispatch between the low- and high-temperature SM entropy-density fits."""
    if (T < 0.12):
        return geff_s_l(T)
    return geff_s_h(T)
    
s_geffSM_fit = np.vectorize(_s_geffSM_fit)

def p_geffSM_fit(TSM: float | np.ndarray, CF: float) -> float | np.ndarray:
    """
    Effective pressure degress of freedom for the SM.

    Parameters
    ----------
    TSM : float or np.ndarray
        Temperature in internal energy units.
    CF : float
        Conversion factor to put everything in GeV.
    """
    TSM_arr = np.asarray(TSM, dtype=float)
    TSM_GeV = TSM_arr * CF
    s_val = np.asarray(s_geffSM_fit(TSM_GeV), dtype=float)
    e_val = np.asarray(e_geffSM_fit(TSM_GeV), dtype=float)
    p_base = 4 * s_val - 3 * e_val

    return float(p_base) if np.asarray(p_base).shape == () else p_base


def e_geffSM(
    TSM: float | np.ndarray,
    CF: float,
    mode: str = "smooth"
) -> float | np.ndarray:
    """
    Effective energy degrees of freedom of the SM
    at a given temperature.

    Parameters
    ----------
    TSM : float or np.ndarray
        Temperature in internal energy units.
    CF : float
        Conversion factor to put everything in GeV.
    mode : str, optional
        "smooth", "fit" or "data"
    """
    TSM_arr = np.asarray(TSM, dtype=float)
    T_GeV = TSM_arr * CF
    if mode == "smooth":
        result = np.asarray(Ie_geffSM_smooth(np.log10(T_GeV)), dtype=float)
    elif mode == "data":
        result = np.asarray(Ie_geffSM(np.log10(T_GeV)), dtype=float)
    elif mode == "fit":
        result = e_geffSM_fit(TSM, CF)

    return float(result) if result.shape == () else result


def s_geffSM(TSM: float | np.ndarray, CF: float, mode: str = "smooth"
             ) -> float | np.ndarray:
    """
    Effective entropy degrees of freedom of the SM
    at a given temperature.

    Parameters
    ----------
    TSM : float or np.ndarray
        Temperature in internal energy units.
    CF : float
        Conversion factor to put everything in GeV.
    mode : str, optional
        "fit", "smooth" or "data".
    """
    TSM_arr = np.asarray(TSM, dtype=float)
    T_GeV = TSM_arr * CF
    if mode == "smooth":
        result = np.asarray(Is_geffSM_smooth(np.log10(T_GeV)), dtype=float)
    elif mode == "data":
        result = np.asarray(Is_geffSM(np.log10(T_GeV)), dtype=float)
    elif mode == "fit":
        result = s_geffSM_fit(TSM, CF)

    return float(result) if result.shape == () else result


def p_geffSM(TSM: float | np.ndarray, CF: float, mode: str = "smooth"
             ) -> float | np.ndarray:
    """
    Effective pressure degress of freedom for the SM.

    Parameters
    ----------
    TSM : float or np.ndarray
        Temperature in internal energy units.
    CF : float
        Conversion factor to put everything in GeV.
    mode : str, optional
        Either "fit", "smooth" or "data".
    """
    TSM_arr = np.asarray(TSM, dtype=float)
    TSM_GeV = TSM_arr * CF
    if mode == "smooth":
        s_val = np.asarray(Is_geffSM_smooth(np.log10(TSM_GeV)), dtype=float)
        e_val = np.asarray(Ie_geffSM_smooth(np.log10(TSM_GeV)), dtype=float)
    elif mode == "data":
        s_val = np.asarray(Is_geffSM(np.log10(TSM_GeV)), dtype=float)
        e_val = np.asarray(Ie_geffSM(np.log10(TSM_GeV)), dtype=float)
    elif mode == "fit":
        s_val = s_geffSM_fit(TSM, CF)
        e_val = e_geffSM_fit(TSM, CF)
    p_base = 4 * s_val - 3 * e_val

    return float(p_base) if np.asarray(p_base).shape == () else p_base

def nDensity(m, T, g, ptype):
    """Numberdensity from tabulated values."""
    pref = special.zeta(3)/np.pi**2 * g * T**3
    if T == 0.0:
        return 0.0
    x = m/T
    res = 0.0
    if ptype == "b":
        if x < Inb.x[0]:
            res = pref
        elif x < Inb.x[-1]:
            res = pref * Inb(x)
    elif ptype == "f":
        if x < Inb.x[0]:
            res = (3/4) * pref
        elif x < Inb.x[-1]:
            res = pref * Inf(x)
    return res


def e_geff(particles: BaseParticle | list[BaseParticle], X: np.ndarray, T: float | np.ndarray):
    """
    Calculate the particles effective degrees of freedom.
    """
    T += 1e-100

    if not isinstance(particles, list):
        particles = [particles]

    geff = np.zeros_like(X)

    for p in particles:
        masses = p.evaluate_prefactor(X)
        if masses is None:
            raise AttributeError(
                "Thermodynamic routines require particles with thermal mass prefactors"
            )
        m = np.sqrt(np.asarray(masses))
        x = m / T
        res = np.zeros_like(X)
        if p.statistic == "fermion":
            res = p.dof * np.where(x < Ief.x[0], 7 / 8, res)
            res = p.dof * np.where((x >= Ief.x[0]) & (x < Ief.x[-1]), Ief(x), res)
        elif p.statistic == "boson":
            res = p.dof * np.where(x < Ieb.x[0], 1.0, res)
            res = p.dof * np.where((x >= Ieb.x[0]) & (x < Ieb.x[-1]), Ieb(x), res)
        geff += res
    return geff


def p_geff(particles: BaseParticle | list[BaseParticle], X: np.ndarray, T: float | np.ndarray):
    """
    Calculate the particles pressure effective degrees of freedom.
    """
    T += 1e-100

    if not isinstance(particles, list):
        particles = [particles]

    geff = np.zeros_like(X)

    for p in particles:
        masses = p.evaluate_prefactor(X)
        if masses is None:
            raise AttributeError(
                "Thermodynamic routines require particles with thermal mass prefactors"
            )
        m = np.sqrt(np.asarray(masses))
        x = m / T
        res = np.zeros_like(X)
        if p.statistic == "fermion":
            res = p.dof * np.where(x < Ipf.x[0], 7 / 8, res)
            res = p.dof * np.where((x >= Ipf.x[0]) & (x < Ipf.x[-1]), Ipf(x), res)
        elif p.statistic == "boson":
            res = p.dof * np.where(x < Ipb.x[0], 1.0, res)
            res = p.dof * np.where((x >= Ipb.x[0]) & (x < Ipb.x[-1]), Ipb(x), res)
        geff += res
    return geff


def e_geff(m, T, g, ptype):
    """Energy degrees of freedom, supports scalar or vector inputs for m and T."""
    m = np.asarray(m)  # Ensure m is a numpy array
    T = np.asarray(T)  # Ensure T is a numpy array
    T = T + 1e-100
    x = np.where(T != 0, m / T, np.inf)  # Avoid division by zero
    res = np.zeros_like(x)

    if ptype == "b":
        res = np.where(x < Ieb.x[0], 1.0, res)
        res = np.where((x >= Ieb.x[0]) & (x < Ieb.x[-1]), Ieb(x), res)
    elif ptype == "f":
        res = np.where(x < Ief.x[0], 7 / 8, res)
        res = np.where((x >= Ief.x[0]) & (x < Ief.x[-1]), Ief(x), res)
    return res * g


def p_geff(m, T, g, ptype):
    """Pressure degrees of freedom, supports scalar or vector inputs for m and T.

    At exactly ``T = 0`` a massless mode returns its full count, as ``e_geff`` does, since
    that is the limit of the coefficient; the pressure itself still vanishes, because every
    caller multiplies by ``T**4``. A massive mode returns zero there.
    """
    m = np.asarray(m)  # Ensure m is a numpy array
    T = np.asarray(T)  # Ensure T is a numpy array
    T = T + 1e-100
    x = np.where(T != 0, m / T, np.inf)  # Avoid division by zero
    res = np.zeros_like(x)

    if ptype == "b":
        # the pressure interpolators raise outside their range, so evaluate them on clipped
        # arguments and keep the massless (1) and Boltzmann-suppressed (0) limits by hand
        xc = np.clip(x, Ipb.x[0], Ipb.x[-1])
        res = np.where(x < Ipb.x[0], 1.0, res)
        res = np.where((x >= Ipb.x[0]) & (x < Ipb.x[-1]), Ipb(xc), res)
    elif ptype == "f":
        xc = np.clip(x, Ipf.x[0], Ipf.x[-1])
        res = np.where(x < Ipf.x[0], 7 / 8, res)
        res = np.where((x >= Ipf.x[0]) & (x < Ipf.x[-1]), Ipf(xc), res)
    return res * g


def s_geff(m, T, g, ptype):
    """Entropy degrees of freedom, supports scalar or vector inputs for m and T."""
    return (3*e_geff(m, T, g, ptype) + p_geff(m, T, g, ptype))/4


def s_geffDS(mSq_bosons, mSq_fermions, T):
    """Entropy degrees of freedom of the dark sector"""
    m2b, gb, c, physical = mSq_bosons
    m2f, gf = mSq_fermions

    geff = 0.0
    for i in range(len(m2b)):
        if physical[i]:
            if m2b[i] > 0:
                geff += s_geff(np.sqrt(m2b[i]), T, gb[i], 'b')
            else:
                # In general, tachyonic modes should not be included
                # in the effective degrees of freedom, but we include them
                # here, assuming that in the plasma they
                # are not tachyonic anymore. One example is the SM Higgs
                # field, which is tachyonic in the symmetric phase,
                # but still contributes to the effective degrees of freedom
                # with 4 degrees of freedom before the phase transition.
                # Truly tachyonic modes, which also
                # don't receive large enough plasma masses in order to have
                # m^2 > 0, should be excluded in the sum.
                # This should be checked in the model / be implemented through the
                # physical flag.
                geff += gb[i]
    for i in range(len(m2f)):
        geff += s_geff(np.sqrt(m2f[i]), T, gf[i], 'f')
    return geff

def e_geffDS(mSq_bosons, mSq_fermions, T):
    """Energy degrees of freedom of the dark sector, supports vectorized T."""
    m2b, gb, c, physical = mSq_bosons
    m2f, gf = mSq_fermions

    T = np.asarray(T)  # Ensure T is a numpy array
    geff = np.zeros_like(T, dtype=float)

    for i in range(len(m2b)):
        if physical[i]:
            if m2b[i] > 0:
                geff += e_geff(np.sqrt(m2b[i]), T, gb[i], 'b')
            else:
                # In general, tachyonic modes should not be included
                # in the effective degrees of freedom, but we include them
                # here, assuming that in the plasma they
                # are not tachyonic anymore. One example is the SM Higgs
                # field, which is tachyonic in the symmetric phase,
                # but still contributes to the effective degrees of freedom
                # with 4 degrees of freedom before the phase transition.
                # Truly tachyonic modes, which also
                # don't receive large enough plasma masses in order to have
                # m^2 > 0, should be excluded in the sum.
                # This should be checked in the model / be implemented through the
                # physical flag.
                geff += gb[i] # modes with negative mass are inlcuded 
    for i in range(len(m2f)):
        geff += e_geff(np.sqrt(m2f[i]), T, gf[i], 'f')
    return geff


def potential_fields_geff(mSq_bosons, mSq_fermions, T, kind: str = "e", mask=None):
    """Degrees of freedom of the fields of a potential, in the Landau gauge counting.

    Every mode counts, the Goldstone modes included; the ghosts are subtracted by the caller,
    as in ``generic_potential.radiationEnergyDensity``. Tachyonic modes count as massless, as
    they do in ``e_geffDS``, on the grounds that the plasma gives them a thermal mass.

    ``kind`` selects energy (``"e"``), pressure (``"p"``) or entropy (``"s"``) degrees of
    freedom; ``mask`` restricts the sum to a subset of the fields, as a pair of boolean arrays
    for bosons and fermions.
    """
    fn = {"e": e_geff, "p": p_geff, "s": s_geff}[kind]
    m2b, gb, _, _ = mSq_bosons
    m2f, gf = mSq_fermions
    mask_b = np.ones(np.shape(m2b), dtype=bool) if mask is None or mask[0] is None else np.asarray(mask[0], dtype=bool)
    mask_f = np.ones(np.shape(m2f), dtype=bool) if mask is None or mask[1] is None else np.asarray(mask[1], dtype=bool)
    total = 0.0
    for i in range(len(m2b)):
        if mask_b[i]:
            total = total + fn(np.sqrt(np.maximum(m2b[i], 0.0)), T, gb[i], "b")
    for i in range(len(m2f)):
        if mask_f[i]:
            total = total + fn(np.sqrt(np.maximum(m2f[i], 0.0)), T, gf[i], "f")
    return total


def sm_fields_in_potential_geff(pot, T, kind: str = "e"):
    """Tabulated degrees of freedom of the Standard Model fields that the potential contains.

    The tables ``e_geffSM``, ``p_geffSM`` and ``s_geffSM`` describe the whole Standard Model.
    A model that puts Standard Model fields into its potential (flagged ``is_SM``, e.g. W, Z,
    t and h in the 2HDM) counts those fields there, with their field- and temperature-dependent
    masses, so their tabulated contribution has to be removed once. It is evaluated at the
    masses the fields have in the zero-temperature vacuum of the model, where they are the
    Standard Model masses, and their ghosts are removed with them, so that the ghosts the
    potential subtracts are not counted twice.
    """
    spectrum = getattr(pot, "mass_spectrum", None)
    if spectrum is None:
        return 0.0
    is_boson_SM = np.asarray(spectrum.is_SM_bosons, dtype=bool)
    is_fermion_SM = np.asarray(spectrum.is_SM_fermions, dtype=bool)
    if not (is_boson_SM.any() or is_fermion_SM.any()):
        return 0.0
    # The subtraction depends on the temperature alone once the model is built, and it is
    # evaluated inside ``Vtot``, so it is tabulated once per potential and interpolated.
    spline = _sm_fields_spline(pot, kind, (is_boson_SM, is_fermion_SM), spectrum)
    T_GeV = np.asarray(T, dtype=float) * pot.conversionFactor
    x = np.clip(np.log10(np.where(T_GeV > 0.0, T_GeV, _SM_FIELDS_T_GeV[0])),
                np.log10(_SM_FIELDS_T_GeV[0]), np.log10(_SM_FIELDS_T_GeV[-1]))
    out = spline(x)
    return float(out) if np.ndim(T) == 0 else out


# Temperatures at which the Standard Model fields of a potential are tabulated, in GeV. The
# range covers everything the tabulated baths themselves cover; outside it the contribution is
# constant, zero below and the full count above, so clamping is exact.
_SM_FIELDS_T_GeV = np.logspace(-8.0, 10.0, 2881)


def _sm_fields_spline(pot, kind: str, mask, spectrum):
    """Cubic spline of ``sm_fields_in_potential_geff`` in log10(T/GeV), built once per model."""
    # The table is keyed on everything it is built from, so a model whose vacuum, spectrum or
    # is_SM flags change after it was first evaluated does not keep a stale table.
    X0 = np.atleast_2d(np.asarray(pot.X0, dtype=float))[0]
    key = (kind, float(pot.conversionFactor), int(spectrum.Nscalars), X0.tobytes(),
           np.asarray(spectrum.dof_bosons, dtype=float).tobytes(),
           np.asarray(mask[0], dtype=bool).tobytes(),
           np.asarray(mask[1], dtype=bool).tobytes())
    cache = getattr(pot, "_sm_fields_geff_splines", None)
    if cache is None:
        cache = {}
        try:
            pot._sm_fields_geff_splines = cache
        except Exception:
            cache = None
    if cache is not None and key in cache:
        return cache[key]
    bosons, fermions = pot.boson_massSq(X0, 0.0), pot.fermion_massSq(X0)
    # One ghost per Standard Model gauge boson. The gauge bosons are listed mode by mode,
    # transverse and longitudinal, so their degrees of freedom add up to three per boson;
    # anything else means a mode is missing and the ghosts cannot be counted.
    gauge_entries = np.asarray(spectrum.dof_bosons, dtype=float)[spectrum.Nscalars:]
    gauge_is_SM = np.asarray(mask[0], dtype=bool)[spectrum.Nscalars:]
    gauge_dof = float(np.sum(gauge_entries[gauge_is_SM]))
    if gauge_dof > 0.0 and abs(gauge_dof / 3.0 - round(gauge_dof / 3.0)) > 1e-9:
        raise ValueError(
            "The Standard Model gauge bosons of the potential carry "
            f"{gauge_dof} degrees of freedom, which is not three per gauge boson, so their "
            "ghosts cannot be counted. List every mode of each gauge boson, the two "
            "transverse ones and the longitudinal one."
        )
    fn = {"e": e_geff, "p": p_geff, "s": s_geff}[kind]
    T_internal = _SM_FIELDS_T_GeV / pot.conversionFactor
    values = np.array([float(potential_fields_geff(bosons, fermions, t, kind, mask)
                             - fn(0.0, t, gauge_dof / 3.0, "b")) for t in T_internal])
    spline = interpolate.CubicSpline(np.log10(_SM_FIELDS_T_GeV), values)
    if cache is not None:
        cache[key] = spline
    return spline


def h_eff_radiation(e_geff_fn, p_geff_fn, T: float, CF: float) -> float:
    """Entropy degrees of freedom of a radiation bath given by its energy and pressure degrees of freedom.

    The Standard Model has its own entropy table; for any other bath ``s = (e + p)/T``
    gives ``h = (3 g_e + g_p)/4``.
    """
    if e_geff_fn is e_geffSM and p_geff_fn is p_geffSM:
        return s_geffSM(T, CF)
    return (3.0 * e_geff_fn(T, CF) + p_geff_fn(T, CF)) / 4.0


def sm_bath(pot) -> str:
    """Radiation bath that holds the Standard Model: ``"coupled"`` or ``"decoupled"``.

    ``pot.SM_bath`` decides if set; otherwise the bath whose energy degrees of freedom are the
    Standard Model table ``e_geffSM``, and the coupled one if neither is.
    """
    explicit = getattr(pot, "SM_bath", None)
    if explicit is not None:
        if explicit not in ("coupled", "decoupled"):
            raise ValueError(f"SM_bath must be 'coupled' or 'decoupled', not {explicit!r}.")
        return explicit
    return "decoupled" if pot.kin_decoupled_e_geff is e_geffSM else "coupled"


def reheating_geff(pot, X_broken, T_DS: float, T_dec: float) -> tuple[float, float, float]:
    """Energy and entropy degrees of freedom right after reheating, referred to the SM bath.

    Three baths contribute: the fields of the potential (the transitioning sector, "DS") in
    the broken phase at ``X_broken``, reheated to ``T_DS``; the coupled bath, reheated with
    them; and the decoupled bath, still at ``T_dec``. Each enters with its temperature ratio to
    the bath that holds the Standard Model, ``g = sum g_i (T_i/T_SM)^4`` and
    ``h = sum h_i (T_i/T_SM)^3`` (arXiv:2109.06208, 2311.06346). Returns ``(g, h, T_SM)``.

    The fields of the potential count in the Landau gauge counting, every mode including the
    Goldstone modes and the ghosts of the gauge bosons subtracted, as in
    ``radiationEnergyDensity``. Fields flagged ``is_SM`` count there and are removed from the
    tabulated bath by ``sm_fields_in_potential_geff``, so that each is counted once.
    """
    CF = pot.conversionFactor
    T_c = T_DS
    T_SM = T_dec if sm_bath(pot) == "decoupled" else T_c
    # zero-temperature masses, as in ``radiationEnergyDensity`` and ``bubbledynamics.h_eff_DS``:
    # the daisy resummation is not part of the energy budget of the plasma anywhere else either
    bosons_DS = pot.boson_massSq(X_broken, 0.0)
    fermions_DS = pot.fermion_massSq(X_broken)
    ghosts = pot.mass_spectrum.number_gauge_bosons
    r_DS, r_c, r_dec = T_DS / T_SM, T_c / T_SM, T_dec / T_SM
    # Floored at zero as in ``bubbledynamics.g_eff_DS`` and ``h_eff_DS``: the ghosts are
    # massless in the Landau gauge while the Goldstone modes they cancel are not, so a sector
    # whose modes have all frozen out would otherwise enter the redshift with a negative
    # number of degrees of freedom.
    g_DS = max(float(potential_fields_geff(bosons_DS, fermions_DS, T_DS, "e")
                     - e_geff(0.0, T_DS, ghosts, "b")), 0.0)
    h_DS = max(float(potential_fields_geff(bosons_DS, fermions_DS, T_DS, "s")
                     - s_geff(0.0, T_DS, ghosts, "b")), 0.0)
    # The Standard Model fields of the potential are subtracted from the coupled bath, which
    # is where they are: a potential that contains them cannot put the Standard Model into the
    # decoupled bath, as ``generic_potential._check_radiation_baths`` refuses that setup.
    g = g_DS * r_DS**4 \
        + (pot.kin_coupled_e_geff(T_c, CF) - sm_fields_in_potential_geff(pot, T_c, "e")) * r_c**4 \
        + pot.kin_decoupled_e_geff(T_dec, CF) * r_dec**4
    h = h_DS * r_DS**3 \
        + (h_eff_radiation(pot.kin_coupled_e_geff, pot.kin_coupled_p_geff, T_c, CF)
           - sm_fields_in_potential_geff(pot, T_c, "s")) * r_c**3 \
        + h_eff_radiation(pot.kin_decoupled_e_geff, pot.kin_decoupled_p_geff, T_dec, CF) * r_dec**3
    return float(g), float(h), float(T_SM)
