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
from . import console

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
    if _sm_mass_cap_GeV is not None:
        TSM_GeV = np.minimum(TSM_GeV, _sm_mass_cap_GeV)
    s_val = np.asarray(s_geffSM_fit(TSM_GeV), dtype=float)
    e_val = np.asarray(e_geffSM_fit(TSM_GeV), dtype=float)
    p_base = 4 * s_val - 3 * e_val

    return float(p_base) if np.asarray(p_base).shape == () else p_base


_sm_mass_cap_GeV: float | None = None


def set_sm_temperature_cap(
    CF: float,
    mSq_bosons: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None,
    boson_is_SM: np.ndarray | None = None,
    mSq_fermions: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    fermion_is_SM: np.ndarray | None = None,
    verbose: bool = False,
) -> float | None:
    """Initialise the SM temperature cap from the spectrum at the true minimum.

    Parameters
    ----------
    CF : float
        Conversion factor to GeV.
    mSq_bosons, mSq_fermions : tuple, optional
        Spectra evaluated at the true vacuum.
    verbose : bool, optional
        Print a warning containing the lightest SM mass when saturation is applied.
    """
    global _sm_mass_cap_GeV

    masses = []
    selected_boson_m2 = np.array([], dtype=float)
    selected_fermion_m2 = np.array([], dtype=float)

    # Numerical massless floor:
    # treat masses below one ULP of the characteristic SM m^2 scale as zero.
    # The scale is computed in GeV^2 and mapped back to internal units with CF.
    eps = np.finfo(float).eps

    # Add only physical, non-BSM boson masses to the list
    if mSq_bosons is not None:
        m2b, _, _, physical = mSq_bosons
        mask = (np.asarray(boson_is_SM, dtype=bool)) & np.asarray(physical, dtype=bool)
        if np.any(mask):
            selected = np.asarray(m2b, dtype=float)[..., mask]
            selected_boson_m2 = np.abs(selected).ravel()

    # Add only non-BSM fermion masses to the list
    if mSq_fermions is not None:
        m2f, _ = mSq_fermions
        mask = np.asarray(fermion_is_SM, dtype=bool)
        if np.any(mask):
            selected = np.asarray(m2f, dtype=float)[..., mask]
            selected_fermion_m2 = np.abs(selected).ravel()

    selected_boson_m2 = selected_boson_m2[np.isfinite(selected_boson_m2)]
    selected_fermion_m2 = selected_fermion_m2[np.isfinite(selected_fermion_m2)]

    if selected_boson_m2.size:
        m2_scale_b_internal = float(np.max(selected_boson_m2))
        m2_scale_b_GeV2 = m2_scale_b_internal * CF * CF
        m2_tol_b_internal = (eps * m2_scale_b_GeV2) / (CF * CF)
    else:
        m2_tol_b_internal = 0.0

    if selected_fermion_m2.size:
        m2_scale_f_internal = float(np.max(selected_fermion_m2))
        m2_scale_f_GeV2 = m2_scale_f_internal * CF * CF
        m2_tol_f_internal = (eps * m2_scale_f_GeV2) / (CF * CF)
    else:
        m2_tol_f_internal = 0.0

    # Collect non-massless states using the numerical floor.
    if mSq_bosons is not None:
        m2b, _, _, physical = mSq_bosons
        mask = (np.asarray(boson_is_SM, dtype=bool)) & np.asarray(physical, dtype=bool)
        if np.any(mask):
            selected = np.asarray(m2b, dtype=float)[..., mask]
            masses.extend(np.sqrt(selected[selected > m2_tol_b_internal]).ravel())

    if mSq_fermions is not None:
        m2f, _ = mSq_fermions
        mask = np.asarray(fermion_is_SM, dtype=bool)
        if np.any(mask):
            selected = np.asarray(m2f, dtype=float)[..., mask]
            masses.extend(np.sqrt(selected[selected > m2_tol_f_internal]).ravel())

    if masses:
        lightest = float(np.min(masses))
        previous_cap = _sm_mass_cap_GeV
        _sm_mass_cap_GeV = 0.5 * lightest * CF
        if verbose and previous_cap is None:
            msg = (f"The lightest (non-massless) SM particle in the model "
                   f"file has a mass of {lightest * CF:.3g} GeV. The tabulated SM "
                   f"effective degrees of freedom will hence be capped at "
                   f"{_sm_mass_cap_GeV:.3g} GeV. The model will compute the "
                   f"effective degrees of freedom dynamically beyond that point.")
            console.print(
                f"[yellow]Warning: {msg}[/yellow]"
            )
    else:
        _sm_mass_cap_GeV = None

    return _sm_mass_cap_GeV


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
    capped_T = TSM_arr * CF
    if _sm_mass_cap_GeV is not None:
        capped_T = np.minimum(capped_T, _sm_mass_cap_GeV)
    if mode == "smooth":
        result = np.asarray(Ie_geffSM_smooth(np.log10(capped_T)), dtype=float)
    elif mode == "data":
        result = np.asarray(Ie_geffSM(np.log10(capped_T)), dtype=float)
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
    capped_T = TSM_arr * CF
    if _sm_mass_cap_GeV is not None:
        capped_T = np.minimum(capped_T, _sm_mass_cap_GeV)
    if mode == "smooth":
        result = np.asarray(Is_geffSM_smooth(np.log10(capped_T)), dtype=float)
    elif mode == "data":
        result = np.asarray(Is_geffSM(np.log10(capped_T)), dtype=float)
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
    if _sm_mass_cap_GeV is not None:
        TSM_GeV = np.minimum(TSM_GeV, _sm_mass_cap_GeV)
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
    """Pressure degrees of freedom, supports scalar or vector inputs for m and T."""
    if T == 0.0:
        return 0.0
    x = m/T
    res = 0.0
    if ptype == "b":
        if x < Ipb.x[0]:
            res =  1.0
        elif x < Ipb.x[-1]:
            res = Ipb(x)
    elif ptype == "f":
        if x < Ipf.x[0]:
            res =  (7/8)
        elif x < Ipf.x[-1]:
            res = Ipf(x)
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
