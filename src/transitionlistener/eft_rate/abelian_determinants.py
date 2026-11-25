"""Functional determinants around a bounce, by the Gel'fand-Yaglom method.

Given a fluctuation operator on a spherically symmetric background, this returns
the ratio of its determinant to the same operator in the false vacuum, partial
wave by partial wave.

BubbleDet solves one field at a time, and on a bounce the longitudinal gauge
mode mixes with the Goldstone through a term proportional to phi'(r) that no
r-independent field redefinition removes. :func:`gy_single` overlaps with
BubbleDet; :func:`gy_mixed` solves the coupled block.

Both solve for the ratio ``T = psi_bounce / psi_falsevacuum`` rather than for
the solutions themselves, because those grow like ``exp(m r)`` and dividing two
enormous numbers at large radius loses all precision. ``T`` stays of order one
and tends to a constant.

The returned ``T_l`` IS the determinant ratio in that partial wave, and
``ln det = sum_l (2l+1) ln T_l`` in three dimensions. That sum does not
converge on its own -- it carries the usual ultraviolet divergence, since large
``l`` means short wavelength -- so a renormalisation is needed before an
absolute determinant can be quoted. A *difference* of two determinants sharing
the same large-``l`` behaviour, such as switching the gauge-Goldstone mixing on
and off, is finite without any subtraction, which is why that particular
quantity can be computed directly.

Renormalisation
---------------
Each single channel's partial-wave sum ``sum_l (2l+1) ln T_l`` is ultraviolet
divergent and needs renormalising before an absolute determinant can be quoted.
The absolute determinant therefore comes in two parts:

* the *factorised* part -- every channel treated independently -- goes through
  BubbleDet, which renormalises it, in the channels that :func:`abelian_higgs_3d.nlo_assembly`
  evaluates;
* the *mixing* correction, the difference between the coupled block and the
  same three channels factorised, comes from :func:`log_gy_mixed` through
  :func:`abelian_higgs_3d.mixing_log_det` and :func:`abelian_higgs_3d.j0_sector`, and is finite without
  any subtraction, because the divergence is common to both.

:func:`abelian_higgs_3d.nlo_assembly` combines the two. Summing the raw single channel logs
returned here would produce a divergent, meaningless number.

Conventions
-----------
The coupled operator is the R_xi system at xi = 1, and for a charge-one scalar
in an Abelian theory the gauge diagonal entry is ``W_X = m_X^2 = g^2 phi^2``.
The SU(2) literature carries ``4 m_X^2`` there because ``m_X = g phi / 2``;
copying that expression into an Abelian model is a silent factor of four.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.special import gammaln, ive

from transitionlistener.compiled.mixing import (_C, _E, _MU_COMPLEX, _MU_REAL, _P, _T, _TI, _TI_COMPLEX, _TI_REAL,
                                                 FAILED_DEPENDENT, FAILED_DET, FAILED_STEP, FAILED_VALUE, OK,
                                                 _log_det, _q_nodes)


class FluctuationSolverError(RuntimeError):
    """The partial-wave solve did not converge, so no determinant is available."""



def gy_single(dW, l, rmax, m_fv, rmin=1e-4, rtol=1e-10, atol=1e-12):
    """det ratio for -d^2_l + W(r) versus -d^2_l + W(inf), in d = 3.

    Solves   T'' + U T' - dW T = 0,  U = 2(l+1)/r + 2 m_fv I'/I,
    with T(rmin) = 1, T'(rmin) = 0, and returns T(rmax).

    dW(r) is W(r) - W(inf); m_fv is the false-vacuum mass sqrt(W(inf)).
    """
    from scipy.special import ive

    def U(r):
        # U = 2/r + 2 psi_FV'/psi_FV  with  psi_FV = r^{-1/2} I_{l+1/2}(m r),
        # so psi_FV'/psi_FV = -1/(2r) + m I'/I  and  U = 1/r + 2 m I'/I.
        # The ratio comes from the continued fraction, not from two Bessel
        # values: at high l those underflow to zero and the ratio becomes 0/0.
        if m_fv <= 0.0:
            return 2.0 * (l + 1.0) / r
        return 1.0 / r + 2.0 * m_fv * _iv_log_deriv(l + 0.5, m_fv * r)

    def rhs(r, y):
        return [y[1], -U(r) * y[1] + dW(r) * y[0]]

    s = solve_ivp(rhs, (rmin, rmax), [1.0, 0.0], rtol=rtol, atol=atol, dense_output=True)
    return float(s.y[0, -1])


def _log_ive(nu: float, z: float) -> float:
    """``ln(I_nu(z) e^{-z})``, stable where ``ive`` itself underflows to zero.

    For large order and modest argument -- exactly the high partial waves on an
    extended bounce -- ``I_nu(z)`` is far below the smallest representable
    double and ``scipy``'s ``ive`` returns zero, so any ratio built from it
    becomes 0/0. The small-argument form ``I_nu(z) ~ (z/2)^nu / Gamma(nu+1)``
    stays perfectly well behaved in logarithms.
    """
    from scipy.special import ive, gammaln

    val = ive(nu, z)
    if val > 0.0 and np.isfinite(val):
        return float(np.log(val))
    return float(nu * np.log(z / 2.0) - gammaln(nu + 1.0) - z)


def _iv_log_deriv(nu: float, z: float, terms: int = 400) -> float:
    """``I'_nu(z) / I_nu(z)``, via a backward continued fraction.

    Uses ``I'_nu/I_nu = I_{nu+1}/I_nu + nu/z`` with the ratio evaluated as the
    continued fraction

        I_{nu+1}/I_nu = 1 / (2(nu+1)/z + 1 / (2(nu+2)/z + ...)) ,

    which is stable for every order and argument, unlike forming the ratio from
    two underflowing Bessel values.
    """
    if z <= 0.0:
        return float("inf")
    frac = 0.0
    for k in range(terms, 0, -1):
        frac = 1.0 / (2.0 * (nu + k) / z + frac)
    return float(frac + nu / z)


def _fv_G(l, m):
    """h(r) = f_G(r)/r^l for the massive false-vacuum Goldstone solution,
    f_G = 2^(l+1/2) m^(-l-1/2) Gamma(l+3/2) I_{l+1/2}(m r)/sqrt(r), normalised
    so that f_G -> r^l as r -> 0. Returns (h, U_G)."""
    from scipy.special import ive, gammaln
    nu = l + 0.5
    if m <= 0.0:
        return (lambda r: 1.0), (lambda r: 2.0 * (l + 1.0) / r)
    logC = (l + 0.5) * np.log(2.0) - nu * np.log(m) + gammaln(l + 1.5)

    def log_h(r):
        """``ln h(r)``; the caller exponentiates only where it is safe."""
        z = m * r
        return logC + z + _log_ive(nu, z) - (l + 0.5) * np.log(r)

    def h(r):
        return float(np.exp(log_h(r)))

    def U(r):
        # I'_nu/I_nu from the continued fraction: stable at every order
        return 1.0 / r + 2.0 * m * _iv_log_deriv(nu, m * r)

    h.log = log_h

    return h, U


def _mixed_rhs_core(l, WX, mX_prime, dWG, mG_fv):
    """The coupled (a_S, a_L, a_G) system, shared by every solver that uses it.

    Returned as a function of ``(r, TS, TL, TG, dS, dL, dG)`` giving the three
    second derivatives, written so that it works elementwise on scalars or on
    arrays of solution columns.

    The operator is, with ``c_-/c_+ = sqrt(l/(2l+1))/sqrt((l+1)/(2l+1))``,

        [ -nabla_(l-1) + W_X        0                 2 c_-  m_X'  ]
        [      0            -nabla_(l+1) + W_X       -2 c_+  m_X'  ]
        [  2 c_- m_X'       -2 c_+ m_X'      -nabla_l + W_G + W_X  ]

    in R_xi gauge at xi = 1, for a charge-one scalar in an Abelian theory, so
    that ``W_X = m_X^2 = g^2 phi^2``. The factor of two in the off-diagonal is
    not a convention: :func:`gauge_orbit_residual` fixes it from the gauge
    structure, and it fails with any other value.
    """
    cm = np.sqrt(l / (2.0 * l + 1.0))
    cp = np.sqrt((l + 1.0) / (2.0 * l + 1.0))
    US = lambda r: 2.0 * l / r                      # f_S = r^(l-1)
    UL = lambda r: 2.0 * (l + 2.0) / r              # f_L = r^(l+1)
    h, UG = _fv_G(l, mG_fv)                         # f_G = r^l h(r)

    def dd(r, TS, TL, TG, dS, dL, dG):
        w, p, g = WX(r), mX_prime(r), dWG(r)
        hh = h(r)
        # f_G/f_S = r h ; f_G/f_L = h/r ; f_S/f_G = 1/(r h) ; f_L/f_G = r/h
        ddS = -US(r) * dS + w * TS + 2.0 * cm * p * (r * hh) * TG
        ddL = -UL(r) * dL + w * TL - 2.0 * cp * p * (hh / r) * TG
        ddG = (-UG(r) * dG + (g + w) * TG
               + 2.0 * cm * p * (1.0 / (r * hh)) * TS
               - 2.0 * cp * p * (r / hh) * TL)
        return ddS, ddL, ddG

    return dd, (cm, cp, h)


def gy_mixed(WX, mX_prime, dWG, l, rmax, mG_fv, rmin=1e-4, rtol=1e-10, atol=1e-13):
    """Coupled (a_S, a_L, a_G) determinant ratio, R_xi gauge with xi = 1, d = 3.

    Fluctuation operator, cross-checked against Ekstedt arXiv:2104.11804
    (Abelian case) and rotated with the matrix R of Kierkla's thesis:

        [ -nabla_l^- + W_X        0                 2 c_-  m_X'          ]
        [      0            -nabla_l^+ + W_X       -2 c_+  m_X'          ]
        [  2 c_- m_X'       -2 c_+ m_X'      -d^2_l + W_G + W_X          ]

    c_-/c_+ = sqrt(l/(2l+1)) / sqrt((l+1)/(2l+1)).  For a charge-one scalar in an
    Abelian theory W_X = m_X^2 = g^2 phi^2.  (In the SU(2) literature one finds
    W_X = 4 m_X^2 because there m_X = g phi / 2 -- do not copy that factor here.)

    Everything is solved for T_alpha = a_alpha / a_alpha,FV, which is what keeps
    the integration bounded: the raw solutions grow like exp(m r) and r^(l+1).

    Arguments WX, mX_prime, dWG are callables of r; dWG(r) = W_G(r) - W_G(inf).
    Returns det of the 3x3 matrix of T values at rmax, i.e. the determinant ratio.
    """
    dd, _ = _mixed_rhs_core(l, WX, mX_prime, dWG, mG_fv)

    def rhs(r, y):
        TS, TL, TG, dS, dL, dG = y
        ddS, ddL, ddG = dd(r, TS, TL, TG, dS, dL, dG)
        return [dS, dL, dG, ddS, ddL, ddG]

    cols = []
    for k in range(3):
        y0 = [0.0]*6
        y0[k] = 1.0
        s = solve_ivp(rhs, (rmin, rmax), y0, rtol=rtol, atol=atol, method="Radau")
        if not s.success or not np.all(np.isfinite(s.y[:3, -1])):
            # High partial waves on a physical bounce can overflow: the
            # centrifugal term makes the system stiff and the raw solutions
            # span many orders of magnitude. Report rather than return a
            # number that silently poisons the partial-wave sum.
            raise FluctuationSolverError(
                f"coupled solve failed at l={l}: {s.message if not s.success else 'non-finite result'}"
            )
        cols.append(s.y[:3, -1])
    return float(np.linalg.det(np.array(cols).T))


def log_det_difference_from_logs(log_t_on, log_t_off, ells, tail_from: int = 12,
                                 max_tail_fraction: float = 0.15) -> dict:
    """As :func:`log_det_difference`, but taking logarithms of the ratios.

    This is the form to use with :func:`log_gy_mixed` and
    :func:`log_gy_single`, which return logarithms precisely because the ratios
    themselves are not representable at high partial waves.
    """
    return _summarise_partial_waves(
        np.asarray(log_t_on, dtype=float) - np.asarray(log_t_off, dtype=float),
        ells, tail_from, max_tail_fraction)


def log_det_difference(t_on, t_off, ells, tail_from: int = 12,
                       max_tail_fraction: float = 0.15) -> dict:
    """``Delta ln det`` between two sets of partial-wave ratios.

    Sums ``(2l+1) [ln T_l^on - ln T_l^off]`` over the supplied waves and
    estimates the tail beyond the largest by fitting a power law to the last
    few points and integrating it.

    A difference is used rather than an absolute determinant deliberately. The
    ultraviolet divergence of ``ln det`` is identical in the two cases whenever
    they share the same short-distance behaviour -- switching the
    gauge-Goldstone mixing on and off is the case of interest -- so it cancels
    and no renormalisation is required. Quoting an absolute determinant needs
    the subtraction that this function does not do.

    Returns a dict with ``partial`` (the sum over the supplied waves), ``tail``
    (the extrapolated remainder), ``total``, ``exponent`` (the fitted power,
    which must be steeper than -1 for the sum to converge at all),
    ``tail_fraction``, and ``converged``.

    **Check ``converged`` before using ``total``.** When the summand falls
    slowly the extrapolated tail can be a large fraction of the answer, and
    then ``total`` describes the fit rather than the determinant. On a
    thick-walled bounce the summand can fall off much more slowly than on a
    thin-walled one, and the tail then carries a large share of the total --
    so the same machinery is well converged in one case and not in the
    other, and only the flag distinguishes them.
    """
    return _summarise_partial_waves(
        np.log(np.asarray(t_on, dtype=float)) - np.log(np.asarray(t_off, dtype=float)),
        ells, tail_from, max_tail_fraction)


def _summarise_partial_waves(dlog, ells, tail_from, max_tail_fraction) -> dict:
    """Sum ``(2l+1) dlog`` and extrapolate the tail; shared by both entry points.

    ``ells`` must be *consecutive* integers. The sum being computed runs over
    every partial wave, so a caller who samples sparsely -- 1, 2, 3, 4, 6, 8,
    12, 16, ... -- gets a number that looks like a partial sum and is an
    undercount, because most of the weight sits at low ``l`` where the gaps
    are. So this is checked. Sample sparsely if you must, but then
    interpolate onto the integers before calling this.
    """
    ells = np.asarray(ells, dtype=float)
    dlog = np.asarray(dlog, dtype=float)
    if ells.size >= 2:
        steps = np.diff(ells)
        if not np.allclose(steps, 1.0):
            raise ValueError(
                "the partial-wave sum runs over every l, so `ells` must be "
                "consecutive integers; got steps "
                f"{np.unique(np.round(steps, 6))[:6]}. Interpolate onto the "
                "integers first, or the partial sum silently undercounts."
            )
    contrib = (2.0 * ells + 1.0) * dlog
    partial = float(np.sum(contrib))

    sel = ells >= tail_from
    if np.count_nonzero(sel) < 2 or np.any(contrib[sel] == 0.0):
        return {"partial": partial, "tail": float("nan"),
                "total": float("nan"), "exponent": float("nan"),
                "tail_fraction": float("nan"), "converged": False}

    slope, intercept = np.polyfit(np.log(ells[sel]), np.log(np.abs(contrib[sel])), 1)
    if slope >= -1.0:
        # the sum does not converge; report rather than return a number
        return {"partial": partial, "tail": float("inf"),
                "total": float("inf"), "exponent": float(slope),
                "tail_fraction": float("inf"), "converged": False}
    amplitude = np.exp(intercept)
    lmax = float(ells[-1])
    tail = np.sign(contrib[-1]) * amplitude * lmax ** (slope + 1.0) / (-slope - 1.0)
    total = partial + tail

    # How much of the answer is extrapolation rather than computation? When the
    # summand falls slowly the tail can dominate, and then the "total" is a
    # statement about the fit, not about the determinant. Report it so a caller
    # cannot mistake one for the other.
    tail_fraction = abs(tail) / abs(total) if total != 0.0 else float("inf")
    converged = bool(tail_fraction <= max_tail_fraction)
    return {"partial": partial, "tail": float(tail), "total": float(total),
            "exponent": float(slope), "tail_fraction": float(tail_fraction),
            "converged": converged}


# ----------------------------------------------------------------------
# logarithmic formulations, for partial waves where T spans many decades
# ----------------------------------------------------------------------
def log_gy_single(dW, l, rmax, m_fv, rmin=1e-4, rtol=1e-10, atol=1e-12):
    r"""``ln T_l(rmax)`` for one field, integrating the logarithm directly.

    On a physical bounce the raw ratio can span nine orders of magnitude across
    the partial waves that matter, which costs precision and eventually
    overflows the stiff integrator. Substituting ``u = ln T`` turns

    .. math::  T'' + U T' - \Delta W\,T = 0

    into a Riccati equation for ``v = u'``,

    .. math::  u' = v, \qquad v' = -v^2 - U v + \Delta W ,

    whose solution stays of order ``ln`` of the original and integrates
    comfortably. Initial conditions ``u = v = 0`` at ``rmin`` correspond to the
    ``T = 1``, ``T' = 0`` of the direct form.

    Not usable where ``T`` crosses zero, since ``ln T`` diverges there. In
    practice that is the ``l = 1`` wave, which contains the translational zero
    modes and is excluded from the sum anyway.
    """
    from scipy.special import ive

    def U(r):
        # U = 2/r + 2 psi_FV'/psi_FV  with  psi_FV = r^{-1/2} I_{l+1/2}(m r),
        # so psi_FV'/psi_FV = -1/(2r) + m I'/I  and  U = 1/r + 2 m I'/I.
        # The ratio comes from the continued fraction, not from two Bessel
        # values: at high l those underflow to zero and the ratio becomes 0/0.
        if m_fv <= 0.0:
            return 2.0 * (l + 1.0) / r
        return 1.0 / r + 2.0 * m_fv * _iv_log_deriv(l + 0.5, m_fv * r)

    def rhs(r, y):
        u, v = y
        return [v, -v * v - U(r) * v + dW(r)]

    s = solve_ivp(rhs, (rmin, rmax), [0.0, 0.0], rtol=rtol, atol=atol)
    if not s.success or not np.all(np.isfinite(s.y[:, -1])):
        raise FluctuationSolverError(
            f"logarithmic single-channel solve failed at l={l}: {s.message}"
        )
    return float(s.y[0, -1])


#: Nodes of the grid of a mass function per unit of ln r, its largest size, and the error allowed between nodes as a
#: fraction of the largest value of the function. Nodes of the per-wave grid per unit of ln r.
PROFILE_NODES_PER_E = 1000
PROFILE_N_MAX = 1 << 18
PROFILE_TOLERANCE = 1e-9
WAVE_NODES_PER_E = 100


def _hermite(f, df, dx):
    """Coefficients c[k, i] of the cubic Hermite interpolant (highest power first, in x - x_i) on a uniform grid."""
    c = np.empty((4, f.size - 1))
    slope = np.diff(f) / dx
    c[3] = f[:-1]
    c[2] = df[:-1]
    c[1] = (3.0 * slope - 2.0 * df[:-1] - df[1:]) / dx
    c[0] = (df[:-1] + df[1:] - 2.0 * slope) / dx ** 2
    return c


def _sample(fun, r):
    """fun at the radii r; the mixing callables of abelian_higgs_3d carry an array form in fun.array."""
    array = getattr(fun, "array", None)
    return np.asarray(array(r), dtype=float) if array is not None else np.array([float(fun(q)) for q in r])


class _Profile:
    """One callable of r as a cubic spline in ln r on a grid uniform in ln r, of at least n nodes.
    c = None if the callable is not represented on any grid up to PROFILE_N_MAX."""

    def __init__(self, fun, rmin, rmax, n):
        self.fun, self.rmin, self.rmax = fun, rmin, rmax
        lo, hi = np.log(rmin), np.log(rmax)
        while n <= PROFILE_N_MAX:
            x = np.linspace(lo, hi, n)
            r = np.exp(x)
            r[0], r[-1] = rmin, rmax
            val = _sample(fun, r)
            if np.all(np.isfinite(val)):
                spl = CubicSpline(x, val)
                mid = 0.5 * (x[:-1] + x[1:])[3::16]         # one cell in 16
                exact = _sample(fun, np.exp(mid))
                big = max(np.max(np.abs(val)), np.max(np.abs(exact)))
                if np.max(np.abs(spl(mid) - exact)) <= PROFILE_TOLERANCE * big:     # False for nan
                    self.c, self.n = np.array(spl.c), n
                    return
            n = 2 * n - 1
        self.c, self.n = None, None


_profile_cache = []         # the last few tables, keyed by the identity of the callable


def _profile(fun, rmin, rmax, n=None):
    """The table of fun: its own grid if n is None, else exactly n nodes (for a grid shared with a finer callable)."""
    for p in _profile_cache:
        if p.fun is fun and p.rmin == rmin and p.rmax == rmax and (p.n == n or n is None):
            return p
    n0 = max(64, int(PROFILE_NODES_PER_E * np.log(rmax / rmin))) + 1
    p = _Profile(fun, rmin, rmax, n0 if n is None else n)
    _profile_cache.append(p)
    del _profile_cache[:-12]
    return p


_wave_cache = {}


def _wave(l, m, rmin, rmax):
    """(n, ln h coefficients, Q coefficients) on the wave grid; the last wave is kept for its on and off solves."""
    key = (l, m, rmin, rmax)
    if key in _wave_cache:
        return _wave_cache[key]
    n = max(64, int(WAVE_NODES_PER_E * np.log(rmax / rmin))) + 1
    x = np.linspace(np.log(rmin), np.log(rmax), n)
    dx = x[1] - x[0]
    nu = l + 0.5
    if m > 0.0:
        z = m * np.exp(x)
        q = _q_nodes(nu, z)
        val = ive(nu, z)
        good = (val > 0.0) & np.isfinite(val)
        log_ive = np.where(good, np.log(np.where(good, val, 1.0)), nu * np.log(z / 2.0) - gammaln(nu + 1.0) - z)
        log_c = (l + 0.5) * np.log(2.0) - nu * np.log(m) + gammaln(l + 1.5)
        ln_h = log_c + z + log_ive - nu * x
        dq = z * z + nu * nu - q * q
        dh = q - nu
    else:
        q, dq, ln_h, dh = np.full(n, nu), np.zeros(n), np.zeros(n), np.zeros(n)
    out = (n, _hermite(ln_h, dh, dx), _hermite(q, dq, dx))
    _wave_cache.clear()
    _wave_cache[key] = out
    return out


def _log_gy_mixed_on_callables(WX, mX_prime, dWG, l, rmax, mG_fv, rmin, rtol, atol, n_chunks):
    """:func:`log_gy_mixed` with scipy's Radau calling WX, mX_prime and dWG, for functions no table represents."""
    dd, _ = _mixed_rhs_core(l, WX, mX_prime, dWG, mG_fv)

    def rhs(r, y):
        TS, TL, TG, dS, dL, dG = y.reshape(6, 3)
        ddS, ddL, ddG = dd(r, TS, TL, TG, dS, dL, dG)
        return np.vstack([dS, dL, dG, ddS, ddL, ddG]).ravel()

    Y = np.zeros((6, 3))
    for k in range(3):
        Y[k, k] = 1.0
    log_scale = 0.0
    edges = np.geomspace(rmin, rmax, n_chunks + 1)

    for a, b in zip(edges[:-1], edges[1:]):
        s = solve_ivp(rhs, (a, b), Y.ravel(), rtol=rtol, atol=atol, method="Radau")
        if not s.success or not np.all(np.isfinite(s.y[:, -1])):
            raise FluctuationSolverError(
                f"logarithmic coupled solve failed at l={l} on [{a:.4g},{b:.4g}]: "
                f"{s.message if not s.success else 'non-finite result'}"
            )
        Q, R = np.linalg.qr(s.y[:, -1].reshape(6, 3))
        diag = np.diag(R)
        if not np.all(diag != 0.0):
            raise FluctuationSolverError(f"the solution columns became dependent at l={l}")
        Y = Q * np.sign(diag)              # orthonormal columns, orientation kept
        log_scale += float(np.sum(np.log(np.abs(diag))))

    det_scaled = float(np.linalg.det(Y[:3]))
    if det_scaled == 0.0:
        raise FluctuationSolverError(f"rescaled determinant vanished at l={l}")
    return float(np.log(abs(det_scaled)) + log_scale)


def log_gy_mixed(WX, mX_prime, dWG, l, rmax, mG_fv, rmin=1e-4,
                 rtol=1e-10, atol=1e-13, n_chunks: int = 200):
    r"""``ln`` of the coupled 3x3 determinant ratio, by rescaled integration.

    The logarithm cannot simply be substituted here, because the object is the
    determinant of a matrix of solutions rather than a single function. Instead
    the interval is integrated in chunks and, after each, the three solution
    columns are re-orthonormalised (QR) with the logarithms of the diagonal of
    R accumulated. The determinant of the final matrix times the exponential of
    the accumulated logarithms is the answer, and no intermediate ever grows
    large or loses the independence of its columns.

    The chunks are spaced geometrically from ``rmin`` to ``rmax``, because the
    bounce core of a strongly supercooled transition, where ``rmax`` lies many
    orders of magnitude above the core radius, needs chunks of its own: inside one solve, the solution is lost
    above ``l ~ 30`` and the summand of the mixing sum grows like ``l^6`` instead
    of falling like ``l^-2``.

    This is what makes high partial waves reachable on a physical bounce, where
    the direct form of :func:`gy_mixed` overflows around ``l = 40``.

    The integration is compiled. WX, mX_prime and dWG are sampled once per bounce on a grid uniform in ln r and read
    from a cubic spline in ln r, and ln h(r) and r U_G(r) of the false-vacuum Goldstone solution are tabulated per wave
    with their exact derivatives in ln r (d ln h/dx = Q - l - 1/2, dQ/dx = (m r)^2 + (l + 1/2)^2 - Q^2, Q = m r I'/I).
    A value taken between the nodes checks that the spline represents the callable, and the grid is doubled if it does
    not. A callable that no grid up to ``PROFILE_N_MAX`` represents, a kinked one for example, is integrated in Python.

    Returns ``ln|det|``; the determinant is positive in the cases of interest.
    """
    rmin, rmax = float(rmin), float(rmax)
    tables = [_profile(f, rmin, rmax) for f in (WX, mX_prime, dWG)] if mG_fv >= 0.0 and rmax > rmin else [None]
    if None in tables or any(table.c is None for table in tables):
        return _log_gy_mixed_on_callables(WX, mX_prime, dWG, l, rmax, mG_fv, rmin, rtol, atol, n_chunks)
    n = max(table.n for table in tables)
    WX_t, mX_t, dWG_t = (table if table.n == n else _profile(table.fun, rmin, rmax, n) for table in tables)
    x0 = np.log(rmin)
    span = np.log(rmax) - x0
    n_wave, ln_h, Q = _wave(int(l), float(mG_fv), rmin, rmax)
    edges = np.geomspace(rmin, rmax, n_chunks + 1)
    stats = np.zeros(4, dtype=np.int64)
    code, value, chunk = _log_det(edges, float(rtol), float(atol), x0, (n - 1) / span, n, WX_t.c, mX_t.c, dWG_t.c,
                                  (n_wave - 1) / span, n_wave, ln_h, Q, float(l), _C, _E, _T, _TI, _TI_REAL,
                                  _TI_COMPLEX, _P, _MU_REAL, _MU_COMPLEX, stats)
    if code == OK:
        return float(value)
    a, b = edges[chunk], edges[chunk + 1]
    raise FluctuationSolverError(
        {FAILED_STEP: f"logarithmic coupled solve failed at l={l} on [{a:.4g},{b:.4g}]: "
                      "Required step size is less than spacing between numbers.",
         FAILED_VALUE: f"logarithmic coupled solve failed at l={l} on [{a:.4g},{b:.4g}]: non-finite result",
         FAILED_DEPENDENT: f"the solution columns became dependent at l={l}",
         FAILED_DET: f"rescaled determinant vanished at l={l}"}[code])


# ----------------------------------------------------------------------
# which channels make up the vector determinant, and with what weight
# ----------------------------------------------------------------------
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Channel:
    """One fluctuation channel: its operator, its multiplicity and its sign.

    ``orbital`` is the orbital angular momentum in terms of the total ``l``,
    which fixes the centrifugal term. ``weight`` counts degrees of freedom and
    carries the sign: ordinary bosons contribute ``+1`` each, and the
    Faddeev-Popov ghosts ``-2``, being a complex anticommuting pair.
    """

    name: str
    orbital: Callable[[int], int]
    weight: float
    mass_sq: Callable[[float], float]
    note: str = ""


def abelian_vector_channels(phi3, g3sq: float, m_debye_sq: float,
                            xi: float = 1.0, include_A0: bool = True):
    """The channels making up ``det_V`` for an Abelian Higgs in three dimensions.

    ``phi3`` is the background as a callable of radius.

    The three components of the gauge field split, at each total angular
    momentum, into two "electric" harmonics of orbital ``l-1`` and ``l+1`` --
    which mix with the Goldstone and are therefore handled together by
    :func:`log_gy_mixed`, not here -- and one "magnetic" harmonic of orbital
    ``l``, which does not mix and is an ordinary single channel. That magnetic
    mode is the transverse piece returned here.

    Degree-of-freedom counting, which is the thing worth checking: three gauge
    components plus one Goldstone minus two ghosts leaves two, matching the two
    transverse polarisations used to build the barrier in the leading-order
    potential. If that identity fails, the bookkeeping is wrong somewhere.

    At ``xi = 1`` the ghost mass equals the transverse gauge mass, so those two
    channels share an operator and their weights simply add to ``-1``. The
    returned list keeps them separate anyway, so that a different gauge can be
    used without rewriting the assembly.
    """
    mX_sq = lambda r: g3sq * float(phi3(r)) ** 2

    channels = [
        Channel("transverse", lambda l: l, +1.0, mX_sq,
                "magnetic vector harmonic; does not mix with the scalars"),
        Channel("ghost", lambda l: l, -2.0, lambda r: xi * mX_sq(r),
                "complex anticommuting Faddeev-Popov pair"),
    ]
    if include_A0:
        channels.append(
            Channel("temporal", lambda l: l, +1.0,
                    lambda r: m_debye_sq + mX_sq(r),
                    "A_0, present while the soft theory has not been reduced further")
        )
    return channels


def net_degrees_of_freedom(channels, n_mixed_gauge: int = 2,
                           n_mixed_scalar: int = 1) -> float:
    """Total degrees of freedom, including the mixed block handled elsewhere.

    ``n_mixed_gauge`` and ``n_mixed_scalar`` are the gauge and scalar channels
    inside the coupled block: for the Abelian case two electric harmonics and
    one Goldstone. The temporal mode is excluded from the physical count, since
    it is not a propagating polarisation.
    """
    from_channels = sum(c.weight for c in channels if c.name != "temporal")
    return float(from_channels + n_mixed_gauge + n_mixed_scalar)


def gauge_orbit_residual(WX, mX_prime, dWG, l, rmax, mG_fv, rmin=1e-4,
                         rtol=1e-11, atol=1e-14, mixing_factor: float = 2.0):
    r"""How nearly the coupled operator annihilates a pure-gauge direction.

    A gauge transformation with parameter :math:`\omega` moves the fields by
    :math:`\delta a_i = \partial_i\omega`, :math:`\delta\chi = g\phi\,\omega`.
    That direction is pure gauge, so what the gauge-fixed operator does to it is
    not free: it must reproduce the Faddeev-Popov operator. In particular, if
    :math:`\omega` solves the FP equation

    .. math:: -\nabla^2\omega + \xi g^2\phi^2\omega = 0 ,

    then :math:`(\partial_i\omega,\, g\phi\,\omega)` is an *exact* zero mode of
    the coupled system. Decomposed into the channels this module uses, and with
    the normalisation of the two electric harmonics fixed by their angular
    integrals, that mode is

    .. math::
        a_S = c_-\left(\omega' + \tfrac{l+1}{r}\omega\right), \quad
        a_L = -c_+\left(\omega' - \tfrac{l}{r}\omega\right), \quad
        a_G = g\phi\,\omega .

    This is the sharpest available check on the coupled block, because the mode
    is annihilated only if the off-diagonal coefficient, both Clebsch-Gordan
    factors, the three orbital assignments, the Goldstone mass and the ghost
    mass are *simultaneously* right -- and because it uses the bounce equation
    of motion, so it also confirms the operator is being applied on shell. It is
    independent of the comparisons with BubbleDet and the sparse LU, which test
    the implementation of an operator derived from the same Lagrangian; neither
    can catch a term derived correctly and then dropped.

    ``mixing_factor`` exists so the test can be made to fail on purpose: the
    correct value is 2, and the residual is O(1) for any other.

    Note ``xi = 1`` throughout, which is where this module's operator is defined.
    At other gauge parameters the vector kinetic term acquires a
    :math:`(1-1/\xi)\partial_i\partial_j` piece that is not in this operator,
    and the residual grows linearly in ``xi - 1``.

    Returns a dict with ``residual`` (the worst relative row residual over the
    sampled radii), ``rows`` (the worst residual per row) and ``sampled``.
    """
    dd, (cm, cp, _h) = _mixed_rhs_core(l, WX, mX_prime, dWG, mG_fv)
    L2 = l * (l + 1.0)

    # regular solution of the FP equation; xi = 1 makes the ghost mass W_X
    def fp(r, u):
        return [u[1], -2.0 * u[1] / r + L2 * u[0] / r**2 + WX(r) * u[0]]

    r0 = max(rmin, 1e-3 * rmax)
    # the equation is linear and homogeneous, so start from omega(r0) = 1
    # rather than r0**l, which overflows for large l on an extended bounce
    sol = solve_ivp(fp, (r0, rmax), [1.0, l / r0],
                    rtol=rtol, atol=atol, method="Radau", dense_output=True)
    if not sol.success:
        raise FluctuationSolverError(
            f"Faddeev-Popov solve failed at l={l}: {sol.message}")

    # g*phi and its derivatives, from W_X = g^2 phi^2 for a charge-one scalar
    gphi_of = lambda r: np.sqrt(abs(WX(r)))

    def _d(f, r, eps=1e-6):
        step = eps * r
        return (f(r + step) - f(r - step)) / (2.0 * step)

    def _dd(f, r, eps=1e-5):
        step = eps * r
        return (f(r + step) - 2.0 * f(r) + f(r - step)) / step**2

    dWX = lambda r: _d(WX, r)

    rows_worst = [0.0, 0.0, 0.0]
    radii = np.linspace(0.08 * rmax, 0.85 * rmax, 60)
    for r in radii:
        w, dw = (float(v) for v in sol.sol(r))
        # second and third derivatives from the FP equation itself, so that the
        # mode is built on shell rather than differentiated numerically -- a_L
        # is a cancellation between omega' and l*omega/r that is exact in the
        # free limit, and finite differencing destroys it.
        d2w = -2.0 * dw / r + L2 * w / r**2 + WX(r) * w
        d3w = (-2.0 * d2w / r + 2.0 * dw / r**2 + L2 * (dw / r**2 - 2.0 * w / r**3)
               + dWX(r) * w + WX(r) * dw)

        gphi = gphi_of(r)                     # = g*phi for a charge-one scalar
        dgphi = _d(gphi_of, r)
        d2gphi = _dd(gphi_of, r)

        aS = cm * (dw + (l + 1.0) * w / r)
        daS = cm * (d2w + (l + 1.0) * dw / r - (l + 1.0) * w / r**2)
        d2aS = cm * (d3w + (l + 1.0) * d2w / r - 2.0 * (l + 1.0) * dw / r**2
                     + 2.0 * (l + 1.0) * w / r**3)
        aL = -cp * (dw - l * w / r)
        daL = -cp * (d2w - l * dw / r + l * w / r**2)
        d2aL = -cp * (d3w - l * d2w / r + 2.0 * l * dw / r**2 - 2.0 * l * w / r**3)
        aG = gphi * w
        daG = dgphi * w + gphi * dw
        d2aG = d2gphi * w + 2.0 * dgphi * dw + gphi * d2w

        k = mixing_factor
        terms = (
            [-(d2aS + 2.0 * daS / r - (l - 1.0) * l * aS / r**2),
             WX(r) * aS, k * cm * mX_prime(r) * aG],
            [-(d2aL + 2.0 * daL / r - (l + 1.0) * (l + 2.0) * aL / r**2),
             WX(r) * aL, -k * cp * mX_prime(r) * aG],
            [-(d2aG + 2.0 * daG / r - L2 * aG / r**2),
             (dWG(r) + mG_fv**2 + WX(r)) * aG,
             k * cm * mX_prime(r) * aS, -k * cp * mX_prime(r) * aL],
        )
        for i, t in enumerate(terms):
            scale = max(abs(x) for x in t)
            if scale > 0.0:
                rows_worst[i] = max(rows_worst[i], abs(sum(t)) / scale)

    return {
        "residual": float(max(rows_worst)),
        "rows": {"S": rows_worst[0], "L": rows_worst[1], "G": rows_worst[2]},
        "sampled": int(len(radii)),
    }


def constant_background_gauge_check(g_sq: float, phi: float,
                                    dV_over_phi: float,
                                    xi_values=(0.0, 0.25, 1.0, 3.0, 10.0)) -> dict:
    r"""Does the channel bookkeeping actually give a gauge-independent answer?

    Every check above tests one operator. This tests the *list*: the weights
    ``+1`` per boson and ``-2`` per Faddeev-Popov pair, and the claim that the
    barrier in the leading-order potential is built from exactly two
    polarisations. It does so where the answer is known in closed form -- a
    constant background -- so nothing has to be solved.

    In three dimensions each bosonic degree of freedom of mass :math:`m`
    contributes :math:`-m^3/12\pi` to the one-loop effective potential. In
    :math:`R_\xi` gauge the gauge-sector channels are two transverse modes at
    :math:`m_X^2 = g^2\phi^2`, one longitudinal mode at :math:`\xi m_X^2`, the
    Goldstone at :math:`V'/\phi + \xi m_X^2`, and the ghosts at
    :math:`\xi m_X^2` with weight :math:`-2`. Their sum is

    .. math::
        \frac{-1}{12\pi}\left[2m_X^3 - \xi^{3/2}m_X^3
        + \left(V'/\phi + \xi m_X^2\right)^{3/2}\right] ,

    which is independent of :math:`\xi` **only** where :math:`V'(\phi) = 0`,
    and there equals :math:`-2m_X^3/12\pi` -- the two transverse polarisations
    that ``N_TRANSVERSE_ABELIAN`` puts into the leading-order potential.

    Both halves matter. The cancellation at a stationary point is what says the
    weights are right, and it is the mechanism by which the very large
    :math:`\xi = 1` Goldstone contribution is paid back by the ghosts and the
    extra polarisation. The failure away from a stationary point is the Nielsen
    identity, and it is why the bounce -- which is nowhere at a stationary point
    of the potential -- has a gauge-dependent action, with the dependence
    cancelling only against the rest of the next-to-leading-order layer.

    Returns ``at_stationary_point`` and ``off_stationary_point``, each a dict
    from ``xi`` to the gauge-sector contribution, plus ``transverse_only``,
    ``gauge_independent`` and ``spread_off_shell``.
    """
    m_X_sq = g_sq * phi**2
    m_X = np.sqrt(m_X_sq)

    def sector(xi: float, dVop: float) -> float:
        m_long_sq = xi * m_X_sq
        m_gold_sq = dVop + xi * m_X_sq
        if m_gold_sq < 0.0:
            return float("nan")
        contributions = (
            2.0 * m_X_sq**1.5              # two transverse
            + 1.0 * m_long_sq**1.5         # one longitudinal
            + 1.0 * m_gold_sq**1.5         # the Goldstone
            - 2.0 * m_long_sq**1.5         # the ghosts, weight -2
        )
        return float(-contributions / (12.0 * np.pi))

    on = {float(x): sector(x, 0.0) for x in xi_values}
    off = {float(x): sector(x, dV_over_phi) for x in xi_values}
    transverse_only = float(-2.0 * m_X**3 / (12.0 * np.pi))
    spread_on = max(on.values()) - min(on.values())
    return {
        "at_stationary_point": on,
        "off_stationary_point": off,
        "transverse_only": transverse_only,
        "gauge_independent": bool(spread_on <= 1e-12 * abs(transverse_only)),
        "spread_on_shell": float(spread_on),
        "spread_off_shell": float(max(off.values()) - min(off.values())),
    }


def _general_xi_rhs(l, WX, mX_prime, WG, xi):
    r"""The coupled electric--Goldstone system at *general* gauge parameter.

    At :math:`\xi\neq1` the gauge kinetic term carries
    :math:`(1-1/\xi)\partial_i\partial_j`, which is
    :math:`\partial_i(\nabla\!\cdot\!a)` and therefore contains second
    derivatives of *both* electric amplitudes. The system stops being
    :math:`-u'' + Wu` and becomes :math:`Au'' + Bu' + Cu = 0` with

    .. math:: A = -\mathbb{1} + \kappa\,vv^{T}, \quad
              \kappa = 1 - 1/\xi, \quad v = (c_-,\,-c_+,\,0) ,

    whose eigenvalues are :math:`-1` on the transverse direction and
    :math:`-1/\xi` on the longitudinal one -- the expected structure, and
    invertible for every finite positive :math:`\xi`. That is the whole reason
    the :math:`\xi=1` solver cannot simply be given a parameter.

    The expansion coded below satisfies the gauge-orbit identity at general
    :math:`\xi`, which fixes it completely, and reduces to :func:`log_gy_mixed`
    at :math:`\xi=1` (tests/test_fluctuations.py).
    ``WG`` is the *full* Goldstone potential term :math:`V'/\phi`; the
    :math:`\xi g^2\phi^2` piece is added here so that the gauge parameter
    appears in exactly one place.
    """
    cm = np.sqrt(l / (2.0 * l + 1.0))
    cp = np.sqrt((l + 1.0) / (2.0 * l + 1.0))
    kappa = 1.0 - 1.0 / xi

    A = np.array([
        [-1.0 + kappa * cm * cm, -kappa * cm * cp, 0.0],
        [-kappa * cm * cp, -1.0 + kappa * cp * cp, 0.0],
        [0.0, 0.0, -1.0],
    ])
    A_inv = np.linalg.inv(A)

    def rhs(r, u, du):
        """Return u'' for state (u, du); u and du are (3, k) arrays."""
        w, p = WX(r), mX_prime(r)
        # first-derivative coefficients
        B = np.array([
            [(-2.0 + kappa * cm * cm * 2.0) / r,
             (-kappa * cm * cp * (2.0 * l + 3.0)) / r, 0.0],
            [(-kappa * cm * cp * (1.0 - 2.0 * l)) / r,
             (-2.0 + kappa * cp * cp * 2.0) / r, 0.0],
            [0.0, 0.0, -2.0 / r],
        ])
        # zeroth-order: centrifugal, the kappa pieces, the masses, the mixing
        C = np.array([
            [(l - 1.0) * l / r**2 + kappa * cm * cm * l * (1.0 - l) / r**2 + w,
             -kappa * cm * cp * l * (l + 2.0) / r**2,
             2.0 * cm * p],
            [kappa * cm * cp * (1.0 - l) * (1.0 + l) / r**2,
             (l + 1.0) * (l + 2.0) / r**2
             - kappa * cp * cp * (l + 2.0) * (l + 1.0) / r**2 + w,
             -2.0 * cp * p],
            [2.0 * cm * p, -2.0 * cp * p,
             l * (l + 1.0) / r**2 + WG(r) + xi * w],
        ])
        return -A_inv @ (B @ du + C @ u)

    return rhs


def log_gy_mixed_general_xi(WX, mX_prime, WG, l, rmax, xi, rmin=1e-4,
                            rtol=1e-9, atol=1e-12, n_chunks=40):
    r"""``ln`` of the coupled determinant ratio at general :math:`\xi`.

    Unlike :func:`log_gy_mixed` this does not work in ratios to known free
    solutions, because at general :math:`\xi` writing the operator in those
    variables means transforming a system whose principal part is not the
    identity, which is more algebra. Instead the full and free systems are integrated *side by side*
    from the same initial conditions, each with its own chunked rescaling, and
    the ratio is taken at the end. The free system is integrated rather than
    assumed, which also removes the question of whether the free solutions are
    still :math:`r^{\ell\mp1}` away from :math:`\xi=1`.

    ``WG`` is :math:`V'/\phi`, the potential part of the Goldstone mass, as a
    function of radius; its false-vacuum limit is used for the free system.

    Returns ``ln|det T|``.
    """
    if xi <= 0.0:
        raise ValueError(f"the gauge parameter must be positive, got {xi}")

    WG_inf = float(WG(rmax))
    full = _general_xi_rhs(l, WX, mX_prime, WG, xi)
    free = _general_xi_rhs(l, lambda r: 0.0, lambda r: 0.0,
                           lambda r: WG_inf, xi)

    def integrate(rhs):
        # three solutions, started as the regular small-r behaviours
        Y = np.zeros((6, 3))
        for k, power in enumerate((l - 1.0, l + 1.0, l)):
            Y[k, k] = rmin ** power if power >= 0 else 0.0
            Y[3 + k, k] = (power * rmin ** (power - 1.0)) if power >= 1 else 0.0
        if not np.any(Y):                      # l = 0: S channel absent
            raise FluctuationSolverError(f"degenerate initial data at l={l}")
        # normalise the starting matrix so both systems begin identically scaled
        norms = np.linalg.norm(Y, axis=0)
        norms[norms == 0.0] = 1.0
        Y = Y / norms
        log_scale = float(np.sum(np.log(norms)))

        def f(r, y):
            Z = y.reshape(6, 3)
            u, du = Z[:3], Z[3:]
            return np.vstack([du, rhs(r, u, du)]).ravel()

        for a, b in zip(np.linspace(rmin, rmax, n_chunks + 1)[:-1],
                        np.linspace(rmin, rmax, n_chunks + 1)[1:]):
            s = solve_ivp(f, (a, b), Y.ravel(), rtol=rtol, atol=atol,
                          method="Radau")
            if not s.success or not np.all(np.isfinite(s.y[:, -1])):
                raise FluctuationSolverError(
                    f"general-xi solve failed at l={l}, xi={xi} on "
                    f"[{a:.4g},{b:.4g}]: "
                    f"{s.message if not s.success else 'non-finite result'}")
            Y = s.y[:, -1].reshape(6, 3)
            col = np.linalg.norm(Y, axis=0)
            if not np.all(col > 0.0):
                raise FluctuationSolverError(f"a column vanished at l={l}")
            Y = Y / col
            log_scale += float(np.sum(np.log(col)))
        det = float(np.linalg.det(Y[:3]))
        if det == 0.0:
            raise FluctuationSolverError(f"rescaled determinant vanished at l={l}")
        return np.log(abs(det)) + log_scale

    return integrate(full) - integrate(free)
