"""Dimensionally reduced matching relations for the conformal U(1) with fermions.

GENERATED FROM DRalgo -- do not hand-edit the expressions.

Provenance
----------
Generated with   : DRalgo 1.5.0 and GroupMath 1.1.3.
Model           : U(1) gauge boson; complex scalar Phi of charge +1; two
                  left-handed Weyl fermions of charge -1/2 and +1/2 with
                  Majorana Yukawas (y/2) Phi chibar^c chi. Anomaly free.
                  Conformal limit is msq -> 0.

Conventions
-----------
* Potential      V = msq (Phi^+ Phi) + lam (Phi^+ Phi)^2, Phi = (phi + ...)/sqrt(2)
                 => V(phi) = msq phi^2/2 + lam phi^4/4: the same normalisation
                 as the TL model file's `self.l`, but not the same scheme --
                 `self.l` is a Coleman-Weinberg quartic, converted to MS-bar
                 by `msbar_quartic_from_cw`.
* Beta functions are d X / d ln(mubar). Confirmed: beta(g^2) = g^4/(12 pi^2)
  reproduces 2 b g^4/(16 pi^2) with b = (2/3) sum_Weyl Q^2 + (1/3) sum_scalar Q^2
  = 2/3 for this field content.
* Debye mass LO = (5/12) g^2 T^2 and the LO scalar thermal mass coefficient
  lam/3 + g^2/4 + y^2/12 both reproduce the hard-coded values in
  models/TL_conformal_u1_fermions.py.

Cross-checks
------------
* The logarithms Lb, Lf. Their slopes d/d ln mubar = 2 follow from requiring g3^2 and lambda3 to be independent of
  mubar order by order (`test_matching_logs.py`). Their relative offset Lf - Lb = 4 ln 2, from the bosonic and
  fermionic Matsubara scales 4 pi T and pi T, is fixed by the comparison with an independent implementation of these
  relations: lambda3, g3^2 and the Debye mass agree to rounding at nonzero y, and m3^2 to the order of the matching
  (`test_matching_against_an_independent_implementation.py`). A common shift of Lb and Lf is a rescaling of mubar.
* The input scheme is converted at the level of the one-loop potential (see `msbar_quartic_from_cw`); what that
  leaves out is listed in `INPUT_SCHEME_LIMITATIONS`.
"""
from __future__ import annotations

import math

ZETA3 = 1.2020569031595943
EULER_GAMMA = 0.5772156649015329


# ----------------------------------------------------------------------
# logarithms entering the NLO matching  (see 'Cross-checks')
# ----------------------------------------------------------------------
def Lb(mubar: float, T: float) -> float:
    """Bosonic matching logarithm, 2 ln(mubar e^{gamma_E}/(4 pi T)).

    Its SLOPE d/d ln(mubar) = 2 is verified in tests/test_matching_logs.py; its
    additive constant is a choice of mubar, see the module docstring.
    """
    return 2.0 * math.log(mubar * math.exp(EULER_GAMMA) / (4.0 * math.pi * T))


def Lf(mubar: float, T: float) -> float:
    """Fermionic matching logarithm, Lb + 4 ln 2.

    Same slope as Lb; the offset is the ratio of the fermionic to the bosonic
    Matsubara scale, pi T against 4 pi T. See the module docstring.
    """
    return Lb(mubar, T) + 4.0 * math.log(2.0)


# ----------------------------------------------------------------------
# 4D one-loop beta functions,  d X / d ln(mubar)
#   DRalgo: {g^2 -> g^4/(12 Pi^2),
#            lam -> (3 g^4 - 6 g^2 lam + 10 lam^2 - |y|^4 + 2 |y|^2 lam)/(8 Pi^2),
#            y   -> y(-3 g^2 + 4 |y|^2)/(32 Pi^2),
#            msq -> msq(-3 g^2 + 4 lam + |y|^2)/(8 Pi^2)}
# ----------------------------------------------------------------------
def beta_g2(g: float, y: float, lam: float) -> float:
    return g**4 / (12.0 * math.pi**2)


def beta_lambda(g: float, y: float, lam: float) -> float:
    return (3.0 * g**4 - 6.0 * g**2 * lam + 10.0 * lam**2
            - y**4 + 2.0 * y**2 * lam) / (8.0 * math.pi**2)


def beta_y(g: float, y: float, lam: float) -> float:
    return y * (-3.0 * g**2 + 4.0 * y**2) / (32.0 * math.pi**2)


def beta_msq(g: float, y: float, lam: float, msq: float) -> float:
    return msq * (-3.0 * g**2 + 4.0 * lam + y**2) / (8.0 * math.pi**2)


# ----------------------------------------------------------------------
# hard -> soft matching (the 3D EFT with A_0 still present)
# ----------------------------------------------------------------------
def g3_sq_LO(g: float, T: float) -> float:
    """3D gauge coupling squared, leading order. Mass dimension 1."""
    return g**2 * T


def gauge_coupling_sq(g: float, T: float, mubar: float, matching_order: str) -> float:
    """g3^2 at the matching order of the masses: `g3_sq_NLO` for "NLO", `g3_sq_LO` for "LO"."""
    if matching_order == "NLO":
        return g3_sq_NLO(g, T, mubar)
    if matching_order == "LO":
        return g3_sq_LO(g, T)
    raise ValueError(f"matching_order must be 'LO' or 'NLO', not {matching_order!r}")


def lambda3_LO(lam: float, T: float) -> float:
    """3D quartic at strict tree level, ``lambda * T``. Mass dimension 1.

    WARNING -- this is almost certainly NOT what you want for a classically
    conformal model. The one-loop ``g^4 L_b`` term dropped here is precisely
    the Coleman-Weinberg contribution, which for a conformal model is the whole
    radiative structure rather than a correction: see the note in
    arXiv:2312.12413 below eq. (4.28), "the Coleman-Weinberg term is captured
    by the L_b term in lambda_3". Their leading-order EFT potential uses the
    one-loop matched quartic, not this one.

    Concretely, at the central scale mu = 4 pi exp(-gamma_E) T the one-loop
    term contributes ``+g^4/(8 pi^2)``, which is +0.0083 for g = 0.9 -- larger
    than the running quartic itself over much of the supercooled range, and of
    the opposite sign. Dropping it makes the 3D potential unbounded where it
    should not be.

    Kept because it is the honest strict-tree matching and is what the
    ``x = lambda3/g3^2`` bookkeeping is defined with; use
    :func:`lambda3_NLO` for anything physical.
    """
    return lam * T


def scalar_thermal_mass_sq_LO(g: float, y: float, lam: float, T: float,
                              msq: float = 0.0) -> float:
    """3D scalar mass squared at LO.

    DRalgo: msq3d = (12 msq + T^2 (3 g^2 + 4 lam) + T^2 |y|^2)/12
                  = msq + T^2 (lam/3 + g^2/4 + y^2/12)

    The T^2 coefficient is exactly `scalar_prefactor` of
    models/TL_conformal_u1_fermions.py -- that agreement is the check that the 4D
    model file's thermal mass is right.
    """
    return msq + T**2 * (lam / 3.0 + g**2 / 4.0 + y**2 / 12.0)


def debye_mass_sq_LO(g: float, T: float) -> float:
    """Temporal-gauge (Debye) mass squared at LO: (5/12) g^2 T^2.

    The 5/12 is (1/3) from the charge-one complex scalar plus (1/6) times
    sum_Weyl Q^2 = 1/2 from the two Weyl fermions.
    """
    return (5.0 / 12.0) * g**2 * T**2


#: The Glaisher-Kinkelin constant appears in DRalgo's NLO thermal masses as
#: Log[Glaisher]; A = 1.2824271291...
LOG_GLAISHER = 0.2487544770337842


def scalar_thermal_mass_sq_NLO(g: float, y: float, lam: float, T: float,
                               mubar: float, mu3: float,
                               msq: float = 0.0) -> float:
    """3D scalar mass squared including the NLO matching. Additive on the LO one.

    DRalgo's PrintScalarMass["NLO"] is a correction, not a replacement: it
    carries no bare ``12 msq`` term and everything in it is down by 1/(16 pi^2).
    This returns LO + that correction.

    It is the piece that makes the 4D matching scale cancel. The LO mass has no
    ``mubar`` in it at all, so every bit of ``mubar`` dependence in a
    leading-order prediction survives uncancelled; the ``Lb``, ``Lf`` and
    ``Log[mu3/mubar]`` terms here are what cancel it against the running of the
    4D couplings. Using the LO mass with the NLO quartic mixes orders and
    inflates the band.

    ``mu3`` is the 3D renormalisation scale, the same one the two-loop
    potential is written at.
    """
    lb, lf = Lb(mubar, T), Lf(mubar, T)
    g3d_sq = g3_sq_LO(g, T)
    lam3d = lambda3_NLO(g, y, lam, T, mubar)
    lam_vl = lambda_VL(g, y, lam, T, mubar)
    y2 = y * y

    correction = (
        216.0 * g**2 * lb * msq
        - 14.0 * g**4 * T**2
        - 216.0 * EULER_GAMMA * g**4 * T**2
        + 129.0 * g**4 * lb * T**2
        + 3.0 * g**4 * lf * T**2
        - 288.0 * lb * msq * lam
        + 48.0 * g**2 * T**2 * lam
        + 288.0 * EULER_GAMMA * g**2 * T**2 * lam
        - 144.0 * g**2 * lb * T**2 * lam
        - 288.0 * EULER_GAMMA * T**2 * lam**2
        + 48.0 * lb * T**2 * lam**2
        - 3.0 * y2 * (24.0 * lf * msq
                      - g**2 * (1.0 + 8.0 * lb) * T**2
                      + 12.0 * lb * T**2 * lam
                      + lf * T**2 * (5.0 * g**2 + 4.0 * lam))
        + 6.0 * (lb + lf) * T**2 * y2**2
        + 2592.0 * g**4 * T**2 * LOG_GLAISHER
        - 3456.0 * g**2 * T**2 * lam * LOG_GLAISHER
        + 3456.0 * T**2 * lam**2 * LOG_GLAISHER
        + 36.0 * math.log(mu3 / mubar) * (8.0 * g3d_sq**2
                                          - 16.0 * g3d_sq * lam3d
                                          + 16.0 * lam3d**2
                                          + lam_vl**2)
    ) / (1152.0 * math.pi**2)
    return scalar_thermal_mass_sq_LO(g, y, lam, T, msq) + correction


def debye_mass_sq_NLO(g: float, y: float, lam: float, T: float, mubar: float,
                      msq: float = 0.0) -> float:
    """Temporal-gauge mass squared including the NLO matching. Additive on LO.

    DRalgo writes the logarithms out explicitly here rather than in terms of
    Lb, and they collect: 20 g^2 T^2 (Log[4 pi^2 T] - Log[mubar] - Log[mubar/T])
    is 20 g^2 T^2 Log[4 pi^2 T^2/mubar^2] = 40 g^2 T^2 Log[2 pi T/mubar], which
    together with the -40 EulerGamma term is
    -40 g^2 T^2 Log[mubar e^{gamma_E}/(2 pi T)].
    """
    y2 = y * y
    correction = g**2 * (
        288.0 * msq
        + 53.0 * g**2 * T**2
        + 96.0 * T**2 * lam
        + 6.0 * T**2 * y2
        - 40.0 * g**2 * T**2 * math.log(
            mubar * math.exp(EULER_GAMMA) / (2.0 * math.pi * T))
    ) / (1152.0 * math.pi**2)
    return debye_mass_sq_LO(g, T) + correction


def g3_sq_NLO(g: float, T: float, mubar: float) -> float:
    """3D gauge coupling squared including the NLO term.

    That this is independent of mubar to the order worked -- the explicit
    Lb + Lf running against the running of g^2 itself -- is what fixes the
    slope of Lb; see tests/test_matching_logs.py.
    """
    return g**2 * T * (1.0 - g**2 * (Lb(mubar, T) + Lf(mubar, T)) / (48.0 * math.pi**2))


def lambda3_NLO(g: float, y: float, lam: float, T: float, mubar: float) -> float:
    """3D quartic including the one-loop matching. Uses Lb/Lf -- see docstring.

    This is the quartic to use for a conformal model: the ``g^4`` term it adds
    over :func:`lambda3_LO` is the Coleman-Weinberg contribution, and for a
    classically scale-invariant potential that term carries the physics rather
    than correcting it. Despite the name, this is what arXiv:2312.12413 uses in
    its *leading-order* EFT potential, eq. (4.28).
    """
    lb, lf = Lb(mubar, T), Lf(mubar, T)
    return (T * (g**4 * (2.0 - 3.0 * lb)
                 + 6.0 * g**2 * lb * lam
                 + 2.0 * lam * (8.0 * math.pi**2 - 5.0 * lb * lam)
                 + lf * y**4
                 - 2.0 * lf * y**2 * lam)) / (16.0 * math.pi**2)


# ----------------------------------------------------------------------
# temporal-scalar couplings: the A_0 sector of the soft theory
#   DRalgo run 2, "### TemporalScalarCouplings":
#     lamVLL[1] -> 7 g^4 T/(8 Pi^2)
#     lamVL[1]  -> -(g^2 T (g^2(-5 + Lb + Lf) - 24(2 Pi^2 + lam)
#                           + 3 |y|^2))/(24 Pi^2)
# ----------------------------------------------------------------------
def lambda_VL(g: float, y: float, lam: float, T: float, mubar: float) -> float:
    """Coupling of the temporal gauge mode to the scalar, ``lambda_VL[1]``.

    This is the coefficient of the A_0^2 |Phi|^2 operator in the soft theory.
    Mass dimension 1.

    The leading term is ``2 g^2 T``, and that normalisation is checked
    independently in :func:`_check_lambda_VL_normalisation` below: it is the
    value for which DRalgo's own ultrasoft mass shift reproduces the
    high-temperature expansion of the A_0 cubic ``-(T/12 pi)(m_D^2 +
    g^2 phi^2)^{3/2}``. Everything beyond the leading term is one-loop and
    uses Lb, Lf.
    """
    lb, lf = Lb(mubar, T), Lf(mubar, T)
    return -(g**2 * T * (g**2 * (-5.0 + lb + lf)
                         - 24.0 * (2.0 * math.pi**2 + lam)
                         + 3.0 * y**2)) / (24.0 * math.pi**2)


def lambda_VLL(g: float, T: float) -> float:
    """Quartic self-coupling of the temporal gauge mode, ``lambda_VLL[1]``.

    ``7 g^4 T/(8 pi^2)``. Mass dimension 1. Enters only the NLO ultrasoft
    scalar mass.
    """
    return 7.0 * g**4 * T / (8.0 * math.pi**2)


# ----------------------------------------------------------------------
# soft -> ultrasoft:  A_0 integrated out
#
# These take SOFT-theory quantities, exactly as DRalgo writes them, so that
# nothing has to be re-derived when composing them. Their inputs are the
# outputs of the hard->soft functions above.
# ----------------------------------------------------------------------
def g3_sq_US(g3_sq: float) -> float:
    """3D gauge coupling squared in the ultrasoft theory.

    DRalgo: ``g3dUS^2 -> g3d^2``. Unchanged at this order -- an abelian A_0
    carries no charge, so integrating it out cannot renormalise the coupling.
    """
    return g3_sq


def lambda3_US(lambda3: float, lambda_vl: float, debye_mass_sq: float) -> float:
    """3D quartic after integrating out A_0.

    DRalgo: ``lam3dUS -> lam3d - lamVL^2/(32 pi sqrt(muSqU1))``.

    Cross-check (see :func:`_check_lambda_VL_normalisation`): with
    ``lambda_vl = 2 g^2 T`` this is ``-g^4 T^2/(8 pi m_D)``, which is what the
    phi^4 term of ``-(1/12 pi)(m_D^2 + g3^2 phi3^2)^{3/2}`` gives.
    """
    if debye_mass_sq <= 0.0:
        raise ValueError("the Debye mass squared must be positive to "
                         "integrate out A_0")
    return lambda3 - lambda_vl**2 / (32.0 * math.pi * math.sqrt(debye_mass_sq))


def scalar_mass_sq_US(m3_sq: float, lambda_vl: float, debye_mass_sq: float, *,
                      order: str = "LO",
                      lambda_vll: float | None = None,
                      mu3: float | None = None) -> float:
    """3D scalar mass squared after integrating out A_0.

    DRalgo, LO:
        ``msq3dUS -> msq3d - sqrt(muSqU1) lamVL/(8 pi)``
    DRalgo, NLO (an ADDITIVE correction on top of the LO relation, note it
    contains no msq3d):
        ``msq3dUS -> lamVL(-2 lamVL - 4 Log[mu3/(2 sqrt(muSqU1))] lamVL
                           + lamVLL)/(128 pi^2)``

    ``order="NLO"`` returns LO + that correction and then requires
    ``lambda_vll`` and ``mu3`` (the SOFT 3D renormalisation scale).
    """
    if debye_mass_sq <= 0.0:
        raise ValueError("the Debye mass squared must be positive to "
                         "integrate out A_0")
    m_debye = math.sqrt(debye_mass_sq)
    value = m3_sq - m_debye * lambda_vl / (8.0 * math.pi)
    if order == "LO":
        return value
    if order != "NLO":
        raise ValueError(f"order must be 'LO' or 'NLO', not {order!r}")
    if lambda_vll is None or mu3 is None:
        raise ValueError("order='NLO' needs both lambda_vll and mu3")
    if mu3 <= 0.0:
        raise ValueError("mu3 must be positive")
    value += lambda_vl * (-2.0 * lambda_vl
                          - 4.0 * math.log(mu3 / (2.0 * m_debye)) * lambda_vl
                          + lambda_vll) / (128.0 * math.pi**2)
    return value


def _check_lambda_VL_normalisation(g: float, T: float) -> tuple:
    """Re-derivation of the two ultrasoft relations, not a transcription.

    Integrating A_0 out of the soft theory is, at this order, nothing but the
    high-temperature expansion of its one-loop cubic. In three dimensions that
    cubic is ``-(1/12 pi)(m_D^2 + g3^2 phi3^2)^{3/2}``, and expanding in
    ``g3^2 phi3^2/m_D^2`` gives

        phi3^2 coefficient : -g3^2 m_D/(24 pi)   =>  delta m3^2 = -g3^2 m_D/(4 pi)
        phi3^4 coefficient : -g3^4/(96 pi m_D)   =>  delta lam3 = -g3^4/(8 pi m_D)

    using ``V = m3^2 phi3^2/2 + lam3 phi3^4/4``. Setting ``g3^2 = g^2 T``,
    those are exactly DRalgo's LO relations provided ``lambda_VL = 2 g^2 T``.

    Returns ``(from_dralgo_mass, from_expansion_mass, from_dralgo_quartic,
    from_expansion_quartic)`` as absolute shifts, for the test suite to compare.
    """
    m_debye = math.sqrt(debye_mass_sq_LO(g, T))
    g3sq = g3_sq_LO(g, T)
    lam_vl_leading = 2.0 * g**2 * T
    return (
        -m_debye * lam_vl_leading / (8.0 * math.pi),
        -g3sq * m_debye / (4.0 * math.pi),
        -lam_vl_leading**2 / (32.0 * math.pi * m_debye),
        -g3sq**2 / (8.0 * math.pi * m_debye),
    )


# ----------------------------------------------------------------------
# two-loop effective potential of the ULTRASOFT theory
#
# DRalgo run 2, "### EffectivePotentialUS NNLO", transcribed verbatim. This is
# the ``exp(-int V3^NLO)`` factor of arXiv:2503.13597 eq. (4.20) -- but for the
# ultrasoft theory, i.e. with A_0 already integrated out. Our leading-order
# potential follows their eq. (4.2) and KEEPS A_0, so this is the two-loop term
# of the OTHER branch and the two must not be mixed. See the module docstring.
#
# Arguments are ultrasoft: m3_sq = msq3dUS, lam3 = lam3dUS, g3_sq = g3dUS^2,
# mu3us the ultrasoft 3D renormalisation scale. Mass dimensions in 3D are
# [phi3] = 1/2, [lam3] = [g3^2] = 1, [m3^2] = 2, and the potential has
# dimension 3.
# ----------------------------------------------------------------------
#: Working precision for the two-loop potential. Three of its groups carry an
#: explicit 1/(4 phi3^2) whose numerator vanishes as phi3^2 while its individual
#: terms stay finite, so the small-field tail of a bounce loses roughly
#: log10(m3^2/(g3^2 phi3^2)) significant digits in double precision. mpmath at
#: this many digits makes that irrelevant at negligible cost -- the integrand is
#: evaluated on a few hundred radial points.
V3_US_TWO_LOOP_DPS = 50


def V3_US_two_loop(phi3, m3_sq: float, lam3: float, g3_sq: float,
                   mu3us: float, *, subtract_origin: bool = True,
                   dps: int = V3_US_TWO_LOOP_DPS, tachyonic: str = "raise"):
    """Two-loop ultrasoft potential, evaluated with mpmath and returned as float.

    ``phi3`` may be a scalar or any 1D sequence; the return type follows.

    With ``subtract_origin`` (the default) the value at ``phi3 = 0`` is removed,
    which is what the bounce integral needs: the false vacuum of the conformal
    model sits at the origin, and only the difference is physical. The
    unsubtracted potential is field-independent-divergent in no sense, but its
    constant part is large and would swamp the difference.

    ``tachyonic`` says what happens where a scalar mass squared is negative:
    "raise" (the default) raises ValueError rather than returning a complex
    number silently; "real_part" continues every mass to ``+i|m|`` (the
    principal square root, the same branch for all of them) and returns the
    real part. The second is a convention, not a derivation -- it is what a
    potential exported to a numerical code and evaluated past the point where
    its scalar masses turn negative gives, and it is needed only to reproduce
    calculations that did that (arXiv:2511.02910).
    """
    if tachyonic not in ("raise", "real_part"):
        raise ValueError(f"tachyonic must be 'raise' or 'real_part', not {tachyonic!r}")
    import mpmath

    scalar_input = not hasattr(phi3, "__len__")
    values = [phi3] if scalar_input else list(phi3)

    with mpmath.workdps(dps):
        msq = mpmath.mpf(repr(float(m3_sq)))
        lam = mpmath.mpf(repr(float(lam3)))
        g2 = mpmath.mpf(repr(float(g3_sq)))
        mu = mpmath.mpf(repr(float(mu3us)))
        complex_ok = tachyonic == "real_part"
        out = [_V3_US_two_loop_point(mpmath.mpf(repr(float(p))),
                                     msq, lam, g2, mu, mpmath, complex_ok)
               for p in values]
        if subtract_origin:
            zero = _V3_US_two_loop_point(mpmath.mpf(0), msq, lam, g2, mu, mpmath,
                                         complex_ok)
            out = [v - zero for v in out]
        out = [float(mpmath.re(v)) for v in out]

    return out[0] if scalar_input else out


def _V3_US_two_loop_point(phi, msq, lam, g2, mu, mp, complex_ok=False):
    """One field value of :func:`V3_US_two_loop`. ``mp`` is the mpmath module.

    Transcribed term by term from DRalgo's InputForm. The three field-dependent
    masses are

        mA = sqrt(g3^2 phi3^2)      transverse gauge
        m1 = sqrt(m3^2 + lam3 phi3^2)   Goldstone
        m2 = sqrt(m3^2 + 3 lam3 phi3^2) radial

    and the limit phi3 -> 0 is taken analytically: there mA = 0, m1 = m2, the
    three 1/(4 phi3^2) groups vanish identically, and only the five algebraic
    terms survive.
    """
    pi = mp.pi
    p2 = phi * phi
    m1sq = msq + lam * p2
    m2sq = msq + 3 * lam * p2
    if complex_ok and (m1sq <= 0 or m2sq <= 0):
        # principal square root: +i|m| for every negative mass squared
        m1sq, m2sq = mp.mpc(m1sq), mp.mpc(m2sq)
    elif m1sq <= 0 or m2sq <= 0 or g2 * p2 < 0:
        raise ValueError(
            "the two-loop ultrasoft potential needs both scalar masses squared "
            "strictly positive -- it is infrared divergent otherwise, every "
            "logarithm carrying a mass in its argument; got "
            f"mA^2={float(g2 * p2):.6g}, m1^2={float(m1sq):.6g}, "
            f"m2^2={float(m2sq):.6g} at phi3={float(phi):.6g}"
        )
    mA = mp.sqrt(g2 * p2)
    m1 = mp.sqrt(m1sq)
    m2 = mp.sqrt(m2sq)

    def L(x):
        """The (1/2 + Log[mu3US/x]) that every logarithm of the output wears."""
        return mp.mpf(1) / 2 + mp.log(mu / x)

    total = (g2 * mA * m1 / (16 * pi**2)
             + 3 * lam * m1sq / (64 * pi**2)
             + g2 * mA * m2 / (16 * pi**2)
             + lam * m1 * m2 / (32 * pi**2)
             + 3 * lam * m2sq / (64 * pi**2)
             - 3 * lam**2 * p2 * L(3 * m2) / (16 * pi**2)
             - lam**2 * p2 * L(2 * m1 + m2) / (16 * pi**2))

    if phi == 0:
        # The numerators of the three 1/(4 phi3^2) groups vanish as phi3^2, so
        # each group has a finite limit -- NOT zero. Group 1's limit is zero,
        # groups 2 and 3 give the same non-zero value. Derived with sympy and
        # checked against the numerical limit in tests/test_ultrasoft.py; its
        # scale derivative, g3^2 m3^2/(8 pi^2), is what makes the subtracted
        # potential satisfy the 3D renormalisation-group check.
        return total + (g2 * msq * (4 * mp.log(mu / (2 * mp.sqrt(msq))) + 3)
                        / (32 * pi**2))

    # group 1: the gauge/radial sector
    A = -msq + g2 * p2 - 3 * lam * p2
    g1 = (g2**2 * p2**2 / (8 * pi**2)
          - g2 * p2 * (-msq + 2 * g2 * p2 - 3 * lam * p2) / (16 * pi**2)
          + g2 * p2 * mA * m2 / (8 * pi**2)
          - m2sq**2 * L(m2) / (16 * pi**2)
          + A**2 * L(mA + m2) / (8 * pi**2)
          - (7 * g2**2 * p2**2 + A**2 - 2 * g2 * p2 * m2sq)
          * L(2 * mA + m2) / (16 * pi**2))

    # groups 2 and 3 differ only by which of m1, m2 sits outside the bracket
    B = (g2**2 * p2**2 + 4 * lam**2 * p2**2
         - 2 * g2 * p2 * (2 * msq + 4 * lam * p2))
    shared = (lam**2 * p2**2 * L(m1 + m2) / (4 * pi**2)
              - B * L(mA + m1 + m2) / (16 * pi**2))
    g2_ = (-mA * (g2 * p2 - 2 * lam * p2) * m1 / (16 * pi**2)
           + m2 * (g2 * p2 * m1 / (4 * pi)
                   - mA * (g2 * p2 + 2 * lam * p2) / (4 * pi)) / (4 * pi)
           + shared)
    g3_ = (-mA * (g2 * p2 + 2 * lam * p2) * m2 / (16 * pi**2)
           + m1 * (-mA * (g2 * p2 - 2 * lam * p2) / (4 * pi)
                   + g2 * p2 * m2 / (4 * pi)) / (4 * pi)
           + shared)

    return total + (g1 + g2_ + g3_) / (4 * p2)


# ----------------------------------------------------------------------
# SOFT-theory next-to-leading-order action  (A_0 kept)
#
# DRalgo run 5, via the HET (Higgs Effective Theory) module: PerformDRsoft,
# UseSoftTheory, PrepareHET[{3},{1}], CalculatePotentialHET. That integrates
# out the two modes that become heavy inside the bubble -- the spatial gauge
# mode and the temporal one -- and leaves the light scalar alone, since it is
# the field the potential depends on.
#
# This is the abelian counterpart of eq. (4.7) of arXiv:2503.13597, and it is
# the factor of their eq. (4.20) this project had been missing. Unlike
# CalculatePotentialUS's two-loop output it contains NO scalar masses, so it
# is finite and real everywhere on the bounce -- which matters, because at the
# physical temperature the bounce runs about three times past the field value
# where the scalar tree masses go negative.
#
# The field-dependent masses DRalgo prints for the soft theory are
#     Goldstone   m3^2 +   lam3 phi3^2
#     radial      m3^2 + 3 lam3 phi3^2
#     temporal    m_D^2 + lambda_VL phi3^2/2      <- confirms h_3 = lambda_VL
# and the last of those settles a question the leading-order potential had
# been answering by hand; see `TEMPORAL_MASS_USES_LAMBDA_VL`.
# ----------------------------------------------------------------------
#: Our LO potential builds the temporal gauge mass as m_D^2 + g3^2 phi3^2,
#: which is the tree-level relation. DRalgo's soft theory gives
#: m_D^2 + lambda_VL phi3^2/2, and lambda_VL = 2 g3^2 only at leading order --
#: it carries one-loop corrections in Lb, Lf. The two agree to the accuracy
#: the LO potential is claimed at, so this is not an error, but the matched
#: coupling is the consistent choice.
TEMPORAL_MASS_USES_LAMBDA_VL = (
    "abelian_higgs_3d._gauge_cubic builds the temporal mass as m_D^2 + g3^2 phi3^2 "
    "(tree level); DRalgo's soft theory gives m_D^2 + lambda_VL phi3^2/2. "
    "They differ at one loop. Use temporal_mass_sq_soft() to compare."
)


def temporal_mass_sq_soft(phi3, debye_mass_sq: float, lambda_vl: float):
    """Field-dependent temporal gauge mass squared, ``m_D^2 + lambda_VL phi3^2/2``.

    Printed directly by DRalgo as the (3,3) entry of the soft theory's
    field-dependent mass matrix, which is what identifies ``lambda_VL`` with
    the ``h_3`` of arXiv:2503.13597 eq. (4.6).
    """
    return debye_mass_sq + lambda_vl * phi3**2 / 2.0


def V3_soft_two_loop(phi3, g3_sq: float, debye_mass_sq: float,
                     lambda_vl: float, lambda_vll: float, mu3: float, *,
                     subtract_origin: bool = True):
    r"""Two-loop soft potential, the abelian counterpart of eq. (4.7).

    DRalgo returns it as four terms; the first cancels half of the second
    identically, and what survives is

    .. math::
        V_3^{\rm NLO} =
        - \frac{g_3^4 \phi_3^2}{8\pi^2}
          \Bigl[\tfrac12 + \ln\frac{\mu_3}{2 m_X}\Bigr]
        - \frac{\lambda_{VL}^2 \phi_3^2}{64\pi^2}
          \Bigl[\tfrac12 + \ln\frac{\mu_3}{2 m_{X_0}}\Bigr]
        + \frac{m_{X_0}^2 \lambda_{VLL}}{128\pi^2}

    with :math:`m_X = g_3|\phi_3|` and
    :math:`m_{X_0}^2 = m_D^2 + \lambda_{VL}\phi_3^2/2`. That the cancellation
    is exact is checked against DRalgo's four-term form in tests/test_soft_nlo.py.

    Every mass here is a gauge mass and manifestly non-negative, which is what
    makes this usable on a bounce where the scalar masses are not. The
    :math:`\phi_3^2\ln\phi_3` at small field tends to zero and is evaluated
    directly; no special handling is needed, and unlike
    :func:`V3_US_two_loop` there is no cancellation, so plain double precision
    is enough.

    ``phi3`` may be a scalar or a 1D numpy array; the return type follows.
    """
    import numpy as np

    phi = np.asarray(phi3, dtype=float)
    if debye_mass_sq <= 0.0:
        raise ValueError("the soft theory needs a positive Debye mass squared")
    if mu3 <= 0.0:
        raise ValueError("mu3 must be positive")

    m0_sq = temporal_mass_sq_soft(phi, debye_mass_sq, lambda_vl)
    if np.any(m0_sq <= 0.0):
        raise ValueError(
            "the temporal gauge mass squared m_D^2 + lambda_VL phi3^2/2 must "
            "stay positive; a negative lambda_VL large enough to overwhelm "
            "the Debye mass means A_0 has been driven tachyonic and the soft "
            "theory has broken down"
        )
    m0 = np.sqrt(m0_sq)
    mA = np.sqrt(g3_sq) * np.abs(phi)

    pi2 = np.pi**2
    # phi^2 ln(mu/(2 mA)) -> 0 as phi -> 0; evaluate the product, not the log
    with np.errstate(divide="ignore", invalid="ignore"):
        gauge_log = np.where(mA > 0.0, np.log(mu3 / (2.0 * np.where(mA > 0.0, mA, 1.0))), 0.0)
    gauge = -g3_sq**2 * phi**2 * (0.5 + gauge_log) / (8.0 * pi2)
    gauge = np.where(phi == 0.0, 0.0, gauge)

    temporal = -lambda_vl**2 * phi**2 * (0.5 + np.log(mu3 / (2.0 * m0))) / (64.0 * pi2)
    quartic = m0_sq * lambda_vll / (128.0 * pi2)

    total = gauge + temporal + quartic
    if subtract_origin:
        total = total - debye_mass_sq * lambda_vll / (128.0 * pi2)
    return float(total) if np.isscalar(phi3) or np.ndim(phi3) == 0 else total


def Z3_soft_NLO(phi3, g3_sq: float, debye_mass_sq: float, lambda_vl: float):
    r"""One-loop gauge-sector correction to the scalar kinetic term, cf. eq. (4.8).

    Not a two-loop quantity: it comes from integrating out the spatial and
    temporal gauge modes at one loop, and it is the derivative expansion of
    their fluctuation determinants. It therefore replaces those determinants
    in a derivative-expansion exponent and must not be added to them (see
    ``abelian_higgs_3d.nlo_assembly``).

    The bounce action carries :math:`\int d^3x\,[\tfrac12 Z_3 (\partial_i
    \phi_3)^2 + V_3]` with :math:`Z_3 = 1 + Z_3^{\rm NLO}`, and this returns
    :math:`Z_3^{\rm NLO}` for the radial direction:

    .. math::
        Z_3^{\rm NLO} = -\frac{11 g_3}{24\pi\,\phi_3}
                        + \frac{\lambda_{VL}^2 \phi_3^2}{192\pi\, m_{X_0}^3}

    DRalgo prints the matrix ``dZ`` for the convention
    ``(delta^ij + 2 dZ^ij) d_mu R[i] d^mu R[j]``, so this is twice its (2,2)
    entry. Three things fix that factor rather than leaving it to the
    convention statement:

    * the soft and ultrasoft runs agree on the gauge part exactly, DRalgo's
      soft (2,2) entry reducing to its ultrasoft one when lambda_VL vanishes;
    * both terms differ from eq. (4.8) of arXiv:2503.13597 by exactly a factor
      three, which is the ratio of SU(2) to U(1) multiplicities -- 6 spatial
      polarisations against 2, and 3 temporal scalars against 1 -- and that
      same factor three is what relates their leading-order cubic to ours;
    * the two terms need the *same* factor, which a wrong overall normalisation
      would not give.

    Note this diverges as ``1/phi3`` at the origin. That is physical, not a
    transcription artefact -- the derivative expansion is being pushed where
    the gauge modes are light -- and it is why the term is reported separately
    rather than folded into the potential. arXiv:2503.13597 finds this
    contribution large.
    """
    import numpy as np

    phi = np.asarray(phi3, dtype=float)
    m0_sq = temporal_mass_sq_soft(phi, debye_mass_sq, lambda_vl)
    if np.any(m0_sq <= 0.0):
        raise ValueError("the temporal gauge mass squared must stay positive")
    gauge = -11.0 * np.sqrt(g3_sq) / (24.0 * np.pi * phi)
    temporal = lambda_vl**2 * phi**2 / (192.0 * np.pi * m0_sq**1.5)
    return gauge + temporal


# ----------------------------------------------------------------------
# the complete soft-theory potential, scalar loops included
#
# Our rate never uses these: the leading-order bounce takes the gauge loops
# only, and the NLO layer takes the gauge-sector two-loop action above plus
# fluctuation determinants. They exist for comparing with calculations that
# instead re-solve the bounce on the full two-loop potential of the soft theory
# (Landau gauge, A_0 kept) -- arXiv:2511.02910 -- and they are assembled from
# pieces that are already DRalgo output and tested:
#   one loop: the soft field-dependent masses DRalgo printed in runs 5 and 6;
#   two loop: the ultrasoft theory's potential (run 2), which is the soft
#     theory's with A_0 removed, plus the three diagrams that contain A_0 --
#     A_0 figure-eight (lambda_VLL), A_0-scalar figure-eight (lambda_VL) and
#     the A_0 A_0 h sunset (lambda_VL^2 phi^2).
# The A_0 diagrams reduce to V3_soft_two_loop's temporal terms when the light
# scalar is massless, the A_0-scalar figure-eight is the scalar loop's response
# to DRalgo's own ultrasoft mass shift -lambda_VL m_D/(8 pi), and the whole
# satisfies the 3D renormalisation-group equation; tests/test_soft_potential_full.py.
# ----------------------------------------------------------------------
def V3_soft_one_loop_full(phi3, m3_sq: float, lam3: float, g3_sq: float,
                          debye_mass_sq: float, lambda_vl: float):
    """Tree level plus one loop in the soft theory, every field included.

    ``m3^2 phi^2/2 + lam3 phi^4/4 - (1/12 pi) [2 m_X^3 + m_{X0}^3 - m_D^3
    + m_G^3 + m_h^3 - 2 m_3^3]``, with ``m_X^2 = g3^2 phi^2``,
    ``m_{X0}^2 = m_D^2 + lambda_VL phi^2/2``, ``m_G^2 = m3^2 + lam3 phi^2``,
    ``m_h^2 = m3^2 + 3 lam3 phi^2``; zero at the origin. A negative scalar mass
    squared contributes the real part of ``(m^2)^{3/2}``, which is zero. With
    the scalar loops dropped and ``lambda_VL = 2 g3^2`` this is
    ``abelian_higgs_3d.V3_LO``.
    """
    import numpy as np

    phi = np.asarray(phi3, dtype=float)
    p2 = phi**2

    def cube(msq):
        return np.where(msq > 0.0, np.abs(msq)**1.5, 0.0)

    loops = (2.0 * (g3_sq * p2)**1.5
             + (debye_mass_sq + 0.5 * lambda_vl * p2)**1.5 - debye_mass_sq**1.5
             + cube(m3_sq + lam3 * p2) + cube(m3_sq + 3.0 * lam3 * p2)
             - 2.0 * cube(np.asarray(m3_sq)))
    out = 0.5 * m3_sq * p2 + 0.25 * lam3 * p2**2 - loops / (12.0 * np.pi)
    return float(out) if np.ndim(phi3) == 0 else out


def dV3_soft_one_loop_full(phi3, m3_sq: float, lam3: float, g3_sq: float,
                           debye_mass_sq: float, lambda_vl: float):
    """First derivative of :func:`V3_soft_one_loop_full` in the 3D field."""
    import numpy as np

    phi = np.asarray(phi3, dtype=float)
    p2 = phi**2
    root = lambda msq: np.sqrt(np.where(msq > 0.0, msq, 0.0))
    m0 = np.sqrt(debye_mass_sq + 0.5 * lambda_vl * p2)
    loops = (6.0 * g3_sq**1.5 * phi * np.abs(phi) + 1.5 * lambda_vl * phi * m0
             + 3.0 * lam3 * phi * root(m3_sq + lam3 * p2)
             + 9.0 * lam3 * phi * root(m3_sq + 3.0 * lam3 * p2))
    out = m3_sq * phi + lam3 * phi**3 - loops / (12.0 * np.pi)
    return float(out) if np.ndim(phi3) == 0 else out


def d2V3_soft_one_loop_full(phi3, m3_sq: float, lam3: float, g3_sq: float,
                            debye_mass_sq: float, lambda_vl: float):
    """Second derivative of :func:`V3_soft_one_loop_full`: the radial fluctuation mass.

    Where a scalar mass squared crosses zero from above, the real part of
    ``(m^2)^{3/2}`` has a second derivative that diverges like ``1/m``; the
    divergence is integrable and is kept as it is (scalar terms contribute only
    where their mass squared is positive).
    """
    import numpy as np

    phi = np.asarray(phi3, dtype=float)
    p2 = phi**2
    m0 = np.sqrt(debye_mass_sq + 0.5 * lambda_vl * p2)

    def scalar(coef, msq):
        pos = msq > 0.0
        m = np.sqrt(np.where(pos, msq, 1.0))
        return np.where(pos, coef * lam3 * m + coef**2 / 3.0 * lam3**2 * p2 / m, 0.0)

    loops = (12.0 * g3_sq**1.5 * np.abs(phi)
             + 1.5 * lambda_vl * m0 + 0.75 * lambda_vl**2 * p2 / m0
             + scalar(3.0, m3_sq + lam3 * p2) + scalar(9.0, m3_sq + 3.0 * lam3 * p2))
    out = m3_sq + 3.0 * lam3 * p2 - loops / (12.0 * np.pi)
    return float(out) if np.ndim(phi3) == 0 else out


def soft_one_loop_subtraction(r, phi3, m3_sq: float, lam3: float, g3_sq: float,
                              debye_mass_sq: float, lambda_vl: float) -> float:
    """The action of every one-loop term of :func:`V3_soft_one_loop_full` on a bounce.

    ``-(1/12 pi) int d^3x Re[2 M_A^3 + M_0^3 - m_D^3 + sum_{s=G,H}(M_s^3 - M_s^3(0))]``:
    the subtraction a determinant rate needs when its leading-order potential
    keeps the scalar loops (arXiv:2609.12098, appendix A, "Determinant
    subtraction"); with the scalar terms dropped it is
    ``abelian_higgs_3d.double_counting_subtraction`` for ``lambda_VL = 2 g3^2``.
    """
    import numpy as np

    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi3, dtype=float)
    tree = 0.5 * m3_sq * phi**2 + 0.25 * lam3 * phi**4
    loops = V3_soft_one_loop_full(phi, m3_sq, lam3, g3_sq, debye_mass_sq, lambda_vl) - tree
    return float(4.0 * np.pi * np.trapz(r**2 * loops, r))


def V3_soft_two_loop_A0(phi3, m3_sq: float, lam3: float, debye_mass_sq: float,
                        lambda_vl: float, lambda_vll: float, mu3: float):
    """The two-loop diagrams of the soft theory that contain A_0; zero at the origin.

        lambda_VLL m_{X0}^2/(128 pi^2)                      A_0 figure-eight
      + lambda_VL m_{X0} (m_h + m_G)/(64 pi^2)              A_0-scalar figure-eight
      - lambda_VL^2 phi^2 [1/2 + ln(mu3/(2 m_{X0} + m_h))]/(64 pi^2)   sunset

    Scalar masses are continued as in ``V3_US_two_loop(tachyonic="real_part")``
    and the real part is returned.
    """
    import numpy as np

    phi = np.asarray(phi3, dtype=float)
    p2 = phi**2
    pi2 = np.pi**2

    def value(p2_):
        m0 = np.sqrt(debye_mass_sq + 0.5 * lambda_vl * p2_)
        mh = np.sqrt((m3_sq + 3.0 * lam3 * p2_) + 0j)
        mG = np.sqrt((m3_sq + lam3 * p2_) + 0j)
        return (lambda_vll * m0**2 / (128.0 * pi2)
                + lambda_vl * m0 * (mh + mG) / (64.0 * pi2)
                - lambda_vl**2 * p2_ * (0.5 + np.log(mu3 / (2.0 * m0 + mh))) / (64.0 * pi2))

    out = np.real(value(p2) - value(np.zeros_like(p2)))
    return float(out) if np.ndim(phi3) == 0 else out


def V3_soft_two_loop_full(phi3, m3_sq: float, lam3: float, g3_sq: float,
                          debye_mass_sq: float, lambda_vl: float,
                          lambda_vll: float, mu3: float, *, dps: int = 30):
    """Complete two-loop soft potential (Landau gauge, A_0 kept), real part.

    ``V3_US_two_loop`` with soft parameters plus :func:`V3_soft_two_loop_A0`;
    zero at the origin. Not the object our rate uses (see the section comment).
    """
    import numpy as np

    phi = np.atleast_1d(np.asarray(phi3, dtype=float))
    us = np.asarray(V3_US_two_loop(list(phi), m3_sq, lam3, g3_sq, mu3,
                                   dps=dps, tachyonic="real_part"))
    out = us + V3_soft_two_loop_A0(phi, m3_sq, lam3, debye_mass_sq, lambda_vl,
                                   lambda_vll, mu3)
    return float(out[0]) if np.ndim(phi3) == 0 else out


# ----------------------------------------------------------------------
# dimension six
# ----------------------------------------------------------------------
def c6_bosonic(g: float, lam: float) -> float:
    """Gauge-invariant 3D sextic coefficient, fermion-free.

    From arXiv:2503.18904 (the reduced, redundancy-removed basis):
        c6 = zeta3/(32 pi^4) [ g^6 - 1.2 g^4 lam + 5 g^2 lam^2 + (20/3) lam^3 ]
    """
    return (ZETA3 / (32.0 * math.pi**4)) * (
        g**6 - 1.2 * g**4 * lam + 5.0 * g**2 * lam**2 + (20.0 / 3.0) * lam**3
    )


def cD2phi4_bosonic(g: float, lam: float, T_GeV: float) -> float:
    """Gauge-invariant coefficient of (Phi^+Phi)(D_i Phi)^+(D_i Phi) in the soft 3D EFT, fermion-free, in 1/GeV.

    The physical (on-shell) basis of arXiv:2503.18904 eq. (2.12); the value of arXiv:2610.12284 app. A,
        alpha[D^2 phi^4] = zeta3/(960 pi^4 T) [ 51 g^4 + 200 g^2 lam - 20 lam^2 ],
    Besides (Phi^+Phi)^3 it is the only dimension-six operator that does not vanish on the
    real bounce (B_0 = B_i = 0); with the 3D field Phi = phi3/sqrt 2 it contributes (alpha/4) phi3^2 (d phi3/dr)^2
    to the action density.
    """
    return (ZETA3 / (960.0 * math.pi**4 * T_GeV)) * (51.0 * g**4 + 200.0 * g**2 * lam - 20.0 * lam**2)


def c6_fermionic(g: float, y: float, lam: float) -> float:
    """The two Weyl fermions' one-loop share of c6, in the normalisation of `c6_bosonic`.

        c6_f = -7 zeta3/(960 pi^4) [ 5 y^6 - 10 lam y^4 + 10 lam^2 y^2 - 10 g^2 lam y^2 + 2 g^4 lam ]

    Derived in `dr_conformal_u1_fermions_dim6_derivation.py` by the method that reproduces `c6_bosonic`
    (hard one-loop matching onto the Green's basis, then the tree-level 3D equations of motion).
    The y^6 term is the fermions' Coleman-Weinberg m^6 term; the lam terms come from the derivative
    operators removed by the equations of motion, the g^4 lam term from the fermion loop in the
    photon two-point function, (d_i F_ij)^2 -> g3^2 J^2. Add it to `c6_bosonic` for the full c6.
    """
    return -(7.0 * ZETA3 / (960.0 * math.pi**4)) * (
        5.0 * y**6 - 10.0 * lam * y**4 + 10.0 * lam**2 * y**2 - 10.0 * g**2 * lam * y**2
        + 2.0 * g**4 * lam
    )


def cD2phi4_fermionic(g: float, y: float, lam: float, T_GeV: float) -> float:
    """The two Weyl fermions' one-loop share of the (Phi^+Phi)(D_i Phi)^+(D_i Phi) coefficient, in 1/GeV.

        c_f = -7 zeta3/(960 pi^4 T) [ 3 g^4 - 15 g^2 y^2 + 10 y^4 ]

    Same derivation and normalisation as `c6_fermionic` and `cD2phi4_bosonic`; add it to the
    latter for the full coefficient. It does not depend on lam at this order (``lam`` is accepted
    for a uniform signature). The g^4 term is there at y = 0: it is the charged fermions' loop in
    the photon two-point function, reduced with d_i F_ij = -g3 J_j.
    """
    return -(7.0 * ZETA3 / (960.0 * math.pi**4 * T_GeV)) * (
        3.0 * g**4 - 15.0 * g**2 * y**2 + 10.0 * y**4
    )


def fermionic_sextic_fraction(g: float, y: float, d: int = 3) -> float:
    """|y|^6 / (2 d g^6): the fermion term against the gauge term in alpha[phi^6].

    Our own DRalgo run gives
        alpha[phi^6] = (2/3)( 2 d g^6 Zb - 6 g^2 xi lam^2 Zb + 56 lam^3 Zb
                              - |y|^6 Zf )
    so the dark fermion competes with the leading gauge term in this ratio, up
    to the O(1) ratio Zf/Zb of the hard thermal master integrals, which has not
    been evaluated. This term is absent from `c6_bosonic` because the published
    coefficient was derived for a fermion-free model.

    Used only as a diagnostic: when this is small the published c6 is an
    adequate stand-in, when it approaches one it is not.

    The ratio Zf/Zb is 7 (sum over odd against non-zero even Matsubara modes of
    |omega_n|^-3), see `c6_fermionic`, which gives the fermion contribution in
    full; this function is kept unchanged so the monitor's thresholds keep their
    meaning.
    """
    return y**6 / (2.0 * d * g**6)


# ----------------------------------------------------------------------
# input scheme: the TL model's Coleman-Weinberg quartic -> MS-bar
# ----------------------------------------------------------------------
#: Constant in the Landau-gauge MS-bar one-loop potential,
#: n m^4 (ln(m^2/mubar^2) - k) / (64 pi^2), for each kind of species.
K_MSBAR_SCALAR = 1.5
K_MSBAR_FERMION = 1.5
K_MSBAR_VECTOR = 5.0 / 6.0
#: The TL model file (`TL_conformal_u1_fermions.py`, V1) instead subtracts 25/6 for
#: every species and takes each log relative to that species' own mass at
#: phi = v. That is the Coleman-Weinberg renormalisation condition: lambda is
#: fixed by V''''(v) = 6 lambda, and V'(v) = 0 then gives `calcSelfcouping`.
K_CW = 25.0 / 6.0


def one_loop_species(g: float, y: float, lam: float) -> tuple:
    """``(n, c, k)`` for every species of the TL one-loop potential.

    ``m^2 = c phi^2``; ``n`` counts degrees of freedom, negative for fermions;
    ``k`` is the MS-bar constant. The same list as the model's V1: radial scalar
    (3 lam), Goldstone (lam), three vector polarisations (g^2), two Weyl
    fermions of two degrees of freedom each (y^2/2).
    """
    return ((1, 3.0 * lam, K_MSBAR_SCALAR), (1, lam, K_MSBAR_SCALAR),
            (3, g**2, K_MSBAR_VECTOR), (-4, y**2 / 2.0, K_MSBAR_FERMION))


def input_scale(g: float, v_GeV: float) -> float:
    """The scale at which (g, y) are read as MS-bar couplings: m_X = g v.

    The convention of arXiv:2607.18233 (app. B) and Kierkla:2022odc, so our
    inputs mean the same as theirs.
    """
    return g * v_GeV


def msbar_quartic_from_cw(g: float, y: float, lam_cw: float, v_GeV: float,
                          mu0_GeV: float, *, scalar_loops: bool = True) -> float:
    """MS-bar quartic at ``mu0`` that gives the same one-loop potential as TL.

    Both potentials are lam phi^4/4 + sum_i n_i c_i^2 phi^4 (ln phi^2 + const_i)
    / (64 pi^2) with the same coefficient of phi^4 ln phi^2, so they are the
    same function of phi exactly when the phi^4 coefficients agree:

        lam(mu0) = lam_CW - sum_i n_i c_i^2 [ln(c_i v^2/mu0^2) + 25/6 - k_i] / (16 pi^2)

    With gauge loops only and mu0 = g v this is lam(mu0) = g^4/(16 pi^2), the
    value arXiv:2607.18233 gets by minimising the MS-bar potential directly,
    while lam_CW = 11 g^4/(16 pi^2). The two quartics describe one potential in
    two schemes.

    ``scalar_loops=False`` drops the radial and Goldstone loops (O(g^8)); it is
    only for comparing with published gauge-only expressions. The scalar masses
    in the loops use ``lam_cw``; using the MS-bar quartic differs at two loops.
    """
    total = 0.0
    for n, c, k in one_loop_species(g, y, lam_cw):
        if c == 0.0 or (not scalar_loops and k == K_MSBAR_SCALAR and n > 0):
            continue
        total += n * c * c * (math.log(c * v_GeV**2 / mu0_GeV**2) + K_CW - k)
    return lam_cw - total / (16.0 * math.pi**2)


def vev_from_zero_quartic(g: float, y: float, mu0_GeV: float) -> float:
    """Minimum of the one-loop MS-bar potential when ``lam(mu0) = 0``.

    With lam = 0 the scalar loops vanish and the potential is
    sum_i n_i c_i^2 phi^4 (ln(c_i phi^2/mu0^2) - k_i)/(64 pi^2) over the vector
    (n = 3, c = g^2, k = 5/6) and the fermions (n = -4, c = y^2/2, k = 3/2).
    It is stationary at

        ln(v^2/mu0^2) = [3 g^4 (1/3 - ln g^2) + y^4 (ln(y^2/2) - 1)] / (3 g^4 - y^4),

    a minimum only for y^4 < 3 g^4. At y = 0 this is e^{1/6} mu0/g, the
    fermion-free model's ``vev_from_zero_quartic``.
    """
    a = 3.0 * g**4 - y**4
    if a <= 0.0:
        raise ValueError(f"no minimum at lam(mu0) = 0: y^4 = {y**4:.4g} >= 3 g^4 = {3 * g**4:.4g}")
    log_v2 = (3.0 * g**4 * (1.0 / 3.0 - math.log(g**2))
              + (y**4 * (math.log(y**2 / 2.0) - 1.0) if y != 0.0 else 0.0)) / a
    return mu0_GeV * math.exp(0.5 * log_v2)


def vacuum_energy_zero_quartic(g: float, y: float, mu0_GeV: float) -> float:
    """V(0) - V(v) [GeV^4] of the potential in ``vev_from_zero_quartic``: (3 g^4 - y^4) v^4/(128 pi^2).

    Gauge and fermion loops only; the TL model adds the scalar loops with its
    Coleman-Weinberg quartic, a correction below one per cent for g <= 0.7.
    """
    v = vev_from_zero_quartic(g, y, mu0_GeV)
    return (3.0 * g**4 - y**4) * v**4 / (128.0 * math.pi**2)


#: What the conversion above does not do.
#:  * g and y are read as MS-bar couplings at mu0 = g v rather than derived
#:    from physical observables, and v is the minimum of the one-loop potential
#:    rather than fixed by pole masses: the vacuum renormalisation with
#:    momentum-dependent self-energies (Biondini:2026uds, Sojka:2024btp) is not
#:    done -- nor in arXiv:2607.18233.
#:  * Which mu0 counts as the input scale is a higher-order choice. Moving it a
#:    factor of 2 either way, with g and y run to it, moves the quartic at the
#:    matching scale by less than one per cent for g <= 0.7 and by a few per
#:    cent at g = 1.
INPUT_SCHEME_LIMITATIONS = (
    "one-loop-potential level: (g, y) MS-bar at mu0 = g v, v = minimum of the "
    "one-loop potential; no pole-mass vacuum renormalisation."
)


# ----------------------------------------------------------------------
# the matched soft-theory parameters at one temperature
# ----------------------------------------------------------------------
def matched_soft_parameters(g: float, y: float, lam: float, v_GeV: float,
                            T_GeV: float, mu4_over_T: float = 7.0553,
                            matching_order: str = "NLO",
                            input_scheme: str = "CW") -> dict:
    """Everything the 3D side needs at one temperature, in one place.

    ``lam`` is the TL model's quartic (``specific_potential.l``). It is
    converted to MS-bar at the input scale, the 4D couplings are run from there
    to ``mu = mu4_over_T * T``, and matched.

    ``matching_order``:
      "NLO" -- next-to-leading-order thermal masses and g3^2 with the one-loop
               quartic. The consistent choice, and the default.
      "LO"  -- leading-order masses and g3^2 with the one-loop quartic. These masses
               carry no mubar, so nothing cancels the 4D matching scale: the
               matching-scale band is wider, and at sub-GeV vevs T_perc differs
               from the NLO value by order one.

    ``input_scheme``:
      "CW"          -- ``lam`` is the Coleman-Weinberg quartic of the TL model
                       file, converted with `msbar_quartic_from_cw` at
                       mu0 = g v. The only scheme of this function; the
                       zero-quartic input is `matched_soft_parameters_zero_quartic`.

    The 3D renormalisation scale ``mu3`` is tied to the 3D theory's own scale,
    g3^2.

    Also returned, for the validity monitor: ``c6``, the dimension-six
    coefficient of the first operator the reduction drops, at the matching
    scale (fermion-free formula; the fermions' own share is
    `fermionic_sextic_fraction`), and ``fermion_masses_over_field``, each hard
    fermion's mass per unit 4D field, m_chi = |y| phi / sqrt 2 for both Weyl
    fermions -- they are not in the 3D theory, so the monitor cannot read them
    off it.
    """
    mu = mu4_over_T * T_GeV
    if input_scheme == "CW":
        mu0 = input_scale(g, v_GeV)
        lam0 = msbar_quartic_from_cw(g, y, lam, v_GeV, mu0)
    else:
        raise ValueError(f"input_scheme must be 'CW', not {input_scheme!r}")
    gr, yr, lr, _ = rg_run(g, y, lam0, mu0, mu)
    g3sq = gauge_coupling_sq(gr, T_GeV, mu, matching_order)
    lam3 = lambda3_NLO(gr, yr, lr, T_GeV, mu)
    if matching_order == "NLO":
        m3sq = scalar_thermal_mass_sq_NLO(gr, yr, lr, T_GeV, mu, mu3=g3sq)
        mD2 = debye_mass_sq_NLO(gr, yr, lr, T_GeV, mu)
    elif matching_order == "LO":
        m3sq = scalar_thermal_mass_sq_LO(gr, yr, lr, T_GeV)
        mD2 = debye_mass_sq_LO(gr, T_GeV)
    else:
        raise ValueError(f"matching_order must be 'LO' or 'NLO', not "
                         f"{matching_order!r}")
    return dict(T=T_GeV, mu4=mu, m3sq=m3sq, lam3=lam3, g3sq=g3sq, mD2=mD2,
                mu3=g3sq, lambda_vl=lambda_VL(gr, yr, lr, T_GeV, mu),
                lambda_vll=lambda_VLL(gr, T_GeV), mu0=mu0, lam0_msbar=lam0,
                c6=c6_bosonic(gr, lr),
                fermion_masses_over_field=(abs(yr) / math.sqrt(2.0),) * 2)


def matched_soft_parameters_zero_quartic(g: float, y: float, mu0_GeV: float, T_GeV: float,
                                         mu4_over_T: float = 7.0553,
                                         matching_order: str = "NLO",
                                         mu3: float | None = None) -> dict:
    """The 3D parameters at one temperature from MS-bar inputs ``(g, y)`` and lam = 0 at ``mu0``.

    The input of the fermion-free model (`dr_conformal_u1_no_fermions_matching.matched_soft_parameters`
    with lam = 0), with the fermions: same structure, same keys, same
    defaults (``g3sq`` at ``matching_order``, ``lam3`` at one loop, ``mu3`` = ``g3sq`` unless given).
    The vev is `vev_from_zero_quartic`.
    ``c6`` and ``c_D2phi4`` include the fermions: the bosonic coefficients of arXiv:2610.12284 app. A
    plus `c6_fermionic` and `cD2phi4_fermionic`.
    """
    mu = mu4_over_T * T_GeV
    gr, yr, lr, _ = rg_run(g, y, 0.0, mu0_GeV, mu)
    g3sq = gauge_coupling_sq(gr, T_GeV, mu, matching_order)
    if mu3 is None:
        mu3 = g3sq
    lam3 = lambda3_NLO(gr, yr, lr, T_GeV, mu)
    if matching_order == "NLO":
        m3sq = scalar_thermal_mass_sq_NLO(gr, yr, lr, T_GeV, mu, mu3=mu3)
        mD2 = debye_mass_sq_NLO(gr, yr, lr, T_GeV, mu)
    elif matching_order == "LO":
        m3sq = scalar_thermal_mass_sq_LO(gr, yr, lr, T_GeV)
        mD2 = debye_mass_sq_LO(gr, T_GeV)
    else:
        raise ValueError(f"matching_order must be 'LO' or 'NLO', not {matching_order!r}")
    return dict(T=T_GeV, mu4=mu, m3sq=m3sq, lam3=lam3, g3sq=g3sq, mD2=mD2,
                mu3=mu3, lambda_vl=lambda_VL(gr, yr, lr, T_GeV, mu),
                lambda_vll=lambda_VLL(gr, T_GeV), mu0=mu0_GeV, lam0_msbar=0.0,
                g4=gr, y4=yr, lam4=lr, c6=c6_bosonic(gr, lr) + c6_fermionic(gr, yr, lr),
                c_D2phi4=cD2phi4_bosonic(gr, lr, T_GeV) + cD2phi4_fermionic(gr, yr, lr, T_GeV),
                fermion_masses_over_field=(abs(yr) / math.sqrt(2.0),) * 2)


# ----------------------------------------------------------------------
# one-loop renormalisation-group running
# ----------------------------------------------------------------------
def rg_run(g: float, y: float, lam: float, mu_from: float, mu_to: float,
           msq: float = 0.0, rtol: float = 1e-10, atol: float = 1e-14, *,
           betas=None):
    """Run (g, y, lam, msq) from ``mu_from`` to ``mu_to`` at one loop.

    The beta functions above are d X / d ln(mubar), so the integration variable
    is t = ln(mubar). Returns ``(g, y, lam, msq)`` at ``mu_to``.

    ``betas`` replaces this model's beta functions by another model's, as a
    tuple ``(beta_g2, beta_y, beta_lambda, beta_msq)`` with the signatures
    above; ``dr_conformal_u1_no_fermions_matching`` uses it for the model without fermions,
    so both models share one integrator and one set of perturbativity checks.

    Both scales must be positive. Running is done in the MS-bar scheme, so the
    inputs must already be MS-bar couplings; the TL model's quartic is not one
    (see `msbar_quartic_from_cw`).

    Raises RuntimeError if the integration leaves the perturbative regime,
    rather than returning a number that looks fine.
    """
    from scipy.integrate import solve_ivp
    from transitionlistener.eft_rate import rg

    if mu_from <= 0.0 or mu_to <= 0.0:
        raise ValueError("renormalisation scales must be positive")
    if mu_from == mu_to:
        return g, y, lam, msq
    b_g2, b_y, b_lam, b_msq = betas if betas is not None else (
        beta_g2, beta_y, beta_lambda, beta_msq)

    def rhs(_t, u):
        gg, yy, ll, mm = u
        # integrate g itself, not g^2:  dg/dt = beta(g^2)/(2 g)
        return [
            b_g2(gg, yy, ll) / (2.0 * gg) if gg > 0 else 0.0,
            b_y(gg, yy, ll),
            b_lam(gg, yy, ll),
            b_msq(gg, yy, ll, mm),
        ]

    t_from, t_to = math.log(mu_from), math.log(mu_to)
    u = rg.flow((g, y, lam, msq, mu_from, b_g2, rtol, atol), rhs, t_from, [g, y, lam, msq], t_to, rtol, atol)
    if u is None:
        sol = solve_ivp(rhs, (t_from, t_to), [g, y, lam, msq],
                        rtol=rtol, atol=atol, dense_output=False, method="RK45")
        if not sol.success:
            raise RuntimeError(f"RG running failed: {sol.message}")
        u = sol.y[:, -1]
    gf, yf, lf_, mf = (float(v) for v in u)
    if not all(math.isfinite(v) for v in (gf, yf, lf_, mf)):
        raise RuntimeError("RG running produced a non-finite coupling")
    if gf**2 > 4.0 * math.pi or yf**2 > 4.0 * math.pi:
        raise RuntimeError(
            f"RG running left the perturbative regime: g={gf:.3g}, y={yf:.3g} "
            f"at mu={mu_to:.4g}"
        )
    return gf, yf, lf_, mf
