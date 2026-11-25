"""Dimensionally reduced matching relations for the conformal U(1) WITHOUT fermions.

GENERATED FROM DRalgo -- do not hand-edit the expressions.

This is the gauge-plus-scalar model of arXiv:2511.02910, 2609.12098 and
2607.18233 (dark sector only): a U(1) gauge boson and a charge-one complex
scalar, no fermions. It exists to compare with those papers like for like.
The model with fermions at ``y -> 0`` is not this model: its two Weyl fermions
still carry gauge charge, which raises the Debye mass by 25%, doubles
``beta(g^2)`` and changes the ``g^4`` terms of every one-loop matching
relation.

Provenance
----------
Generated with  : DRalgo 1.5.0.
Checks passed   : Debye mass LO = g^2 T^2/3; beta(g^2) = g^4/(24 pi^2); the soft
                  index layout {1,2} = Phi, {3} = A_0.

Conventions are those of ``dr_conformal_u1_fermions_matching`` (V = msq |Phi|^2 +
lam |Phi|^4, beta functions d/d ln mubar, mass dimensions as there). Everything
that depends only on the 3D theory -- which is the same for both models: U(1),
one charge-one scalar, A_0 -- is imported from that module rather than copied:
the logarithm ``Lb``, the soft two-loop potentials, the kinetic term, the
input-scheme conversion and the renormalisation-group integrator. This module
holds only what the fermions change: the hard matching and the beta functions.

Every relation here also agrees, term by term, with the appendix of
arXiv:2511.02910 (eqs. A.2-A.6), which printed DRalgo output for the same
model; ``tests/test_matching_nofermion.py`` checks that, the scale
independence, and the difference from the fermion model.
"""
from __future__ import annotations

import math

import dr_conformal_u1_fermions_matching as _dark
from dr_conformal_u1_fermions_matching import (  # noqa: F401  (re-exported)
    EULER_GAMMA,
    LOG_GLAISHER,
    Lb,
    V3_soft_one_loop_full,
    dV3_soft_one_loop_full,
    d2V3_soft_one_loop_full,
    soft_one_loop_subtraction,
    V3_soft_two_loop,
    V3_soft_two_loop_A0,
    V3_soft_two_loop_full,
    V3_US_two_loop,
    Z3_soft_NLO,
    input_scale,
    temporal_mass_sq_soft,
)


# ----------------------------------------------------------------------
# 4D one-loop beta functions,  d X / d ln(mubar)
#   DRalgo run 6, "### BetaFunctions4D":
#     {g^2 -> g^4/(24 Pi^2),
#      lam -> (3 g^4 - 6 g^2 lam + 10 lam^2)/(8 Pi^2),
#      msq -> -1/8 (msq (3 g^2 - 4 lam))/Pi^2}
# Signatures keep an unused y so that `dr_conformal_u1_fermions_matching.rg_run`
# can integrate them.
# ----------------------------------------------------------------------
def beta_g2(g: float, y: float = 0.0, lam: float = 0.0) -> float:
    return g**4 / (24.0 * math.pi**2)


def beta_lambda(g: float, y: float, lam: float) -> float:
    return (3.0 * g**4 - 6.0 * g**2 * lam + 10.0 * lam**2) / (8.0 * math.pi**2)


def beta_y(g: float, y: float, lam: float) -> float:
    return 0.0


def beta_msq(g: float, y: float, lam: float, msq: float) -> float:
    return -msq * (3.0 * g**2 - 4.0 * lam) / (8.0 * math.pi**2)


BETAS = (beta_g2, beta_y, beta_lambda, beta_msq)


def rg_run(g: float, lam: float, mu_from: float, mu_to: float, msq: float = 0.0,
           **kwargs):
    """Run (g, lam, msq) at one loop; returns ``(g, lam, msq)`` at ``mu_to``.

    The integrator and its perturbativity checks are the fermion model's
    ``rg_run``, given this model's beta functions.
    """
    gf, _y, lf, mf = _dark.rg_run(g, 0.0, lam, mu_from, mu_to, msq, betas=BETAS, **kwargs)
    return gf, lf, mf


# ----------------------------------------------------------------------
# hard -> soft matching
#   DRalgo run 6, "### PrintCouplings", "### PrintScalarMass LO/NLO",
#   "### PrintDebyeMass LO/NLO", "### TemporalScalarCouplings"
# ----------------------------------------------------------------------
def g3_sq_LO(g: float, T: float) -> float:
    """3D gauge coupling squared, leading order: ``g^2 T``."""
    return g**2 * T


def g3_sq_NLO(g: float, T: float, mubar: float) -> float:
    """``g3d^2 -> g^2 T - g^4 Lb T/(48 Pi^2)``. Only Lb: no fermion loop."""
    return g**2 * T - g**4 * Lb(mubar, T) * T / (48.0 * math.pi**2)


def gauge_coupling_sq(g: float, T: float, mubar: float, matching_order: str) -> float:
    """g3^2 at the matching order of the masses: `g3_sq_NLO` for "NLO", `g3_sq_LO` for "LO"."""
    if matching_order == "NLO":
        return g3_sq_NLO(g, T, mubar)
    if matching_order == "LO":
        return g3_sq_LO(g, T)
    raise ValueError(f"matching_order must be 'LO' or 'NLO', not {matching_order!r}")


def lambda3_NLO(g: float, lam: float, T: float, mubar: float) -> float:
    """``lam3d -> T (g^4 (2 - 3 Lb) + 6 g^2 Lb lam + 2 lam (8 Pi^2 - 5 Lb lam))/(16 Pi^2)``.

    The ``g^4`` term is the Coleman-Weinberg contribution; see
    ``dr_conformal_u1_fermions_matching.lambda3_LO`` for why it is kept.
    """
    lb = Lb(mubar, T)
    return (T * (g**4 * (2.0 - 3.0 * lb) + 6.0 * g**2 * lb * lam
                 + 2.0 * lam * (8.0 * math.pi**2 - 5.0 * lb * lam))
            / (16.0 * math.pi**2))


def lambda_VL(g: float, lam: float, T: float, mubar: float) -> float:
    """``lamVL[1] -> g^2 T (-g^2 (-4 + Lb) + 24 (2 Pi^2 + lam))/(24 Pi^2)``.

    The ``A_0^2 |Phi|^2`` coupling, ``h_3`` of arXiv:2607.18233; leading term
    ``2 g^2 T``.
    """
    return (g**2 * T * (-g**2 * (-4.0 + Lb(mubar, T)) + 24.0 * (2.0 * math.pi**2 + lam))
            / (24.0 * math.pi**2))


def lambda_VLL(g: float, T: float) -> float:
    """``lamVLL[1] -> g^4 T/Pi^2`` (the fermion model has 7/8 of this)."""
    return g**4 * T / math.pi**2


def scalar_thermal_mass_sq_LO(g: float, lam: float, T: float, msq: float = 0.0) -> float:
    """``msq3d -> msq + T^2 (3 g^2 + 4 lam)/12``."""
    return msq + T**2 * (3.0 * g**2 + 4.0 * lam) / 12.0


def scalar_thermal_mass_sq_NLO(g: float, lam: float, T: float, mubar: float,
                               mu3: float, msq: float = 0.0, *,
                               g3d_sq: float | None = None,
                               lam3d: float | None = None,
                               lam_vl: float | None = None) -> float:
    """3D scalar mass squared, LO plus DRalgo's (additive) NLO correction.

    ``(12 g^2 (Lb (9 msq - 6 T^2 lam) + 2 T^2 lam (1 + 6 gE - 72 Log[A]))
       + 24 lam (Lb (-6 msq + T^2 lam) - 6 T^2 lam (gE - 12 Log[A]))
       + g^4 T^2 (-8 - 108 gE + 69 Lb + 1296 Log[A])
       + 18 Log[mu3/mubar] (8 g3d^4 - 16 g3d^2 lam3d + 16 lam3d^2 + lamVL^2))
      /(576 Pi^2)``

    The 3D couplings inside the logarithm default to the same choice as the
    fermion model's production code (``g3d^2`` at leading order, ``lam3d`` and
    ``lamVL`` at one loop); pass them explicitly for another choice. Which one
    is taken is a higher-order difference: arXiv:2511.02910 uses all three at
    one loop in its "2-L (NLO)" scheme and at tree level in its eq. (A.6).
    """
    lb = Lb(mubar, T)
    if g3d_sq is None:
        g3d_sq = g3_sq_LO(g, T)
    if lam3d is None:
        lam3d = lambda3_NLO(g, lam, T, mubar)
    if lam_vl is None:
        lam_vl = lambda_VL(g, lam, T, mubar)
    correction = (
        12.0 * g**2 * (lb * (9.0 * msq - 6.0 * T**2 * lam)
                       + 2.0 * T**2 * lam * (1.0 + 6.0 * EULER_GAMMA - 72.0 * LOG_GLAISHER))
        + 24.0 * lam * (lb * (-6.0 * msq + T**2 * lam)
                        - 6.0 * T**2 * lam * (EULER_GAMMA - 12.0 * LOG_GLAISHER))
        + g**4 * T**2 * (-8.0 - 108.0 * EULER_GAMMA + 69.0 * lb + 1296.0 * LOG_GLAISHER)
        + 18.0 * math.log(mu3 / mubar) * (8.0 * g3d_sq**2 - 16.0 * g3d_sq * lam3d
                                          + 16.0 * lam3d**2 + lam_vl**2)
    ) / (576.0 * math.pi**2)
    return scalar_thermal_mass_sq_LO(g, lam, T, msq) + correction


def debye_mass_sq_LO(g: float, T: float) -> float:
    """``muSqU1 -> g^2 T^2/3``: the charge-one complex scalar alone."""
    return g**2 * T**2 / 3.0


def debye_mass_sq_NLO(g: float, lam: float, T: float, mubar: float,
                      msq: float = 0.0) -> float:
    """Debye mass squared, LO plus DRalgo's (additive) NLO correction.

    DRalgo: ``g^2 (36 msq + 7 g^2 T^2 - 2 gE g^2 T^2 + 12 T^2 lam
    + 2 g^2 T^2 Log[4 Pi T] - 2 g^2 T^2 Log[mu])/(144 Pi^2)``. The logarithms
    collect into ``-g^2 T^2 Lb``: ``Lb = 2 (Log[mu] + gE - Log[4 Pi T])``.
    """
    correction = (g**2 * (36.0 * msq + 12.0 * T**2 * lam
                          + g**2 * T**2 * (7.0 - Lb(mubar, T)))
                  / (144.0 * math.pi**2))
    return debye_mass_sq_LO(g, T) + correction


# ----------------------------------------------------------------------
# inputs and the matched parameters at one temperature
# ----------------------------------------------------------------------
def vev_from_zero_quartic(g: float, mu0_GeV: float) -> float:
    """Minimum of the one-loop MS-bar potential when ``lam(mu0) = 0``, gauge loops only.

    ``3 g^4 phi^4 (ln(g^2 phi^2/mu0^2) - 5/6)/(64 pi^2)`` is stationary at
    ``g phi = e^{1/6} mu0``: arXiv:2511.02910 eq. (2.2), and the input of
    arXiv:2609.12098.
    """
    return math.exp(1.0 / 6.0) * mu0_GeV / g


def matched_soft_parameters(g: float, lam: float, mu0_GeV: float, T_GeV: float,
                            mu4_over_T: float = 7.0553,
                            matching_order: str = "NLO",
                            mu3: float | None = None) -> dict:
    """The 3D parameters at one temperature from MS-bar inputs ``(g, lam)`` at ``mu0``.

    Structured exactly like the fermion model's ``matched_soft_parameters``, so
    either can feed the same 3D code: ``g3sq`` and the masses at ``matching_order``, ``lam3`` at
    one loop. ``mu3`` defaults to ``g3sq``, as
    there. Also returns the running 4D couplings at the matching scale and the two dimension-six
    coefficients that act on the bounce, ``c6`` of (Phi^+Phi)^3 and ``c_D2phi4`` (1/GeV) of
    (Phi^+Phi)(D_i Phi)^+(D_i Phi), at the matching scale.
    """
    mu = mu4_over_T * T_GeV
    gr, lr, _ = rg_run(g, lam, mu0_GeV, mu)
    g3sq = gauge_coupling_sq(gr, T_GeV, mu, matching_order)
    if mu3 is None:
        mu3 = g3sq
    lam3 = lambda3_NLO(gr, lr, T_GeV, mu)
    if matching_order == "NLO":
        m3sq = scalar_thermal_mass_sq_NLO(gr, lr, T_GeV, mu, mu3=mu3)
        mD2 = debye_mass_sq_NLO(gr, lr, T_GeV, mu)
    elif matching_order == "LO":
        m3sq = scalar_thermal_mass_sq_LO(gr, lr, T_GeV)
        mD2 = debye_mass_sq_LO(gr, T_GeV)
    else:
        raise ValueError(f"matching_order must be 'LO' or 'NLO', not {matching_order!r}")
    return dict(T=T_GeV, mu4=mu, m3sq=m3sq, lam3=lam3, g3sq=g3sq, mD2=mD2,
                mu3=mu3, lambda_vl=lambda_VL(gr, lr, T_GeV, mu),
                lambda_vll=lambda_VLL(gr, T_GeV), mu0=mu0_GeV, lam0_msbar=lam,
                g4=gr, lam4=lr, c6=_dark.c6_bosonic(gr, lr), c_D2phi4=_dark.cD2phi4_bosonic(gr, lr, T_GeV),
                fermion_masses_over_field=())
