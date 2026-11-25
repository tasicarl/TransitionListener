"""The kinetic Z-factor and the gauge determinants are the same physics.

arXiv:2503.13597 eq. (4.20) -- and arXiv:2607.18233 eq. (3.5) -- build the NLO
exponent from the LO action, the fluctuation determinants, the cubic
subtraction C and the two-loop soft potential. The kinetic term
int 1/2 Z_3 (d phi)^2 belongs to the derivative-expansion exponent, eq. (4.6),
where it stands in for the gauge determinants minus C (their paragraph on
renormalisation-scale invariance says so explicitly). `nlo_assembly` adds only the determinants.

The temporal mode shows the overlap directly: it factorises from everything
else, and for it the derivative expansion is excellent, so its determinant
minus its own cubic must reproduce the temporal part of Z_3.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

pytest.importorskip("BubbleDet")

G, Y, V, T = 0.7, 0.01, 0.173, 1.241e-3


def _bounce():
    import TL_conformal_u1_fermions as m4d
    from transitionlistener.eft_rate import abelian_higgs_3d

    lam = m4d.specific_potential({"g": G, "y": Y, "v_GeV": V}, verbose=False).l
    p = dr.matched_soft_parameters(G, Y, lam, V, T)
    S, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"],
                                          m_debye_sq=p["mD2"])
    return p, S, r, phi3


def test_temporal_determinant_minus_its_cubic_is_the_temporal_Z_term():
    from BubbleDet import BubbleConfig, ParticleConfig, BubbleDeterminant
    from transitionlistener.eft_rate import abelian_higgs_3d

    p, _S, r, phi3 = _bounce()
    m3, l3, g3, mD = p["m3sq"], p["lam3"], p["g3sq"], p["mD2"]
    kw = dict(m_debye_sq=mD, n_temporal=1)
    bubble = BubbleConfig(
        V=lambda q: abelian_higgs_3d.V3_LO(q, m3, l3, g3, 2, **kw), phi_metaMin=0.0,
        R=r, Phi=phi3, dim=3, dV=lambda q: abelian_higgs_3d.dV3_LO(q, m3, l3, g3, 2, **kw),
        d2V=lambda q: abelian_higgs_3d.d2V3_LO(np.asarray(q, float), m3, l3, g3, 2, **kw),
        dPhi=np.gradient(phi3, r))
    # the temporal mass nlo_assembly uses: m_D^2 + g3^2 phi3^2, i.e. lambda_VL = 2 g3^2
    part = ParticleConfig(W_Phi=lambda q: mD + g3 * np.asarray(q, float) ** 2,
                          spin=0, dof_internal=1, zero_modes="None")
    det, _err = BubbleDeterminant(bubble, [part]).findSingleDeterminant(part)
    cubic = abelian_higgs_3d.double_counting_subtraction(r, phi3, g3, 0, m_debye_sq=mD, n_temporal=1)

    # g3 -> 0 in Z3_soft_NLO leaves only its temporal term
    z = dr.Z3_soft_NLO(np.maximum(np.abs(phi3), 1e-12), 0.0, mD, 2.0 * g3)
    kinetic = 4.0 * np.pi * np.trapz(r**2 * 0.5 * z * np.gradient(phi3, r) ** 2, r)

    overlap = float(det) - cubic
    assert kinetic > 0.0 and overlap > 0.0
    assert overlap == pytest.approx(kinetic, rel=0.05)
    # and the overlap is a tiny remainder of two large numbers, so the agreement
    # is not an accident of both being small
    assert abs(cubic) > 100.0 * abs(overlap)


def test_the_nlo_total_does_not_include_the_kinetic_term():
    from transitionlistener.eft_rate import abelian_higgs_3d

    p, S, r, phi3 = _bounce()
    V3n = lambda ph: dr.V3_soft_two_loop(ph, p["g3sq"], p["mD2"], p["lambda_vl"],
                                         p["lambda_vll"], p["mu3"])
    Z3n = lambda ph: dr.Z3_soft_NLO(ph, p["g3sq"], p["mD2"], p["lambda_vl"])
    a = abelian_higgs_3d.nlo_assembly(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], p["mD2"],
                           S, 0.0, T=T, V3_nlo=V3n, Z3_nlo=Z3n)
    assert a["total"] == pytest.approx(a["total_without_NLO_potential"] + a["nlo_potential"],
                                       rel=1e-12)
    assert a["nlo_kinetic"] < 0.0             # still reported, for comparison
    # a kinetic term alone does not make an exponent complete
    b = abelian_higgs_3d.nlo_assembly(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], p["mD2"],
                           S, 0.0, T=T, Z3_nlo=Z3n)
    assert "total" not in b and b["nlo_kinetic"] == pytest.approx(a["nlo_kinetic"])
