"""The renormalised factorised determinant via BubbleDet (WP5b).

BubbleDet supplies what this package's own solver deliberately does not: the
ultraviolet subtraction that turns a partial-wave sum into an absolute
determinant. Its gauge treatment assumes the determinant factorises, which on a
bounce it does not, so the mixing correction still comes from
transitionlistener.eft_rate.abelian_determinants. These tests cover the part BubbleDet owns.
"""
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "models")
)

pytest.importorskip("BubbleDet", reason="BubbleDet is optional, for the 3D mode")

from transitionlistener.eft_rate import abelian_higgs_3d
import dr_conformal_u1_fermions_matching as dr


@pytest.fixture(scope="module")
def bounce():
    """Sub-GeV point A at its published percolation temperature."""
    Tc, f = 0.034772, 0.0656
    T = f * Tc
    mu = 7.0553 * T
    g, y, lam, _ = dr.rg_run(0.677, 0.224, 0.0146238, 0.173, mu)
    m3sq = dr.scalar_thermal_mass_sq_LO(g, y, lam, T)
    l3 = dr.lambda3_NLO(g, y, lam, T, mu)
    g3sq = dr.g3_sq_LO(g, T)
    mD2 = dr.debye_mass_sq_LO(g, T)
    S3, r, phi3, info = abelian_higgs_3d.solve_LO_bounce(m3sq, l3, g3sq,
                                              m_debye_sq=mD2)
    return dict(T=T, m3sq=m3sq, l3=l3, g3sq=g3sq, mD2=mD2,
                S3=S3, r=r, phi3=phi3)


# ------------------------------------------------- the xi = 1 assembly
def test_assembly_bookkeeping_closes(bounce):
    """The arithmetic identities the assembly must satisfy.

    These are pure bookkeeping and are the part most likely to be got wrong,
    because two conventions meet here: BubbleDet returns one-loop contributions
    to the action (already half a log determinant) while this package's solver
    returns log-determinant differences, which therefore enter halved.
    """
    mix = -64.8                                   # at point A
    a = abelian_higgs_3d.nlo_assembly(bounce["r"], bounce["phi3"], bounce["m3sq"],
                           bounce["l3"], bounce["g3sq"], bounce["mD2"],
                           bounce["S3"], mix, T=bounce["T"])
    # the counting: tests/test_gauge_mode_counting.py; the J = 0 zero mode: tests/test_j0_sector.py
    assert a["one_loop"] == pytest.approx(
        a["radial"] + a["goldstone_xi1"] + a["gauge_single_channel"]
        + 0.5 * mix + a["temporal"] + a["j0_correction"],
        rel=1e-12)
    # C enters with a plus sign in the exponent (2503.13597 eq. 4.20), so it is
    # subtracted from the action
    assert a["total_without_NLO_potential"] == pytest.approx(
        a["S3_LO"] + a["one_loop"] - a["double_counting_C"], rel=1e-12)


def test_the_mixing_correction_enters_halved(bounce):
    """A log-determinant difference contributes half of itself to the action.

    Getting this factor wrong is worth a factor of exp(16) on the rate at this
    benchmark, and nothing else in the assembly would flag it.
    """
    kw = (bounce["r"], bounce["phi3"], bounce["m3sq"], bounce["l3"],
          bounce["g3sq"], bounce["mD2"], bounce["S3"])
    a = abelian_higgs_3d.nlo_assembly(*kw, -64.8, T=bounce["T"])
    b = abelian_higgs_3d.nlo_assembly(*kw, -64.8 - 2.0, T=bounce["T"])
    assert (b["total_without_NLO_potential"]
            - a["total_without_NLO_potential"]) == pytest.approx(-1.0, rel=1e-9)


def test_the_goldstone_channel_is_the_xi_one_one_not_landau(bounce):
    """At xi = 1 the Goldstone is heavy, and that is the point.

    Its mass is V'/phi + g^2 phi^2, so on this bounce it sits close to the
    gauge channel rather than near zero. If it came out close to the Landau
    value instead, the assembly would be mixing gauges.
    """
    a = abelian_higgs_3d.nlo_assembly(bounce["r"], bounce["phi3"], bounce["m3sq"],
                           bounce["l3"], bounce["g3sq"], bounce["mD2"],
                           bounce["S3"], -64.8, T=bounce["T"])
    assert a["goldstone_xi1"] < 0.0
    assert abs(a["goldstone_xi1"]) > 0.5 * abs(a["gauge_single_channel"])

