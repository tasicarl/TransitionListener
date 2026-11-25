"""The one-loop determinants of the 3D rate are a smooth function of T.

Two bounces 2e-8 apart in T (no fermions, g = 0.41, S3_LO/T = 180) are the same physics; their determinants differ by
less than 1e-5. With BubbleDet at its default Gelfand-Yaglom tolerance, on the shooting's own radial grid, and with the
tail amplitude of the J = 0 sector taken where phi r e^{mr} still varies, they differed by 1.1e-3.
"""
import os
import sys

import pytest

pytest.importorskip("BubbleDet")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
from transitionlistener.eft_rate import nlo_tables as nt  # noqa: E402

V_GEV = 2.881366860647917
T_GEV = 2.422872567340843e-06


@pytest.fixture(scope="module")
def two_rows():
    pot = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.41, "v_GeV": V_GEV}, verbose=False)
    return [nt.fast_row(pot, T_GEV * f) for f in (1.0, 1.0 - 2e-8)]


def test_two_neighbouring_bounces_have_the_same_determinants(two_rows):
    a, b = two_rows
    assert abs(a["S3_LO"] - b["S3_LO"]) < 1e-4
    assert a["determinants"] == pytest.approx(b["determinants"], abs=5e-5)


def test_the_tail_amplitude_is_the_asymptotic_one():
    """phi r e^{mr} at the default window equals its value a decade deeper in the tail."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    pot = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.41, "v_GeV": V_GEV}, verbose=False)
    p, S3, r, phi3, info = nt._bounce(pot, T_GEV)
    r, phi, m3sq, lam3, g3sq, mD2 = nt._in_units(p, r, phi3, T_GEV)
    kw = dict(m_debye_sq=mD2, h3=nt._h3_in_units(pot, p, T_GEV))
    default = abelian_higgs_3d.j0_sector(r, phi, m3sq, lam3, g3sq, **kw)["tail_amplitude"]
    deeper = abelian_higgs_3d.j0_sector(r, phi, m3sq, lam3, g3sq, tail_window=(1e-7, 1e-6), **kw)["tail_amplitude"]
    assert default == pytest.approx(deeper, rel=1e-6)


def test_the_gelfand_yaglom_tolerance_is_converged(monkeypatch):
    """Every BubbleDet channel at the default tolerance equals its value at a hundred times tighter."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    pot = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.41, "v_GeV": V_GEV}, verbose=False)
    p, S3, r, phi3, info = nt._bounce(pot, T_GEV)
    args = nt._in_units(p, r, phi3, T_GEV)
    h3 = nt._h3_in_units(pot, p, T_GEV)

    def channels():
        a = abelian_higgs_3d.nlo_assembly(*args, S3, 0.0, T=1.0, h3=h3)
        return [a[k] for k in ("radial", "goldstone_xi1", "gauge_single_channel", "temporal")]
    default = channels()
    monkeypatch.setattr(abelian_higgs_3d, "BUBBLEDET_GY_TOL", abelian_higgs_3d.BUBBLEDET_GY_TOL / 100.0)
    assert default == pytest.approx(channels(), abs=1e-5)


