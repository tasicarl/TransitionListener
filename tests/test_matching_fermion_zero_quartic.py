"""Zero-quartic input for the conformal U(1) with two Weyl fermions.

lambda(mu0) = 0 with (g, y) given at mu0: the input of the fermion-free pipeline. Checked
against a numerical minimum of the one-loop MS-bar potential, the fermion-free limit, the
table of an independent implementation at one benchmark point,
and the TL model's ``input_scheme = "zero_quartic"``.
"""
import math
import os
import sys

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_no_fermions_matching as nf  # noqa: E402
import dr_conformal_u1_fermions_matching as dark  # noqa: E402


def _msbar_potential(phi, g, y, mu0):
    """One-loop MS-bar potential at lam = 0, typed independently of the module."""
    v = 3 * g**4 * phi**4 * (np.log(g**2 * phi**2 / mu0**2) - 5 / 6)
    if y:
        v -= 4 * (y**2 / 2) ** 2 * phi**4 * (np.log(y**2 / 2 * phi**2 / mu0**2) - 1.5)
    return v / (64 * math.pi**2)


@pytest.mark.parametrize("g,y,mu0", [(0.3, 0.2, 1.0), (0.5, 0.3, 1.0), (0.7, 0.5, 1.0), (1.0, 0.8, 37.0)])
def test_vev_and_vacuum_energy_are_the_minimum(g, y, mu0):
    v = dark.vev_from_zero_quartic(g, y, mu0)
    r = minimize_scalar(lambda p: _msbar_potential(p, g, y, mu0), bracket=(0.3 * v, v, 3 * v), tol=1e-12)
    assert r.x == pytest.approx(v, rel=1e-7)
    assert -r.fun == pytest.approx(dark.vacuum_energy_zero_quartic(g, y, mu0), rel=1e-10)


def test_vev_without_fermions_is_the_fermion_free_one():
    for g in (0.4, 0.7, 1.1):
        assert dark.vev_from_zero_quartic(g, 0.0, 1.0) == pytest.approx(nf.vev_from_zero_quartic(g, 1.0), rel=1e-14)


def test_no_minimum_when_the_fermions_win():
    with pytest.raises(ValueError):
        dark.vev_from_zero_quartic(0.5, 0.7, 1.0)          # y^4 = 0.24 > 3 g^4 = 0.19


def test_an_independent_benchmark_table():
    """g = 0.7, y = 0.5, lam = 0 at 1 GeV, T = 0.1 GeV, mu4 = 2 pi T, mu3 = g T: the table of an independent
    implementation (printed to 7 digits, so 5e-7 relative). Its m3^2 has the opposite sign, and its log term in m3^2
    takes the soft couplings at NLO where ours takes g3^2 at LO; that row is compared to 1e-3."""
    T = 0.1
    p = dark.matched_soft_parameters_zero_quartic(0.7, 0.5, 1.0, T, mu4_over_T=2 * math.pi, mu3=0.7 * T)
    assert p["y4"] == pytest.approx(5.003444e-01, rel=5e-7)
    assert p["g4"] ** 2 == pytest.approx(4.890597e-01, rel=5e-7)
    assert p["lam4"] == pytest.approx(-3.891042e-03, rel=5e-7)
    assert p["mD2"] == pytest.approx(2.044527e-03, rel=5e-7)
    assert p["lam3"] == pytest.approx(1.248443e-04, rel=5e-7)
    assert p["lambda_vl"] == pytest.approx(9.790934e-02, rel=5e-7)
    assert p["lambda_vll"] == pytest.approx(2.120470e-03, rel=5e-7)
    assert -p["m3sq"] == pytest.approx(-1.319165e-03, rel=1e-3)


def test_tl_model_zero_quartic_matches_the_module():
    import TL_conformal_u1_fermions_3d as m3d
    g, y, mu0 = 0.6, 0.4, 1.0
    v = dark.vev_from_zero_quartic(g, y, mu0)
    pot = m3d.specific_potential({"g": g, "y": y, "v_GeV": v}, verbose=False)
    conf = pot.config.nucleationConf
    conf.mu4_over_T, conf.input_scheme = 7.0553, "zero_quartic"
    for T in (0.05, 0.2):
        conf.mu3_choice = "g3sq"
        got = pot.matched_3d_parameters(T)
        p = dark.matched_soft_parameters_zero_quartic(g, y, mu0, T)
        assert got == pytest.approx((p["m3sq"], p["lam3"], p["g3sq"], p["mD2"]), rel=1e-12)
        conf.mu3_choice = "gT"
        p = dark.matched_soft_parameters_zero_quartic(g, y, mu0, T, mu3=g * T)
        assert pot.matched_3d_parameters(T) == pytest.approx((p["m3sq"], p["lam3"], p["g3sq"], p["mD2"]), rel=1e-12)
        conf.mu3_choice = "g3sq"


def test_the_dimension_six_coefficients_include_the_fermions():
    T = 0.1
    p = dark.matched_soft_parameters_zero_quartic(0.7, 0.5, 1.0, T, mu4_over_T=2 * math.pi, mu3=0.7 * T)
    g, y, lam = p["g4"], p["y4"], p["lam4"]
    assert p["c6"] == pytest.approx(dark.c6_bosonic(g, lam) + dark.c6_fermionic(g, y, lam), rel=1e-14)
    assert p["c_D2phi4"] == pytest.approx(dark.cD2phi4_bosonic(g, lam, T) + dark.cD2phi4_fermionic(g, y, lam, T), rel=1e-14)
    assert p["c6"] != pytest.approx(dark.c6_bosonic(g, lam), rel=1e-3)
