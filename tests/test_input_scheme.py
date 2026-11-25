"""The TL model's quartic is a Coleman-Weinberg coupling, not an MS-bar one.

The model file fixes lambda by V''''(v) = 6 lambda with every log taken
relative to the species' own mass at phi = v and a constant 25/6. The matching
relations and beta functions are MS-bar. These tests check the conversion
between the two against the TL potential itself, against a published
closed form, and against a direct minimisation, and pin how much the choice
of input scale matters.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

LOOP = 16.0 * math.pi**2


def _msbar_potential(phi, g, y, lam_msbar, lam_loops, mu0):
    """Landau-gauge MS-bar one-loop potential, written out independently."""
    phi = np.asarray(phi, dtype=float)
    out = lam_msbar * phi**4 / 4.0
    for n, c, k in dr.one_loop_species(g, y, lam_loops):
        if c == 0.0:
            continue
        m2 = c * phi**2
        out = out + n * m2**2 * (np.log(m2 / mu0**2) - k) / (64.0 * math.pi**2)
    return out


@pytest.mark.parametrize("g,y", [(0.6, 0.01), (0.7, 0.01), (0.677, 0.224), (1.0, 0.3)])
def test_the_msbar_potential_at_mu0_is_the_tl_potential(g, y):
    """Same function of phi, to round-off, over eight orders of magnitude.

    Uses the model's own V0 and V1, so the species list and the 25/6 are
    checked against the code that produces the 4D results, not a copy of it.
    """
    import TL_conformal_u1_fermions as m4d

    pot = m4d.specific_potential({"g": g, "y": y, "v_GeV": 0.173}, verbose=False)
    v = pot.v                                        # internal units
    mu0 = dr.input_scale(g, v)
    lam_bar = dr.msbar_quartic_from_cw(g, y, pot.l, v, mu0)

    phi = v * np.logspace(-6, 2, 41)
    X = phi[:, None]
    tl = pot.V0(X) + pot.V1_from_X(X)
    ms = _msbar_potential(phi, g, y, lam_bar, pot.l, mu0)
    scale = (3.0 * g**4 / (64.0 * math.pi**2)) * phi**4
    assert np.max(np.abs(tl - ms) / scale) < 1e-10


def test_gauge_loops_only_give_the_published_quartic():
    """lambda(M_X) = g^4/(16 pi^2), arXiv:2607.18233 app. B via Kierkla:2022odc.

    With gauge loops only, TL's condition gives lambda_CW = 11 g^4/(16 pi^2).
    """
    for g in (0.5, 0.7, 1.0):
        lam_cw = 11.0 * g**4 / LOOP
        v = 0.173
        got = dr.msbar_quartic_from_cw(g, 0.0, lam_cw, v, g * v, scalar_loops=False)
        assert got == pytest.approx(g**4 / LOOP, rel=1e-12)


@pytest.mark.parametrize("g,y", [(0.6, 0.01), (1.0, 0.3)])
def test_the_converted_potential_has_its_minimum_at_v(g, y):
    """Found by minimising, not by the identity the conversion is built on."""
    from scipy.optimize import minimize_scalar
    import TL_conformal_u1_fermions as m4d

    lam_cw = m4d.specific_potential({"g": g, "y": y, "v_GeV": 0.173}, verbose=False).l
    v = 0.173
    lam_bar = dr.msbar_quartic_from_cw(g, y, lam_cw, v, g * v)
    res = minimize_scalar(lambda p: _msbar_potential(p, g, y, lam_bar, lam_cw, g * v),
                          bounds=(0.3 * v, 3.0 * v), method="bounded",
                          options={"xatol": 1e-12 * v})
    assert res.x == pytest.approx(v, rel=1e-6)


def test_the_cw_quartic_is_an_order_of_magnitude_above_the_msbar_one():
    """At the input scale the two differ by a factor ~11."""
    import TL_conformal_u1_fermions as m4d

    g, y, v = 0.7, 0.01, 0.173
    lam_cw = m4d.specific_potential({"g": g, "y": y, "v_GeV": v}, verbose=False).l
    lam_bar = dr.msbar_quartic_from_cw(g, y, lam_cw, v, g * v)
    assert 9.0 < lam_cw / lam_bar < 12.0


def test_matched_parameters_run_from_the_converted_quartic():
    g, y, v, T = 0.7, 0.01, 0.173, 1.241e-3
    import TL_conformal_u1_fermions as m4d

    lam_cw = m4d.specific_potential({"g": g, "y": y, "v_GeV": v}, verbose=False).l
    mu = 7.0553 * T

    p = dr.matched_soft_parameters(g, y, lam_cw, v, T)
    lam0 = dr.msbar_quartic_from_cw(g, y, lam_cw, v, g * v)
    gr, yr, lr, _ = dr.rg_run(g, y, lam0, g * v, mu)
    assert p["mu0"] == g * v and p["lam0_msbar"] == lam0
    assert p["lam3"] == dr.lambda3_NLO(gr, yr, lr, T, mu)

    with pytest.raises(ValueError, match="input_scheme must be"):
        dr.matched_soft_parameters(g, y, lam_cw, v, T, input_scheme="bogus")


def test_the_model_uses_the_converted_quartic_on_request():
    import TL_conformal_u1_fermions_3d as m3d

    g, y, v, T = 0.7, 0.01, 0.173, 1.241e-3
    pot = m3d.specific_potential({"g": g, "y": y, "v_GeV": v}, verbose=False)
    assert pot.config.nucleationConf.input_scheme == "zero_quartic"
    pot.config.nucleationConf.input_scheme, pot.config.nucleationConf.mu3_choice = "CW", "g3sq"
    runtime = pot.matched_3d_parameters(T)
    offline = dr.matched_soft_parameters(g, y, pot.l, v, T)
    assert runtime == (offline["m3sq"], offline["lam3"], offline["g3sq"], offline["mD2"])
    pot.config.nucleationConf.input_scheme = "zero_quartic"
    assert pot.matched_3d_parameters(T)[1] != runtime[1]


def test_the_input_scale_is_a_small_higher_order_choice():
    """A factor 2 in mu0, with g and y run to it, moves lambda(mu4) by < 1% at g = 0.7.

    Pins the number quoted in INPUT_SCHEME_LIMITATIONS; the 3.4% at g = 1 is the
    largest on the line scan.
    """
    import TL_conformal_u1_fermions as m4d

    y, v = 0.01, 0.173
    for g, T, bound in ((0.7, 1.241e-3, 0.01), (1.0, 4.888e-3, 0.035)):
        mu4 = 7.0553 * T
        lam_cw = m4d.specific_potential({"g": g, "y": y, "v_GeV": v}, verbose=False).l
        base = dr.rg_run(g, y, dr.msbar_quartic_from_cw(g, y, lam_cw, v, g * v), g * v, mu4)[2]
        for f in (0.5, 2.0):
            mu0 = f * g * v
            g0, y0, _, _ = dr.rg_run(g, y, lam_cw, g * v, mu0)
            lam_cw0 = m4d.specific_potential({"g": g0, "y": y0, "v_GeV": v}, verbose=False).l
            lam0 = dr.msbar_quartic_from_cw(g0, y0, lam_cw0, v, mu0)
            moved = dr.rg_run(g0, y0, lam0, mu0, mu4)[2]
            assert abs(moved - base) / abs(base) < bound, f"g={g}, mu0 x{f}"
