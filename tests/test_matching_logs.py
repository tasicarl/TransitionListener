"""The matching logarithms Lb, Lf: slope verified, constant deliberately not.

The 3D parameters of the dimensionally reduced theory are physical quantities of
the effective theory. They must not depend on the 4D renormalisation scale
mubar: the explicit Lb, Lf inside them has to cancel the running of the 4D
couplings they are built from, order by order.

That requirement fixes d Lb/d ln(mubar) = d Lf/d ln(mubar) = 2 and nothing else.
A common additive shift Lb -> Lb + c, Lf -> Lf + c is identically the rescaling
mubar -> mubar e^{c/2}, so no scale-independence argument can ever see it; it is
absorbed into whatever renormalisation-scale band we quote.

The two tests below use different parameter combinations (gauge only, versus
gauge + quartic + Yukawa), so they constrain the slope independently.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

T = 3.0
MU0 = 2.0 * math.pi * T
G0, Y0, LAM0 = 0.65, 0.50, 0.045


def _residual_scaling(quantity):
    """How fast `quantity(mu)` varies with mu, over two shrinking intervals.

    `quantity` takes (g, y, lam, mu). We run the couplings from MU0 to mu with
    the one-loop beta functions, so any leftover mubar dependence is a genuine
    higher-order effect and must shrink faster than the one-loop term it would
    otherwise contaminate. Returns (spread over mu in [MU0/2, 2 MU0],
    spread over mu in [MU0/1.1, 1.1 MU0]).
    """
    def spread(factor):
        vals = []
        for mu in (MU0 / factor, MU0, MU0 * factor):
            g, y, lam, _ = dr.rg_run(G0, Y0, LAM0, MU0, mu)
            vals.append(quantity(g, y, lam, mu))
        return max(vals) - min(vals)
    return spread(2.0), spread(1.1)


def test_g3_squared_is_scale_independent_and_that_fixes_the_slope_of_Lb_plus_Lf():
    """g3^2 = g^2 T - g^4 (Lb + Lf) T/(48 pi^2) must not depend on mubar.

    d/d ln mubar of the first term is beta(g^2) T = g^4 T/(12 pi^2). Cancelling
    it needs d(Lb + Lf)/d ln mubar = 4.
    """
    wide, narrow = _residual_scaling(
        lambda g, y, lam, mu: dr.g3_sq_NLO(g, T, mu))
    # residual is two-loop, O(g^6): relative to the one-loop term it drops
    assert narrow < wide
    assert wide / dr.g3_sq_LO(G0, T) < 2.0e-4

    # and the slope itself, read off directly
    d = 1e-6
    slope = (dr.Lb(MU0 * math.exp(d), T) + dr.Lf(MU0 * math.exp(d), T)
             - dr.Lb(MU0 / math.exp(d), T) - dr.Lf(MU0 / math.exp(d), T)) / (2 * d)
    assert slope == pytest.approx(4.0, rel=1e-6)


def test_lambda3_is_scale_independent_which_constrains_Lb_and_Lf_separately():
    """The same for lambda3, whose Lb and Lf appear with different coefficients.

    d lambda3/d ln mubar has an explicit part proportional to
    (-6 g^4 + 12 g^2 lam - 20 lam^2 + 2 y^4 - 4 y^2 lam) dL/d ln mubar / (16 pi^2)
    and an implicit part T beta_lambda from the 2 lam (8 pi^2) term. They cancel
    only for dL/d ln mubar = 2, and here y enters, so this is not the previous
    test again.
    """
    wide, narrow = _residual_scaling(
        lambda g, y, lam, mu: dr.lambda3_NLO(g, y, lam, T, mu))
    assert narrow < wide
    assert wide / abs(dr.lambda3_NLO(G0, Y0, LAM0, T, MU0)) < 1.0e-2


def test_a_wrong_slope_would_be_caught():
    """Guard: with d Lb/d ln mubar = 1 instead of 2, the test above would fail.

    Without this, the two tests could be passing because the residual is small
    for some unrelated reason.
    """
    real_Lb, real_Lf = dr.Lb, dr.Lf
    try:
        dr.Lb = lambda mu, TT: 1.0 * math.log(
            mu * math.exp(dr.EULER_GAMMA) / (4.0 * math.pi * TT))
        dr.Lf = lambda mu, TT: dr.Lb(mu, TT) + 4.0 * math.log(2.0)
        wide, _ = _residual_scaling(lambda g, y, lam, mu: dr.g3_sq_NLO(g, T, mu))
        assert wide / dr.g3_sq_LO(G0, T) > 2.0e-4
    finally:
        dr.Lb, dr.Lf = real_Lb, real_Lf
