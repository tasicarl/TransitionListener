"""Where the 4D and 3D potentials must agree.

TransitionListener's 4D potential and the 3D leading-order potential mapped
back, T V3(phi/sqrt T), are the same one-loop physics where g phi << T. Fitting
their difference D/T^4 in x = phi/T, the x^4 coefficient isolates the quartic:

* at T = g v / 7.0553, where the matching scale equals the input scale, it must
  vanish -- and the Coleman-Weinberg quartic read as MS-bar leaves about
  -2 g^4/16pi^2 there;
* below it, the 3D quartic is RG-resummed and the 4D one is not, so the
  mismatch is the two-loop leading log: with beta_lambda and beta_g2 of this
  model, lambda_3D - lambda_4D = -(7/8) g^6 t^2/(8 pi^4), t = ln(mu0/mu4), i.e.
  c4 = (7/32)(2/pi^2) g^2 t^2 = 0.0443 g^2 t^2 in units of g^4/16pi^2, plus
  single logs that make c4/(g^2 t^2) approach 0.0443 from below as t grows.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

Y, V = 0.01, 0.173


def _V4_from_V3(phi, T, m3sq, lam3, g3sq, **kw):
    """The potential of V3_LO in 4D normalisation, T V3(phi / sqrt T): its bounce action over T is the 3D action."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    return T * abelian_higgs_3d.V3_LO(np.asarray(phi, dtype=float) / np.sqrt(T), m3sq, lam3, g3sq, **kw)


def _c4(g, T, scheme):
    import TL_conformal_u1_fermions as m4d
    from transitionlistener.eft_rate import abelian_higgs_3d

    pot = m4d.specific_potential({"g": g, "y": Y, "v_GeV": V}, verbose=False)
    cf = pot.conversionFactor
    p = dr.matched_soft_parameters(g, Y, pot.l, V, T, matching_order="LO", input_scheme=scheme)
    x = np.linspace(0.02, 0.5 / g, 60)
    phi = x * T
    v4 = (pot.Vtot((phi / cf)[:, None], T / cf, include_radiation=False)
          - pot.Vtot(np.zeros((1, 1)), T / cf, include_radiation=False)[0]) * cf**4
    kw = dict(m_debye_sq=p["mD2"])
    v3 = (_V4_from_V3(phi, T, p["m3sq"], p["lam3"], p["g3sq"], **kw)
          - _V4_from_V3(0.0, T, p["m3sq"], p["lam3"], p["g3sq"], **kw))
    A = np.vstack([x**2, x**3, x**4, x**6]).T
    c, *_ = np.linalg.lstsq(A, (v4 - v3) / T**4, rcond=None)
    return c[2] / (g**4 / (16 * math.pi**2))


@pytest.mark.parametrize("g", [0.5, 0.7])
def test_no_quartic_mismatch_where_nothing_runs(g):
    T = g * V / 7.0553
    assert abs(_c4(g, T, "CW")) < 0.05


def test_below_it_the_mismatch_is_the_two_loop_leading_log():
    g = 0.6
    leading = (7.0 / 32.0) * (2.0 / math.pi**2)          # 0.0443
    ratios = []
    for f in (25.0, 125.0):
        t = math.log(f)
        ratios.append(_c4(g, g * V / 7.0553 / f, "CW") / (g**2 * t**2))
    assert 0.0 < ratios[0] < ratios[1] < 1.05 * leading
    assert ratios[1] > 0.75 * leading
