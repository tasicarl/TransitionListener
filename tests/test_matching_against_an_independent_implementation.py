"""The matched 3D parameters of both conformal U(1) models against an independent implementation of the same NLO
matching relations, at g = 0.7, T = 0.1 GeV, mu4 = 2 pi T, mu3 = g T, with lambda = 0 at mu0 = 1 GeV.

lambda3, g3^2 and m_D^2 agree to rounding (lambda3 to the tolerance of the running from mu0 to mu4). In the model with
fermions the fermionic matching logarithm L_f enters all three, so the agreement fixes its offset from the bosonic
one, L_f - L_b = 4 ln 2, which the slopes tested in test_matching_logs.py leave open. The scalar mass m3^2 agrees to
the order of the matching: the two implementations evaluate the coefficient of its ln(mu3/mubar) term with couplings
of different order, a difference beyond NLO.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as fermions  # noqa: E402
import dr_conformal_u1_no_fermions_matching as no_fermions  # noqa: E402

G, T, MU0 = 0.7, 0.1, 1.0
# m3sq, lam3, g3sq, mD2 of the independent implementation (GeV units)
REFERENCE = {
    0.0: (0.001109651910817514, -1.6318876008731952e-05, 0.04896466950467622, 0.0016437820798763571),
    0.5: (0.0013191651019204204, 0.0001248442788507061, 0.04878940358966556, 0.0020445265362836343),
}
KEYS = ("m3sq", "lam3", "g3sq", "mD2")
RTOL = {"m3sq": 1e-3, "lam3": 1e-8, "g3sq": 1e-12, "mD2": 1e-12}


def matched(y):
    kw = dict(mu4_over_T=2 * math.pi, matching_order="NLO", mu3=G * T)
    if y == 0.0:
        return no_fermions.matched_soft_parameters(G, 0.0, MU0, T, **kw)
    return fermions.matched_soft_parameters_zero_quartic(G, y, MU0, T, **kw)


@pytest.mark.parametrize("y", sorted(REFERENCE))
@pytest.mark.parametrize("key", KEYS)
def test_the_matched_parameter_agrees_with_the_independent_implementation(y, key):
    ref = REFERENCE[y][KEYS.index(key)]
    assert matched(y)[key] == pytest.approx(ref, rel=RTOL[key], abs=0.0)


def test_the_comparison_sees_the_offset_of_the_fermionic_logarithm(monkeypatch):
    """With L_f = L_b, the offset the comparison is meant to fix, g3^2 of the model with fermions misses its reference."""
    monkeypatch.setattr(fermions, "Lf", fermions.Lb)
    g3sq = matched(0.5)["g3sq"]
    assert abs(g3sq / REFERENCE[0.5][2] - 1.0) > 1e-3
