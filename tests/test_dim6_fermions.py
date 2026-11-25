"""The fermions' one-loop share of the dimension-six coefficients c6 and c_kin.

The closed forms in models/dr_conformal_u1_fermions_matching.py are pinned against the sympy derivation
in models/dr_conformal_u1_fermions_dim6_derivation.py, and the derivation is pinned against the bosonic
coefficients of arXiv:2610.12284 app. A, which the same code must reproduce.
"""
import math
import os
import sys

import pytest

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
)

import dr_conformal_u1_fermions_matching as dr


# ------------------------------------------------------------ closed forms
def test_fermionic_coefficients_closed_form():
    z = dr.ZETA3
    for g, y, lam, T in ((0.5, 0.65, 0.0, 1e-6), (0.7, 0.5, -0.01, 0.05), (0.9, 0.3, 0.046, 3.0)):
        assert dr.c6_fermionic(g, y, lam) == pytest.approx(
            -7 * z / (960 * math.pi**4) * (5 * y**6 - 10 * lam * y**4 + 10 * lam**2 * y**2
                                           - 10 * g**2 * lam * y**2 + 2 * g**4 * lam), rel=1e-12)
        assert dr.cD2phi4_fermionic(g, y, lam, T) == pytest.approx(
            -7 * z / (960 * math.pi**4 * T) * (3 * g**4 - 15 * g**2 * y**2 + 10 * y**4), rel=1e-12)


def test_fermionic_sextic_is_the_coleman_weinberg_term_at_zero_quartic():
    """At lam = 0 only the fermions' m^6 term is left: -(2/3) (y^2)^3 sum_F 1/P^6, times T^2."""
    g, y = 0.6, 0.4
    sum_F = 7 * dr.ZETA3 / (128 * math.pi**4)          # T^2 sum_{P fermionic} 1/P^6
    assert dr.c6_fermionic(g, y, 0.0) == pytest.approx(-2.0 / 3.0 * y**6 * sum_F, rel=1e-12)


def test_fermions_lower_the_kinetic_coefficient_at_y_zero():
    """The charged fermions' vacuum polarisation alone: -21 zeta3 g^4/(960 pi^4 T)."""
    assert dr.cD2phi4_fermionic(0.5, 0.0, 0.0, 1.0) == pytest.approx(
        -21 * dr.ZETA3 * 0.5**4 / (960 * math.pi**4), rel=1e-12)


# ------------------------------------------------------------ derivation
@pytest.fixture(scope="module")
def derivation():
    pytest.importorskip("sympy")
    import dr_conformal_u1_fermions_dim6_derivation as der
    return der


def _numeric(expr, der, g, y, lam):
    return float(expr.subs({der.GS: g, der.YS: y, der.LAMS: lam}))


def test_derivation_reproduces_the_published_bosonic_coefficients(derivation):
    """The check of the method: bosons, ghosts and xi-dependent Green's basis -> 2610.12284 app. A."""
    import sympy as sp
    der = derivation
    cb = der.green_coefficients(include_fermions=False)
    assert any(sp.sympify(v).has(der.XIS) for v in cb.values())   # off shell: gauge dependent
    c6, ckin = der.onshell(cb)
    assert sp.expand(c6 - der.PUBLISHED_C6) == 0                   # on shell: xi drops out
    assert sp.expand(ckin - der.PUBLISHED_CKIN) == 0
    for g, lam in ((0.45, 0.0), (0.9, -0.02)):
        assert _numeric(c6, der, g, 0, lam) * dr.ZETA3 / math.pi**4 == pytest.approx(
            dr.c6_bosonic(g, lam), rel=1e-12)
        assert _numeric(ckin, der, g, 0, lam) * dr.ZETA3 / math.pi**4 == pytest.approx(
            dr.cD2phi4_bosonic(g, lam, 1.0), rel=1e-12)


def test_derivation_gives_the_fermionic_closed_forms(derivation):
    der = derivation
    cf = der.green_coefficients(include_bosons=False)
    # Green's basis, cross-checked against DRalgo's run with fermions (Zf = 7 Zb = 7 zeta3/(128 pi^4 T^2)):
    # alpha[phi^6] -(2/3)|y|^6 Zf, alpha[D^2 phi^4, 1] -(4/3)|y|^4 Zf, alpha[D^2 phi^4, 2] (1/3)|y|^4 Zf,
    # alpha[D^4 phi^2] -(1/3)|y|^2 Zf, alpha[D^2 F^2] -(1/15) g^2 Zf
    import sympy as sp
    Zf = sp.Rational(7, 128)
    y, g = der.YS, der.GS
    assert sp.expand(cf["c1"] + sp.Rational(2, 3) * y**6 * Zf) == 0
    assert sp.expand(cf["c2"] + sp.Rational(4, 3) * y**4 * Zf) == 0
    assert sp.expand(cf["c4"] - sp.Rational(1, 3) * y**4 * Zf) == 0
    assert sp.expand(cf["c6"] + sp.Rational(1, 3) * y**2 * Zf) == 0
    assert sp.expand(cf["c7"] + sp.Rational(1, 15) * g**2 * Zf) == 0
    c6, ckin = der.onshell(cf)
    for g_, y_, lam in ((0.5, 0.65, 0.0), (0.7, 0.5, -0.01), (0.9, 0.3, 0.046)):
        assert _numeric(c6, der, g_, y_, lam) * dr.ZETA3 / math.pi**4 == pytest.approx(
            dr.c6_fermionic(g_, y_, lam), rel=1e-12)
        assert _numeric(ckin, der, g_, y_, lam) * dr.ZETA3 / math.pi**4 == pytest.approx(
            dr.cD2phi4_fermionic(g_, y_, lam, 1.0), rel=1e-12)


def test_gauge_invariance_fixes_the_sign_of_the_fermion_charge(derivation):
    """With the charges of the two Weyl fermions flipped relative to their Yukawas the hard
    effective action is not background gauge invariant: the Phi Phi^* A three-point function at
    O(k^3) then has no solution in |D^2 Phi|^2 and d_i F_ij J_j."""
    import sympy as sp
    der = derivation
    c6, c8 = sp.symbols("c6 c8")

    def solve(sign):
        def loop(bg, N):
            return (der.fermion_loop(bg, N, sp.Rational(-sign, 2), conj_phi=False)
                    + der.fermion_loop(bg, N, sp.Rational(sign, 2), conj_phi=True))
        eqs = []
        for bg, N in der._configs("2pt") + der._configs("3pt"):
            ops = der._ops(bg)
            eqs.append(sp.expand(loop(bg, N) - c6 * ops["O6"] - c8 * ops["O8"]))
        return sp.solve(eqs, [c6, c8], dict=True)

    assert solve(+1) == [{c6: -7 * der.YS**2 / 384, c8: -7 * der.GS * der.YS**2 / 384}]
    assert solve(-1) == []
