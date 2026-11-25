"""The J = 0 gauge-Goldstone sector: its zero mode, and the treatment that removes it.

At xi = 1 the J = 0 block (photon at orbital 1, Goldstone at orbital 0) has the
normalisable zero mode (alpha', -g3 phi alpha), alpha the regular l = 0 ghost
solution -- the global U(1) rotation of the bounce with a pure-gauge photon tail.
Assembled naively the block made the exponent run away with the radial domain as
-3/2 ln R. abelian_higgs_3d.j0_sector integrates the rotation as a collective
coordinate instead; these tests check the zero mode is there, that the treated
value is what a direct removal at xi = 1 gives (gauge independence of the
sector), that it does not depend on the domain, and that nlo_assembly adds
exactly the reported correction.
"""
import math
import os
import sys

import numpy as np
import pytest
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import CubicSpline

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402
from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402

G, Y, V, T = 0.7, 0.01, 0.173, 6.73e-3


def _bounce():
    import TL_conformal_u1_fermions as m4d

    lam = m4d.specific_potential({"g": G, "y": Y, "v_GeV": V}, verbose=False).l
    p = dr.matched_soft_parameters(G, Y, lam, V, T)
    S, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    return p, S, r, phi3


def _dV(p):
    return lambda q: abelian_higgs_3d.dV3_LO(q, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])


def _continued(p, r, phi3, factor):
    """The profile continued by its analytic tail A e^{-m r}/r to `factor` x its domain."""
    j0 = abelian_higgs_3d.j0_sector(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    A, m = j0["tail_amplitude"], j0["false_vacuum_mass"]
    extra = np.linspace(r[-1], factor * r[-1], int(200 * (factor - 1)) + 1)[1:]
    r2 = np.concatenate([r, extra])
    phi2 = np.concatenate([phi3, A * np.exp(-m * extra) / extra])
    return r2, phi2, A, m


@pytest.fixture(scope="module")
def bounce():
    return _bounce()


def _functions(p, r, phi3, A, m):
    core = CubicSpline(r, phi3)
    dcore = core.derivative(1)
    r_join = float(r[np.argmin(np.abs(phi3 - 1e-4 * phi3[0]))])
    phi = lambda x: float(core(x)) if x <= r_join else A * math.exp(-m * x) / x
    dphi = lambda x: float(dcore(x)) if x <= r_join else -A * math.exp(-m * x) * (m / x + 1 / x**2)
    return phi, dphi


def _j0_regular_solutions(p, phi, dphi, dV, m, r0, R):
    g3sq, g3 = p["g3sq"], math.sqrt(p["g3sq"])

    def WG(x):
        q = phi(x)
        return float(dV(q)) / q if q > 1e-300 else m * m

    def rhs(x, y):
        a, da, Gf, dG = y
        w = g3sq * phi(x) ** 2
        mp = g3 * dphi(x)
        return [da, -2 / x * da + 2 / x**2 * a + w * a - 2 * mp * Gf,
                dG, -2 / x * dG + (WG(x) + w) * Gf - 2 * mp * a]

    out = []
    for y0 in ([r0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]):
        s = solve_ivp(rhs, (r0, R), y0, method="DOP853", rtol=1e-11, atol=1e-20)
        a, _, Gf, _ = s.y[:, -1]
        x = s.t[-1]
        out.append((a / x, Gf * x * math.exp(-m * x)))            # (p, s): a ~ p R, G ~ s e^{mR}/R
    return out


def test_the_regular_solutions_grow_collinearly_which_is_the_zero_mode(bounce):
    """det of the regular solutions loses its leading growth only if a zero mode exists."""
    p, _S, r, phi3 = bounce
    _r2, _p2, A, m = _continued(p, r, phi3, 1.0 + 1e-9)
    phi, dphi = _functions(p, r, phi3, A, m)
    (p1, s1), (p2, s2) = _j0_regular_solutions(p, phi, dphi, _dV(p), m, 1e-6 * r[-1], 25.0 / m)
    assert p1 / s1 == pytest.approx(p2 / s2, rel=1e-4)


def test_the_treated_sector_is_what_a_removal_at_xi_one_gives(bounce):
    """Gauge independence of the J = 0 sector: xi = 1 with the zero mode removed = Fermi gauge."""
    p, _S, r, phi3 = bounce
    j0 = abelian_higgs_3d.j0_sector(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    A, m = j0["tail_amplitude"], j0["false_vacuum_mass"]
    phi, dphi = _functions(p, r, phi3, A, m)
    g3sq, g3, phi0 = p["g3sq"], math.sqrt(p["g3sq"]), float(phi3[0])
    r0, R_end = 1e-6 * r[-1], 40.0 / m

    gh = solve_ivp(lambda x, y: [y[1], g3sq * phi(x) ** 2 * y[0] - 2 * y[1] / x], (r0, R_end),
                   [1.0, g3sq * phi0**2 * r0 / 3.0], method="DOP853", rtol=1e-11, atol=1e-14,
                   dense_output=True)
    rr = np.geomspace(r0, R_end, 20000)
    alpha, dalpha = gh.sol(rr)
    b = float(-(rr[-1] ** 2) * dalpha[-1])
    a_inf = float(alpha[-1] - b / rr[-1])
    phis = np.array([phi(x) for x in rr])
    norm = 4 * math.pi * simpson(rr**2 * (dalpha**2 + g3sq * phis**2 * alpha**2), x=rr)
    norm += 4 * math.pi * b**2 / rr[-1]                          # the r^-2 photon tail beyond R_end
    overlap = 4 * math.pi * simpson(rr**2 * g3 * phis**2 * alpha, x=rr)
    (_p1, _s1), (_p2, s2) = _j0_regular_solutions(p, phi, dphi, _dV(p), m, r0, 25.0 / m)
    det_prime = -m * s2 * norm / (6 * math.pi * b * (g3sq * phi0**2 / 3))
    S_xi1 = -math.log(math.sqrt(2 * math.pi) * abs(overlap) / math.sqrt(norm)
                      / math.sqrt(abs(det_prime)) * a_inf)
    assert S_xi1 == pytest.approx(j0["treated"], abs=2e-3)


def test_naive_sector_runs_away_with_the_domain_and_the_treated_one_does_not(bounce):
    p, _S, r, phi3 = bounce
    kw = dict(m_debye_sq=p["mD2"])
    one = abelian_higgs_3d.j0_sector(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], **kw)
    r2, phi2, _A, _m = _continued(p, r, phi3, 2.0)
    two = abelian_higgs_3d.j0_sector(r2, phi2, p["m3sq"], p["lam3"], p["g3sq"], **kw)
    # -3/2 ln 2 = -1.04 from the zero mode, plus slower 1/R pieces of the massless channels
    assert two["naive"] - one["naive"] < -0.8
    assert two["treated"] == pytest.approx(one["treated"], abs=1e-3)


def test_nlo_assembly_adds_exactly_the_reported_correction(bounce, monkeypatch):
    pytest.importorskip("BubbleDet")
    p, S, r, phi3 = bounce
    args = (r, phi3, p["m3sq"], p["lam3"], p["g3sq"], p["mD2"], S, -50.0)
    new = abelian_higgs_3d.nlo_assembly(*args, T=T)
    monkeypatch.setattr(abelian_higgs_3d, "j0_sector", lambda *a, **k: dict(new["j0_sector"], correction=0.0))
    old = abelian_higgs_3d.nlo_assembly(*args, T=T)
    assert old["j0_correction"] == 0.0 and new["j0_correction"] != 0.0
    assert (new["total_without_NLO_potential"] - old["total_without_NLO_potential"]
            == pytest.approx(new["j0_correction"], abs=1e-9))
    assert new["j0_sector"]["tail_amplitude_spread"] < 0.02
