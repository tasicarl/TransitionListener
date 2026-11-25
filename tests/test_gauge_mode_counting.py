"""How many photon states there are per partial wave, and what that implies.

At xi = 1 the three spatial photon components, with the Goldstone mixing
switched off, are three copies of -nabla^2 + g3^2 phi3^2. In vector harmonics
the same states are the electric ones (total J = l -/+ 1, including J = 0,
which holds only the orbital-1 radial mode) and the magnetic ones (J = l >= 1).
Per orbital l both carry 3(2l+1) states, so the determinant cannot depend on
the decomposition. The offset 3 ln T0 - ln T1 belongs to an identity that starts
the electric sum at J = 1 and keeps l = 0 in the magnetic channel, and adding it
to the determinant counts it twice. The Goldstone mixes with the J = 0 radial
mode too, and the mixing sum includes it.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402

G, Y, V, T = 0.7, 0.01, 0.173, 6.73e-3


def _bounce():
    import TL_conformal_u1_fermions as m4d
    from transitionlistener.eft_rate import abelian_higgs_3d

    lam = m4d.specific_potential({"g": G, "y": Y, "v_GeV": V}, verbose=False).l
    p = dr.matched_soft_parameters(G, Y, lam, V, T)
    S, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    return p, S, r, phi3


def test_state_counting_on_arbitrary_sequences():
    """sum_J electric + sum_J magnetic = 3 sum_l (2l+1) a_l up to a 1/L boundary."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        L = 400
        a = rng.normal(size=L + 2) / (2 * np.arange(L + 2) + 1)   # falls like a log T_l
        electric = a[1] + sum((2 * J + 1) * (a[J - 1] + a[J + 1]) for J in range(1, L + 1))
        magnetic = sum((2 * J + 1) * a[J] for J in range(1, L + 1))
        scalars = 3 * sum((2 * k + 1) * a[k] for k in range(0, L + 1))
        boundary = (2 * L + 1) * a[L + 1] - (2 * L + 3) * a[L]
        assert electric + magnetic - scalars == pytest.approx(boundary, abs=1e-9)


def test_the_real_gauge_channel_has_no_offset():
    from transitionlistener.eft_rate import abelian_determinants as fl

    p, _S, r, phi3 = _bounce()
    W = lambda rr: p["g3sq"] * float(np.interp(rr, r, phi3)) ** 2
    rmax = float(r[-1])
    a = [fl.log_gy_single(W, k, rmax, 0.0) for k in range(0, 42)]

    def mismatch(L):
        electric = a[1] + sum((2 * J + 1) * (a[J - 1] + a[J + 1]) for J in range(1, L + 1))
        magnetic = sum((2 * J + 1) * a[J] for J in range(1, L + 1))
        return electric + magnetic - 3 * sum((2 * k + 1) * a[k] for k in range(0, L + 1))

    # the physical counting converges to three single channels, like 1/L
    assert abs(mismatch(40)) < 0.6 * abs(mismatch(20))
    assert abs(mismatch(40)) < 4.0
    # while the offset is not small: adding it to the assembly would show
    assert 3 * a[0] - a[1] > 10.0


def test_the_J0_mixing_term_is_present_and_continues_the_sequence():
    from transitionlistener.eft_rate import abelian_higgs_3d
    from transitionlistener.eft_rate import abelian_determinants as fl

    p, _S, r, phi3 = _bounce()
    kw = dict(m_debye_sq=p["mD2"])
    l0 = abelian_higgs_3d.mixing_l0_term(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], **kw)
    WX, mXp, dWG, zero, rmax, mG = abelian_higgs_3d._mixing_callables(
        r, phi3, p["m3sq"], p["lam3"], p["g3sq"], n_temporal=1, **kw)
    l1 = 3 * (fl.log_gy_mixed(WX, mXp, dWG, 1, rmax, mG) - fl.log_gy_mixed(WX, zero, dWG, 1, rmax, mG))
    assert l0 < 0.0 and l1 < 0.0
    assert 0.5 < l0 / l1 < 2.0            # same size as its neighbour, not an outlier
