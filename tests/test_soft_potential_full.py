"""The complete soft-theory potential with scalar loops, for re-solved bounces."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_fermions_matching as dr  # noqa: E402
from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402

M3, LAM3, G3, MD2, MU3 = 0.08, -0.02, 0.5, 0.2, 0.4
LVL, LVLL = 1.02, 0.03
PHI = np.linspace(0.0, 3.0, 31)


def test_one_loop_is_V3_LO_plus_the_scalar_loops():
    full = dr.V3_soft_one_loop_full(PHI, M3, LAM3, G3, MD2, 2 * G3)
    lo = abelian_higgs_3d.V3_LO(PHI, M3, LAM3, G3, m_debye_sq=MD2)
    cube = lambda x: np.where(x > 0, np.abs(x)**1.5, 0.0)
    scalars = -(cube(M3 + LAM3 * PHI**2) + cube(M3 + 3 * LAM3 * PHI**2) - 2 * M3**1.5) / (12 * math.pi)
    assert np.allclose(full - lo, scalars, rtol=1e-12, atol=1e-14)
    assert np.any(M3 + 3 * LAM3 * PHI**2 < 0)          # the tachyonic branch is exercised


def test_A0_diagrams_reduce_to_the_HET_terms_for_a_massless_scalar():
    het = dr.V3_soft_two_loop(PHI, 0.0, MD2, LVL, LVLL, MU3)
    a0 = dr.V3_soft_two_loop_A0(PHI, 0.0, 0.0, MD2, LVL, LVLL, MU3)
    assert np.allclose(a0, het, rtol=1e-12, atol=1e-15)


def test_the_A0_scalar_figure_eight_is_the_response_to_DRalgos_mass_shift():
    """Integrating out A_0 shifts the scalar mass by -lambda_VL m_D/(8 pi)
    (DRalgo, scalar_mass_sq_US); the one-loop scalar potential's first-order
    response to that shift, with m_D -> m_X0(phi), is the figure-eight."""
    m3, lam3 = 0.08, 0.01                            # positive masses: no branch question
    for phi in (0.3, 1.0, 2.0):
        m0sq = MD2 + 0.5 * LVL * phi**2
        delta = dr.scalar_mass_sq_US(m3, LVL, m0sq) - m3
        mh, mG = math.sqrt(m3 + 3 * lam3 * phi**2), math.sqrt(m3 + lam3 * phi**2)
        response = -1.5 * delta * (mh + mG) / (12 * math.pi)
        fig8 = LVL * math.sqrt(m0sq) * (mh + mG) / (64 * math.pi**2)
        assert fig8 == pytest.approx(response, rel=1e-12)


@pytest.mark.parametrize("lam3", [0.01, LAM3])
def test_the_two_loop_potential_satisfies_the_3D_renormalisation_group(lam3):
    """In 3D only m3^2 runs: d m3^2/d ln mu3 = (8 g3^4 - 16 g3^2 lam3 + 16 lam3^2
    + lambda_VL^2)/(32 pi^2), so d V2/d ln mu3 = -phi^2/2 times that -- also on
    the tachyonic branch, since the scale sits in the logarithms only."""
    phi = np.array([0.2, 0.9, 2.5])
    eps = 1e-4
    up = dr.V3_soft_two_loop_full(phi, M3, lam3, G3, MD2, LVL, LVLL, MU3 * math.exp(eps))
    dn = dr.V3_soft_two_loop_full(phi, M3, lam3, G3, MD2, LVL, LVLL, MU3 * math.exp(-eps))
    slope = (up - dn) / (2 * eps)
    beta = (8 * G3**2 - 16 * G3 * lam3 + 16 * lam3**2 + LVL**2) / (32 * math.pi**2)
    assert np.allclose(slope, -0.5 * phi**2 * beta, rtol=1e-6, atol=1e-12)


def test_the_real_part_branch_changes_nothing_where_masses_are_positive():
    phi = [0.1, 0.7, 1.5]
    a = dr.V3_US_two_loop(phi, 0.08, 0.01, G3, MU3)
    b = dr.V3_US_two_loop(phi, 0.08, 0.01, G3, MU3, tachyonic="real_part")
    assert np.allclose(a, b, rtol=1e-14, atol=0)
    with pytest.raises(ValueError):
        dr.V3_US_two_loop([2.5], M3, LAM3, G3, MU3)


def test_the_generic_bounce_solver_is_the_LO_one():
    S, r, phi, _ = abelian_higgs_3d.solve_LO_bounce(M3, -0.004, G3, m_debye_sq=MD2)
    V = lambda p: abelian_higgs_3d.V3_LO(p, M3, -0.004, G3, m_debye_sq=MD2)
    dV = lambda p: abelian_higgs_3d.dV3_LO(p, M3, -0.004, G3, m_debye_sq=MD2)
    esc_lo = abelian_higgs_3d.escape_point_LO(M3, -0.004, G3, m_debye_sq=MD2)
    assert abelian_higgs_3d.escape_point(V, 1.0, bounded=False) == pytest.approx(esc_lo, rel=1e-10)
    S2 = abelian_higgs_3d.solve_bounce_1d(V, dV, esc_lo, bounded=False)[0]
    assert S2 == pytest.approx(S, rel=1e-9)
