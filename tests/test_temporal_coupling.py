"""The temporal mass m_D^2 + h3 phi^2 / 2 of the 3D rate (nucleationConf.temporal_coupling).

h3 at its leading-order value 2 g3^2 is the leading-order rate; the NLO-matched h3 enters the leading-order potential,
its derivatives, the bounce, the temporal determinant and the double-counting subtraction alike. Against an independent
implementation of the bounce with the NLO h3: S_LO = 225.1116 at g = 0.7, y = 0 and 331.2510 at y = 0.5, T = 0.1 GeV,
mu4 = 2 pi T, from the 3D parameters below.
"""
import math

import numpy as np
import pytest

from transitionlistener.eft_rate import abelian_higgs_3d

# m3sq, lam3, g3sq, mD2, h3 in GeV units at the two reference points, and the independent S_LO

from transitionlistener.eft_rate import nlo_tables

# m3sq, lam3, g3sq, mD2, h3 in GeV units at the two reference points, and the independent S_LO
REFERENCE = [((0.001109651910817514, -1.6318876008731952e-05, 0.04896466950467622, 0.0016437820798763571,
           0.09831282260233748), 225.1116),
         ((0.0013191651019204204, 0.0001248442788507061, 0.04878940358966556, 0.0020445265362836343,
           0.09790933630431217), 331.2510)]
PHI = np.linspace(0.0, 0.6, 61)


@pytest.mark.parametrize("args,S_ref", REFERENCE)
def test_the_bounce_with_h3_matches_an_independent_implementation(args, S_ref):
    m3sq, lam3, g3sq, mD2, h3 = args
    S = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2, h3=h3)[0]
    assert S == pytest.approx(S_ref, abs=2e-4)
    # and h3 is what moves it: the leading-order value is about one unit away
    assert abs(abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)[0] - S_ref) > 0.5


def test_the_leading_order_value_is_the_rate_as_it_was():
    m3sq, lam3, g3sq, mD2, _ = REFERENCE[0][0]
    for f in (abelian_higgs_3d.V3_LO, abelian_higgs_3d.dV3_LO, abelian_higgs_3d.d2V3_LO):
        assert np.allclose(f(PHI, m3sq, lam3, g3sq, m_debye_sq=mD2, h3=2 * g3sq),
                           f(PHI, m3sq, lam3, g3sq, m_debye_sq=mD2), rtol=1e-14, atol=0.0)
    a = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2)
    b = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2, h3=2 * g3sq)
    assert a[0] == pytest.approx(b[0], rel=1e-13)
    assert abelian_higgs_3d.double_counting_subtraction(a[1], a[2], g3sq, m_debye_sq=mD2) == pytest.approx(
        abelian_higgs_3d.double_counting_subtraction(a[1], a[2], g3sq, m_debye_sq=mD2, h3=2 * g3sq), rel=1e-13)


@pytest.mark.parametrize("ratio", [0.9, 1.0, 1.3])
def test_the_derivatives_are_those_of_the_potential(ratio):
    m3sq, lam3, g3sq, mD2, _ = REFERENCE[1][0]
    h3 = 2 * g3sq * ratio
    kw = dict(m_debye_sq=mD2, h3=h3)
    x = PHI[1:-1]
    d = 1e-6
    V = lambda p: abelian_higgs_3d.V3_LO(p, m3sq, lam3, g3sq, **kw)
    dV = lambda p: abelian_higgs_3d.dV3_LO(p, m3sq, lam3, g3sq, **kw)
    assert np.allclose(dV(x), (V(x + d) - V(x - d)) / (2 * d), rtol=1e-6, atol=1e-12)
    assert np.allclose(abelian_higgs_3d.d2V3_LO(x, m3sq, lam3, g3sq, **kw), (dV(x + d) - dV(x - d)) / (2 * d),
                       rtol=1e-6, atol=1e-10)


def test_the_temporal_terms_carry_h3():
    """The temporal piece of V and of the subtraction density is -(m0^3 - m_D^3)/(12 pi), m0^2 = m_D^2 + h3 phi^2/2."""
    m3sq, lam3, g3sq, mD2, h3 = REFERENCE[0][0]
    with_t = abelian_higgs_3d.V3_LO(PHI, m3sq, lam3, g3sq, m_debye_sq=mD2, h3=h3)
    without = abelian_higgs_3d.V3_LO(PHI, m3sq, lam3, g3sq, m_debye_sq=mD2, n_temporal=0)
    m0_sq = mD2 + 0.5 * h3 * PHI ** 2
    assert np.allclose(with_t - without, -(m0_sq ** 1.5 - mD2 ** 1.5) / (12 * math.pi), rtol=1e-12, atol=1e-16)
    r = np.linspace(0.0, 50.0, 2001)
    phi = 0.3 * np.exp(-(r / 10.0) ** 2)
    density = 2 * (g3sq * phi ** 2) ** 1.5 + (mD2 + 0.5 * h3 * phi ** 2) ** 1.5 - mD2 ** 1.5
    assert abelian_higgs_3d.double_counting_subtraction(r, phi, g3sq, m_debye_sq=mD2, h3=h3) == pytest.approx(
        -np.trapz(4 * math.pi * r ** 2 * density, r) / (12 * math.pi), rel=1e-12)


class _Conf:
    def __init__(self, choice):
        self.temporal_coupling = choice


class _Model:
    def __init__(self, choice):
        self.config = type("C", (), {"nucleationConf": _Conf(choice)})()


def test_the_switch_reads_the_matched_coupling():
    p = {"lambda_vl": 0.0983}
    assert nlo_tables.temporal_coupling(_Model("h3"), p) == 0.0983
    assert nlo_tables.temporal_coupling(_Model("2g3sq"), p) is None
    with pytest.raises(ValueError, match="lambda_vl"):
        nlo_tables.temporal_coupling(_Model("h3"), {})
    with pytest.raises(ValueError, match="'h3' or '2g3sq'"):
        nlo_tables.temporal_coupling(_Model("NLO"), p)


class _Grid:
    factors = [1.0]

    def __init__(self, header):
        self.header = header

    def describe(self):
        return "grid"


@pytest.mark.parametrize("recorded,run,refused", [(None, "h3", True), (None, "2g3sq", False), ("h3", "h3", False),
                                                  ("h3", "2g3sq", True)])
def test_a_grid_of_the_other_temporal_coupling_is_refused(recorded, run, refused):
    """A grid that does not record the temporal coupling was built with 2 g3^2."""
    header = dict(mu3_choice="gT", input_scheme="zero_quartic", matching_order="NLO", mu4_over_T=7.0553,
                  g3sq_order="NLO")
    if recorded is not None:
        header["temporal_coupling"] = recorded
    conf = type("C", (), dict(mu3_choice="gT", input_scheme="zero_quartic", matching_order="NLO", mu4_over_T=7.0553,
                              temporal_coupling=run))()
    if refused:
        with pytest.raises(nlo_tables.NLOGridError, match="temporal_coupling"):
            nlo_tables._check_conventions(_Grid(header), conf)
    else:
        assert nlo_tables._check_conventions(_Grid(header), conf) == 1.0


def test_the_assembly_with_h3_is_the_assembly_with_every_temporal_piece_replaced():
    """nlo_assembly(h3=...) against the same assembly with the potential, the temporal mass and the subtraction
    handed over explicitly."""
    pytest.importorskip("BubbleDet")
    m3sq, lam3, g3sq, mD2, h3 = REFERENCE[0][0]
    S, r, phi, _ = abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=mD2, h3=h3)
    kw = dict(m_debye_sq=mD2, h3=h3)
    a = abelian_higgs_3d.nlo_assembly(r, phi, m3sq, lam3, g3sq, mD2, S, 0.0, T=0.1, h3=h3)
    b = abelian_higgs_3d.nlo_assembly(
        r, phi, m3sq, lam3, g3sq, mD2, S, 0.0, T=0.1,
        potential=(lambda p: abelian_higgs_3d.V3_LO(p, m3sq, lam3, g3sq, **kw), lambda p: abelian_higgs_3d.dV3_LO(p, m3sq, lam3, g3sq, **kw),
                   lambda p: abelian_higgs_3d.d2V3_LO(p, m3sq, lam3, g3sq, **kw)),
        temporal_mass_sq=lambda q: mD2 + 0.5 * h3 * q ** 2,
        subtraction=abelian_higgs_3d.double_counting_subtraction(r, phi, g3sq, **kw))
    key = "total_without_NLO_potential"
    assert a[key] == pytest.approx(b[key], rel=1e-12)
    c = abelian_higgs_3d.nlo_assembly(r, phi, m3sq, lam3, g3sq, mD2, S, 0.0, T=0.1)
    assert abs(c[key] - a[key]) > 1e-3
