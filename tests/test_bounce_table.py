"""transitionlistener.eft_rate.abelian_lo_bounce_table: one table of the leading-order S_3/T for a class of potentials.

The scaling law and the edges are tested against abelian_higgs_3d.solve_LO_bounce, the table against direct solves, the seam on
both conformal models.
"""
import json
import math
import os
import sys

import numpy as np
import pytest

from transitionlistener.eft_rate import abelian_lo_bounce_table

from transitionlistener.eft_rate import one_field_rate_3d
from transitionlistener.compiled import bounce_table as kernels

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, "models")
if MODELS not in sys.path:
    sys.path.insert(0, MODELS)


@pytest.fixture
def table():
    abelian_lo_bounce_table.clear()
    abelian_lo_bounce_table.load()
    yield abelian_lo_bounce_table
    abelian_lo_bounce_table.clear()


def _bounce(*a, **k):
    from transitionlistener.eft_rate import abelian_higgs_3d
    return abelian_higgs_3d.solve_LO_bounce(*a, **k)[0]


def _model(name, params):
    from transitionlistener.helper_functions import load_potential
    pot = load_potential(os.path.join(MODELS, name), "specific_potential")(dict(params), verbose=False)
    c = pot.config.nucleationConf
    c.rate_mode, c.mu4_over_T, c.input_scheme, c.mu3_choice, c.matching_order = (
        "model", 2 * math.pi, "zero_quartic", "gT", "NLO")
    return pot


# A, B on the three branches: runaway quartic, positive quartic, close to the edge of the barrier
CANONICAL = [(0.3, -0.05), (0.15, 0.02), (1.0, -0.5), (0.5, 0.012)]


@pytest.mark.parametrize("A,B", CANONICAL)
@pytest.mark.parametrize("n_t,n_0", [(2, 1), (6, 3), (10, 5)])
@pytest.mark.parametrize("M2,mD2", [(0.1, 0.3), (2.5, 0.04), (0.37, 2.2)])
def test_scaling_law_of_the_direct_solve(table, A, B, n_t, n_0, M2, mD2):
    """S_3/T = kappa^(-1/2) (mD/M2)^(3/2) F(A, B): the bounce of a potential from the bounce of the canonical one."""
    kappa = n_t / 2
    mD = math.sqrt(mD2)
    m3sq, lam3 = A * kappa * M2 * mD, B * kappa * M2 ** 2 / mD
    assert table.coordinates(m3sq, lam3, M2, mD2, n_t, n_0)[:2] == pytest.approx((A, B), rel=1e-14)
    F = _bounce(A, B, 1.0, m_debye_sq=1.0)
    prefactor = table.coordinates(m3sq, lam3, M2, mD2, n_t, n_0)[2]
    assert prefactor == pytest.approx(kappa ** -0.5 * (mD / M2) ** 1.5, rel=1e-14)
    S = _bounce(m3sq, lam3, M2, n_t, m_debye_sq=mD2, n_temporal=n_0)
    assert S == pytest.approx(prefactor * F, rel=2e-7)


def test_su2_doublet_through_the_table(table):
    """kappa = 3 with the mass coupling g^2/4 in place of g3sq, against the direct solve."""
    g2, m3sq, lam3, mD2 = 0.04, 3e-4, -2e-4, 2e-3
    M2 = g2 / 4
    S = table.action_over_T(m3sq, lam3, M2, mD2, 6, 3)
    assert table.stats["table"] == 1
    assert S == pytest.approx(_bounce(m3sq, lam3, M2, 6, m_debye_sq=mD2, n_temporal=3), rel=1e-6)


@pytest.mark.parametrize("n_t,n_0", [(2, 0), (6, 1), (2, 2), (0, 0), (-2, -1), (3, 1)])
@pytest.mark.parametrize("mD2", [1.0, 0.0])
def test_refuses_other_multiplicities(table, n_t, n_0, mD2):
    with pytest.raises(ValueError, match="2 : 1"):
        table.action_over_T(0.3, -0.05, 1.0, mD2, n_t, n_0)


@pytest.mark.parametrize("c", [0.95, 1.0, 1.05])
def test_edge_is_the_double_root(table, c):
    """At (A, B_E(A, c)) the potential touches zero: its minimum over the field is zero; below it the barrier ends."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    A_S = c * abelian_lo_bounce_table.A_S
    f = np.geomspace(1e-3, 1e5, 400001)
    for a in (1e-3, 0.01, 0.1, 0.5, 1.0, 5.0, 100.0):
        A = A_S + a
        B = float(kernels.edge_B(A, c))
        V = abelian_higgs_3d.V3_LO(f, A, B, 1.0, m_debye_sq=1.0, h3=2 * c)
        assert abs(np.min(V / f ** 2)) < 2e-6 * A
        assert np.min(abelian_higgs_3d.V3_LO(f, A, B * (1 - 1e-3), 1.0, m_debye_sq=1.0, h3=2 * c)) < 0.0
        assert np.min(abelian_higgs_3d.V3_LO(f, A, B * (1 + 1e-3), 1.0, m_debye_sq=1.0, h3=2 * c)[1:]) > 0.0


@pytest.mark.parametrize("A,B", [(0.5, 0.03), (2.0, 0.02), (0.09, 0.8)])
def test_no_bounce_beyond_the_edge_without_a_solve(table, monkeypatch, A, B):
    from transitionlistener.eft_rate import abelian_higgs_3d
    with pytest.raises(ValueError):
        _bounce(A, B, 1.0, m_debye_sq=1.0)
    monkeypatch.setattr(abelian_higgs_3d, "solve_LO_bounce", lambda *a, **k: pytest.fail("solved"))
    assert table.action_over_T(A, B, 1.0, 1.0) is None
    assert table.stats["none"] == 1


@pytest.mark.parametrize("B", [-0.5, 0.0, 0.5])
def test_tachyonic_false_vacuum_has_no_bounce(table, monkeypatch, B):
    from transitionlistener.eft_rate import abelian_higgs_3d
    A = 0.9 / (4 * np.pi)
    with pytest.raises(ValueError):
        _bounce(A, B, 1.0, m_debye_sq=1.0)
    monkeypatch.setattr(abelian_higgs_3d, "solve_LO_bounce", lambda *a, **k: pytest.fail("solved"))
    assert table.action_over_T(A, B, 1.0, 1.0) is None


@pytest.mark.parametrize("A,B", [(300.0, -2.0), (0.5, -2e5), (1 / (4 * np.pi) + 1e-4, -1.0)])
def test_outside_the_grid_solves_the_bounce(table, A, B):
    """Beyond the grid on the far side of s, tau and of the cold edge: the direct number, to the last digit."""
    assert table.action_over_T(A, B, 1.0, 1.0) == _bounce(A, B, 1.0, m_debye_sq=1.0)
    assert table.stats == {"table": 0, "none": 0, "direct": 1}


def test_near_the_divergence_outside_the_grid(table):
    """tau above the grid and still a bounce or not: what the direct path says, a number or None."""
    A = 0.5
    B = float(kernels.edge_B(A)) * (1 - math.exp(-6.8))
    try:
        direct = _bounce(A, B, 1.0, m_debye_sq=1.0)
    except ValueError:
        direct = None
    assert table.action_over_T(A, B, 1.0, 1.0) == direct
    assert table.stats["direct"] == 1


def test_debye_mass_not_positive_is_solved_directly(table):
    assert table.action_over_T(1.0, -0.1, 1.0, 0.0) == _bounce(1.0, -0.1, 1.0, m_debye_sq=0.0)
    assert table.stats == {"table": 0, "none": 0, "direct": 1}


def test_nan_is_not_read_from_the_table(table):
    assert table.action_over_T(float("nan"), -0.1, 1.0, 1.0) is None
    assert table.action_over_T(1.0, float("nan"), 1.0, 1.0) is None
    assert table.stats["table"] == 0


def test_agrees_with_direct_solves(table):
    """Random (s, tau, c) inside the grid: the table against the bounce solved with the temporal coupling h3 = 2 c."""
    rng = np.random.default_rng(5)
    errors = []
    for s, tau, c in zip(rng.uniform(-6, 4.5, 120), rng.uniform(-6, 5.5, 120), rng.uniform(*table.C_RANGE, 120)):
        A = c * table.A_S + math.exp(s)
        B = float(kernels.edge_B(A, c)) * -math.expm1(-tau)
        errors.append(abs(table.action_over_T(A, B, 1.0, 1.0, h3=2 * c)
                          / _bounce(A, B, 1.0, m_debye_sq=1.0, h3=2 * c) - 1.0))
    assert table.stats["table"] == 120
    assert max(errors) < 1e-6


def test_a_temporal_coupling_outside_the_grid_is_solved(table):
    c = table.C_RANGE[1] + 0.01
    A = c * table.A_S + 0.3
    B = float(kernels.edge_B(A, c)) * 0.5
    assert table.action_over_T(A, B, 1.0, 1.0, h3=2 * c) == pytest.approx(_bounce(A, B, 1.0, m_debye_sq=1.0, h3=2 * c),
                                                                         rel=1e-9)
    assert table.stats["direct"] == 1 and table.stats["table"] == 0


def test_evaluation_is_the_spline_of_scipy(table):
    """The compiled evaluation against scipy's B-spline with the same knots and coefficients, to rounding."""
    from scipy.interpolate import BSpline
    x0, x1, hx, y0, y1, hy, z0, z1, hz, tx, ty, tz, c = table._table["args"]
    rng = np.random.default_rng(2)
    for s, tau, q in zip(rng.uniform(x0, x1, 200), rng.uniform(y0, y1, 200), rng.uniform(z0, z1, 200)):
        plane = BSpline(tx, c, 5, axis=0)(s)
        column = BSpline(ty, plane, 5, axis=0)(tau)
        assert kernels._spline3(s, tau, q, x0, hx, y0, hy, z0, hz, tx, ty, tz, c) == pytest.approx(
            float(BSpline(tz, column, 5)(q)), rel=1e-12, abs=1e-12)


def test_no_kink_at_the_cell_edges(table):
    """The second difference across a cell edge equals the one beside it: the first derivative has no jump there."""
    x0, x1, hx, y0, y1, hy, z0, z1, hz, tx, ty, tz, c = table._table["args"]
    d = 1e-4
    G = lambda s, tau: kernels._spline3(s, tau, 1.0, x0, hx, y0, hy, z0, hz, tx, ty, tz, c)
    for s in (-3.04, 0.33, 2.41):
        for k in range(20, 120, 7):
            tau = y0 + k * hy
            across = (G(s, tau + d) - 2 * G(s, tau) + G(s, tau - d)) / d ** 2
            beside = 0.5 * ((G(s, tau + 3 * d) - 2 * G(s, tau + 2 * d) + G(s, tau + d))
                            + (G(s, tau - d) - 2 * G(s, tau - 2 * d) + G(s, tau - 3 * d))) / d ** 2
            assert abs(across - beside) < 2e-5


def test_noise_estimate_reads_each_axis(table):
    rng = np.random.default_rng(1)
    along_second = np.outer(np.ones(60), rng.standard_normal(60))
    a, b = table._noise(along_second)
    assert a == 0.0 and 0.5 < b < 2.0


def test_format_and_header(table, tmp_path):
    header = table.load()
    assert header["format"] == table.FORMAT and len(header["commit"]) >= 40
    assert header["solve_LO_bounce"]["xtol"] == 1e-11 and header["solve_LO_bounce"]["n_temporal"] == 1
    bad = str(tmp_path / "bad.npz")
    np.savez(bad, c=np.zeros((10, 10)), header=np.array(json.dumps(dict(header, format=table.FORMAT + 1))))
    table.clear()
    with pytest.raises(ValueError, match="format"):
        table.load(bad)


def test_shipped_by_the_wheel():
    """The table sits inside the package directory, which is what the wheel holds, and is small."""
    file = os.path.join(ROOT, "src", "transitionlistener", "tab_data", "lo_bounce_tables", "lo_bounce_table.npz")
    assert 1e4 < os.path.getsize(file) < 3e6


def test_build_round_trip(table, tmp_path):
    """A small grid through the builder: interpolates its nodes, and reads back through load()."""
    out = str(tmp_path / "t.npz")
    header = table.build(out, s_range=(-1.0, 0.0), tau_range=(-0.5, 0.5), step=0.1, workers=1)
    assert header["n"] == [11, 11, 7] and header["noise_rms_sixth_difference"][0] < 1e-6
    table.clear()
    table.load(out)
    for s, tau, c in ((-0.5, 0.2, 1.0), (-0.73, -0.31, 0.97), (-0.1, 0.42, 1.031)):
        A = c * table.A_S + math.exp(s)
        B = float(kernels.edge_B(A, c)) * -math.expm1(-tau)
        assert table.action_over_T(A, B, 1.0, 1.0, h3=2 * c) == pytest.approx(
            _bounce(A, B, 1.0, m_debye_sq=1.0, h3=2 * c), rel=3e-7)
    assert table.action_over_T(table.A_S + math.exp(0.5), 0.0, 1.0, 1.0) is not None
    assert table.stats["direct"] == 1


def test_build_refuses_a_missing_node(table, tmp_path, monkeypatch):
    real = table._solve_node
    monkeypatch.setattr(table, "_solve_node", lambda n: math.nan if n[:2] == (0.0, 0.0) else real(n))
    monkeypatch.setattr(table, "_pool_map", lambda f, nodes, workers: [f(n) for n in nodes])
    with pytest.raises(RuntimeError, match="no bounce"):
        table.build(str(tmp_path / "t.npz"), s_range=(-0.4, 0.4), tau_range=(-0.4, 0.4), step=0.1)
    assert not os.path.exists(tmp_path / "t.npz")


@pytest.mark.parametrize("name,params", [("TL_conformal_u1_no_fermions_3d.py", dict(g=0.7, v_GeV=1.0)),
                                         ("TL_conformal_u1_fermions_3d.py", dict(g=0.6, y=0.4, v_GeV=1.0))])
def test_model_seam(table, name, params):
    """Both models hand their matched parameters to the table, with the counts the table holds."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    pot = _model(name, params)
    for T in (1e-6, 1e-3, 0.05):
        p = pot.soft_parameters(T)
        assert pot.lo_action_over_T(T) == pytest.approx(abelian_higgs_3d.solve_LO_bounce(
            p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"], h3=p["lambda_vl"])[0], rel=1e-6)
    assert table.stats["table"] == 3


def test_no_barrier_keeps_the_cap(table):
    """Hot of the barrier edge the table answers None, and actionNuc gives the cap, as the direct path."""
    import TL_conformal_u1_fermions_3d as dark3d
    pot = _model("TL_conformal_u1_no_fermions_3d.py", dict(g=0.7, v_GeV=1.0))
    T = 0.6
    m3sq, lam3, g3sq, mD2 = pot.matched_3d_parameters(T)
    assert lam3 > 0
    conv = pot.conversionFactor
    direct = pot.actionNuc(T / conv, float("nan"))
    assert pot.lo_action_over_T(T) is None
    assert pot.actionNuc(T / conv, float("nan")) == direct == T / conv * one_field_rate_3d.ACTION_CAP


def test_matching_failure_answers_none(table, monkeypatch):
    pot = _model("TL_conformal_u1_no_fermions_3d.py", dict(g=0.7, v_GeV=1.0))
    monkeypatch.setattr(type(pot), "soft_parameters", lambda self, T: (_ for _ in ()).throw(RuntimeError("pole")))
    assert pot.lo_action_over_T(0.1) is None


def test_the_end_to_end_run_agrees_with_direct_solves(table, monkeypatch):
    """One point through the whole pipeline, with the table and with a bounce solved at every temperature."""
    from transitionlistener.eft_rate import abelian_higgs_3d
    from transitionlistener.interface import _compute_single_point
    from transitionlistener.helper_functions import load_potential

    cls = load_potential(os.path.join(MODELS, "TL_conformal_u1_fermions_3d.py"), "specific_potential")

    def factory(p, *a, **k):
        pot = cls(p, *a, **k)
        c = pot.config.nucleationConf
        c.rate_mode, c.mu4_over_T, c.input_scheme, c.mu3_choice, c.matching_order = (
            "model", 2 * math.pi, "zero_quartic", "gT", "NLO")
        c.include_determinants = False
        pot.config.percolationConf.integral_method = "double_integral"
        return pot

    def observables():
        ctx, error = _compute_single_point(dict(g=0.7, y=0.5, v_GeV=1.5323027730238088), factory, timeout=-1,
                                           verbose=False, include_smbhb=False, max_stage=3)
        assert error is None
        return ctx["strongest_transition_observables"]

    a = observables()
    assert table.stats["table"] > 50

    def direct(m3sq, lam3, g3sq, m_debye_sq, *, h3=None):
        try:
            return abelian_higgs_3d.solve_LO_bounce(m3sq, lam3, g3sq, m_debye_sq=m_debye_sq, h3=h3)[0]
        except ValueError:
            return None

    monkeypatch.setattr(one_field_rate_3d.abelian_lo_bounce_table, "action_over_T", direct)
    b = observables()
    for key in ("Tperc_SM_GeV", "alpha", "betaH_RH", "betaH_S3", "Treh_SM_GeV"):
        assert abs(b[key] / a[key] - 1.0) < 1e-4
