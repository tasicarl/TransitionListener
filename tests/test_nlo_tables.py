"""The NLO table builder, the grid file, and how a 3D model uses it."""
import json
import math
import os
import sys
import types
import warnings

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "models"))

from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402

MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")

from transitionlistener.eft_rate import nlo_tables as nt  # noqa: E402
from transitionlistener.interface import nlo_cli  # noqa: E402

EXAMPLE_FERMION = os.path.join(ROOT, "tests", "data", "EXAMPLE_ONLY_conformal_u1_fermions.npz")
EXAMPLE_U1 = os.path.join(ROOT, "tests", "data", "EXAMPLE_ONLY_conformal_u1_no_fermions.npz")

#: A table row at l_max = 6 from the Python integrator, with the bounce and the mixing solved in GeV. Recorded with the
#: temporal mass at h3 = 2 g3^2, as are the example grids, so the tests that use them set temporal_coupling "2g3sq".
RECORDED = {
    "dark": dict(T_GeV=0.01686325619024715, S3_LO=147.30182784613382, determinants=9.145791716310214,
                 radial=2.188226741507205, log_A_dyn_over_T=-2.1348691472838013,
                 nlo_potential=9.979224433679411, nlo_kinetic=-23.615864203720367, total_no_mixing=166.42684397601585),
    "u1": dict(T_GeV=0.015, S3_LO=101.1770271302694, determinants=9.566923586596602, radial=3.0567892027599783,
               log_A_dyn_over_T=-2.2474944034834694, nlo_potential=10.054214731337831,
               nlo_kinetic=-20.80001925030772, total_no_mixing=120.7981654488172),
}
RECORDED_L0 = {"dark": -6.505540460880255, "u1": -6.578139976716297}
RECORDED_MIXING = {"dark": -39.35312763218035, "u1": -35.705092423318284}
RECORDED_DELTA_S = {"dark": 6.378824869418354, "u1": 8.30287963802095}


@pytest.fixture(autouse=True)
def leading_order_temporal_coupling(request, monkeypatch):
    """The node tables, recorded rows and example grids of these tests are for h3 = 2 g3^2 and g3^2 at LO; the shipped
    grids are not. With the run's g3^2 pinned to that of the example grids, their g3^2 convention is not compared."""
    if not request.node.name.startswith("test_the_shipped_grids"):
        from transitionlistener.config import NucleationConf
        monkeypatch.setattr(NucleationConf, "temporal_coupling", "2g3sq")
        pin_leading_order_g3sq(monkeypatch)
        monkeypatch.delitem(nt.CONVENTIONS, "g3sq_order")


def pin_leading_order_g3sq(monkeypatch):
    """g3^2 = g^2 T in both matching modules, as when the recorded numbers were taken."""
    import importlib
    nt.load_model("TL_conformal_u1_no_fermions_3d")
    for name in ("dr_conformal_u1_fermions_matching", "dr_conformal_u1_no_fermions_matching"):
        mod = importlib.import_module(name)
        monkeypatch.setattr(mod, "gauge_coupling_sq", lambda g, T, mubar, order, _m=mod: _m.g3_sq_LO(g, T))


def model(kind, **conf):
    if kind == "dark":
        pot = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.7, "y": 0.5, "v_GeV": 1.5323027730238088}, verbose=False)
        settings = dict(mu4_over_T=2.0 * math.pi, input_scheme="zero_quartic", mu3_choice="gT")
    else:
        pot = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.7, "v_GeV": 1.6833}, verbose=False)
        settings = dict(mu4_over_T=7.0553, input_scheme="zero_quartic", mu3_choice="g3sq")
    settings = dict(settings, temporal_coupling="2g3sq")
    settings.update(conf, rate_mode="model", matching_order="NLO")
    for key, value in settings.items():
        setattr(pot.config.nucleationConf, key, value)
    return pot


# ------------------------------------------------- one table
@pytest.fixture
def direct_rg(monkeypatch):
    """Recorded rows are for a direct solve of the RG equations; they resolve 1e-12 in the quartic."""
    from transitionlistener.eft_rate import rg
    monkeypatch.setattr(rg, "flow", lambda *a: None)


@pytest.mark.usefixtures("direct_rg")
@pytest.mark.parametrize("kind", sorted(RECORDED))
def test_a_recorded_row_of_both_models(kind):
    """The row in units of T against the recorded row in GeV."""
    want = RECORDED[kind]
    pot = model(kind)
    row = nt.fast_row(pot, want["T_GeV"])
    for key, value in want.items():
        # the determinants move by up to 5e-5 with the bounce solved in units of T; nlo_kinetic, a diagnostic that is
        # not in the exponent, by 5e-4
        assert row[key] == pytest.approx(value, abs=1e-3 if key == "nlo_kinetic" else 1e-4), key
    assert row["S3_LO"] == pytest.approx(want["S3_LO"], abs=1e-6)
    row["mixing"] = nt.mixing_row(pot, want["T_GeV"], 6)
    assert row["mixing"]["value"] == pytest.approx(RECORDED_MIXING[kind], abs=2e-5)
    assert row["mixing"]["l0_term"] == pytest.approx(RECORDED_L0[kind], abs=2e-5)
    assert nt.delta_S_of_row(row) == pytest.approx(RECORDED_DELTA_S[kind], abs=1e-4)


@pytest.mark.usefixtures("direct_rg")
def test_units_of_T_rescue_a_cold_row():
    """S_LO = 20 at g = 0.42: the mixing sum in units of T, and the same sum in GeV on the same bounce."""
    cold = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.42, "v_GeV": 1.0}, verbose=False)
    conf = cold.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice, conf.temporal_coupling = "model", "zero_quartic", "gT", "2g3sq"
    T = 1.1078220303368486e-20
    assert nt.mixing_row(cold, T, 70)["value"] == pytest.approx(-17.4954, abs=1e-3)
    p, _, r, phi3, _ = nt._bounce(cold, T)
    in_GeV = abelian_higgs_3d.mixing_log_det(r, phi3, p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"], l_max=70)
    assert abs(in_GeV["value"]) > 1e3


def test_the_dimension_six_shifts_are_stored_apart_from_delta_S():
    pot = model("u1")
    row = nt.fast_row(pot, 0.015)
    p, _, r, phi3, info = nt._bounce(pot, 0.015)
    assert row["dS6"] == abelian_higgs_3d.dim6_action_shift(r, phi3, p["c6"])
    assert row["dSkin"] == abelian_higgs_3d.dim6_kinetic_action_shift(r, phi3, info["dphi3"], p["c_D2phi4"])
    assert row["dS6"] > 0.0
    row.update(mixing=dict(value=-30.0))
    plain = nt.delta_S_of_row(row)
    row["dS6"] *= 5.0
    assert nt.delta_S_of_row(row) == plain


def _fake_rows(monkeypatch, calls):
    def fast(pot, T):
        calls.append(("fast", T))
        return dict(T_GeV=T, S3_LO=100.0 * T, log_A_dyn_over_T=-2.0,
                    total_no_mixing=100.0 * T + 10.0, dS6=0.1, dSkin=0.01)

    def mixing(pot, T, l_max):
        calls.append(("mixing", T))
        return dict(value=-30.0, spread=0.2)

    monkeypatch.setattr(nt, "fast_row", fast)
    monkeypatch.setattr(nt, "mixing_row", mixing)


def test_a_rerun_skips_finished_rows_and_resumes_the_rest(tmp_path, monkeypatch):
    calls = []
    _fake_rows(monkeypatch, calls)
    out = str(tmp_path / "t.json")
    temps = [(f"{T:g}", T, dict(mult=T)) for T in (1.0, 2.0, 3.0, 4.0)]
    nt.build_table(None, temps[:2], out, l_max=3)
    assert len(calls) == 4
    with open(out) as f:
        assert json.load(f)["table"] is None
    calls.clear()
    state = nt.build_table(None, temps, out, l_max=3)
    assert calls == [("fast", 3.0), ("fast", 4.0), ("mixing", 3.0), ("mixing", 4.0)]
    assert state["table"]["dS6"] == [0.1] * 4
    assert "table_error" not in state
    calls.clear()
    nt.build_table(None, temps, out, l_max=3)
    assert calls == []


def test_a_table_of_another_assembly_is_refused(tmp_path, monkeypatch):
    calls = []
    _fake_rows(monkeypatch, calls)
    out = str(tmp_path / "t.json")
    nt.build_table(None, [("a", 1.0, {})], out, l_max=3)
    with open(out) as f:
        state = json.load(f)
    state["assembly"] = nt.ASSEMBLY - 1
    with open(out, "w") as f:
        json.dump(state, f)
    calls.clear()
    with pytest.raises(nt.NLOGridError, match="holds assembly"):
        nt.build_table(None, [("a", 1.0, {})], out, l_max=3)
    assert calls == []
    with pytest.raises(nt.NLOGridError, match="holds assembly"):
        nt._node_arrays(out, np.array([1.0]))


def test_a_failed_row_is_recorded_and_the_timeout_is_not(tmp_path, monkeypatch):
    from transitionlistener import errors
    out = str(tmp_path / "t.json")

    def boom(pot, T):
        raise ValueError("no barrier")
    monkeypatch.setattr(nt, "fast_row", boom)
    state = nt.build_table(None, [("a", 1.0, {})], out, l_max=3)
    assert state["rows"]["a"]["error"].startswith("ValueError")

    def late(pot, T):
        raise errors.Timeout()
    monkeypatch.setattr(nt, "fast_row", late)
    with pytest.raises(errors.Timeout):
        nt.build_table(None, [("b", 1.0, {})], out, l_max=3)


def test_temperature_at_action_finds_the_target_on_both_models():
    for kind, target in (("dark", 150.0), ("u1", 350.0), ("u1", 20.0), ("u1", 2.0e4)):     # the last just below the barrier edge
        pot = model(kind)
        T = nt.temperature_at_action(pot, target)
        assert nt._bounce(pot, T)[1] == pytest.approx(target, rel=1e-6)


def test_temperature_at_action_near_the_line_where_v_is_far_from_mu0():
    """y^4 = 3 g^4: v/mu0 runs away, while the 3D theory only knows T/mu0. Here S_LO = 150 lies above T = v."""
    y = 0.995 * 3.0 ** 0.25 * 0.76
    pot = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.76, "y": y, "v_GeV": 1.0}, verbose=False)
    conf = pot.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice = "model", "zero_quartic", "gT"
    T = nt.temperature_at_action(pot, 150.0)
    assert nt._bounce(pot, T)[1] == pytest.approx(150.0, rel=1e-6)
    # and where T is far above v: v/mu0 ~ e^-100, which a scan on the scale of v never reaches
    y = 0.999 * 3.0 ** 0.25 * 0.6
    pot = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.6, "y": y, "v_GeV": 1.0}, verbose=False)
    conf = pot.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice = "model", "zero_quartic", "gT"
    assert pot.input_scale_GeV() > 1e20
    T = nt.temperature_at_action(pot, 150.0)
    assert T > 1e20 and nt._bounce(pot, T)[1] == pytest.approx(150.0, rel=1e-6)


def test_the_bounce_reaches_the_low_actions():
    """In GeV the absolute tolerance of the escape point stops this node at S_LO = 30, T ~ 1e-37 GeV."""
    pot = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.46, "y": 0.5, "v_GeV": 1.0}, verbose=False)
    conf = pot.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice = "model", "zero_quartic", "gT"
    T = nt.temperature_at_action(pot, 5.0)
    assert nt._bounce(pot, T)[1] == pytest.approx(5.0, rel=5e-2)


def test_the_bounce_in_units_of_T_is_the_bounce_in_GeV():
    pot = model("dark")
    T = RECORDED["dark"]["T_GeV"]
    p = pot.soft_parameters(T)
    S_g, r_g, phi_g, info_g = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
    _, S_t, r_t, phi_t, info_t = nt._bounce(pot, T)
    assert S_t == pytest.approx(S_g, rel=1e-6)
    assert r_t[-1] == pytest.approx(r_g[-1], rel=1e-3)
    assert phi_t[0] == pytest.approx(phi_g[0], rel=1e-5)
    assert np.interp(100.0, r_t, info_t["dphi3"]) == pytest.approx(np.interp(100.0, r_g, info_g["dphi3"]), rel=1e-3)


def test_temperature_at_action_says_when_there_is_none():
    with pytest.raises(nt.NLOGridError, match="no temperature"):
        nt.temperature_at_action(model("u1"), 1.0e-3, ln_T_over_mu0=(3.0, -3.0))


def test_temperature_at_action_treats_a_failed_shooting_as_no_bounce(monkeypatch):
    """Far below v the shooting can end in a TunnelingError instead of the ValueError of a missing barrier."""
    from transitionlistener import errors
    pot = model("u1")
    real = nt._bounce

    def bounce(p, T):
        if T < 1e-3:
            raise errors.TunnelingError()
        return real(p, T)

    monkeypatch.setattr(nt, "_bounce", bounce)
    with pytest.raises(nt.NLOGridError, match="no temperature"):
        nt.temperature_at_action(pot, 20.0)
    assert nt.temperature_at_action(pot, 150.0) > 1e-3


def test_the_action_depends_on_T_over_v_only():
    """Why one table holds for every v: the model has one dimensionful input."""
    pot_a = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.7, "y": 0.5, "v_GeV": 1.5}, verbose=False)
    pot_b = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.7, "y": 0.5, "v_GeV": 15000.0}, verbose=False)
    for pot in (pot_a, pot_b):
        conf = pot.config.nucleationConf
        conf.rate_mode, conf.input_scheme, conf.mu3_choice = "model", "zero_quartic", "gT"
    x = 0.011
    assert nt._bounce(pot_a, x * 1.5)[1] == pytest.approx(nt._bounce(pot_b, x * 15000.0)[1], rel=1e-6)


# ------------------------------------------------- the grid: synthetic nodes with a known answer
def exact_r(g, y, S):
    return 0.01 + 0.05 * (g - 0.5) ** 2 + 0.02 * y * g + 0.03 * g * math.log(S / 100.0) ** 2


S_GRID = (40.0, 80.0, 120.0, 160.0, 200.0, 240.0, 300.0)


def write_node(path, g, y, factor=1.0, *, mu4=7.0553, mu3="gT", scheme="zero_quartic", model_name="conformal_u1_fermions",
               S_values=S_GRID):
    rows = {}
    for s in S_values:
        # a row such that delta_S_of_row(row) is exactly the wanted correction
        want = exact_r(g, y if y is not None else 0.0, s) * s * (1.0 + 0.1 * (factor - 1.0))
        exponent = want - 1.5 * math.log((s + want) / (2.0 * math.pi))
        T = 1e-2 * (s / 100.0) ** 6
        rows[f"S{s:g}"] = dict(T_GeV=T, S3_LO=s, log_A_dyn_over_T=0.0,
                               total_no_mixing=s + exponent, dS6=1e-3 * s, dSkin=2e-4 * s,
                               mixing=dict(value=0.0, spread=0.1))
    state = dict(model=model_name, g=g, y=y, v_GeV=1.0, mu0_GeV=1.0, mu4_over_T=mu4 * factor, mu4_over_T_central=mu4, mu4_factor=factor,
                 mu3_choice=mu3, input_scheme=scheme, matching_order="NLO", l_max=70, assembly=nt.ASSEMBLY, rows=rows)
    with open(path, "w") as f:
        json.dump(state, f)
    return str(path)


NODES = [(g, y) for g in (0.5, 0.6, 0.7, 0.8, 0.9) for y in (0.0, 0.15, 0.3, 0.45)]


@pytest.fixture(scope="module")
def grid_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("grid")
    tables = {f: [write_node(d / f"n{i}_{f}.json", g, y, f) for i, (g, y) in enumerate(NODES)] for f in (0.5, 1.0, 2.0)}
    out = str(d / "grid.npz")
    nt.assemble_grid(tables, S_GRID, out, purpose="production", couplings=("g", "y"),
                     extra_header={"note": "synthetic"})
    return out


@pytest.fixture
def pot():
    p = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.65, "y": 0.2, "v_GeV": 1.5}, verbose=False)
    conf = p.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice = "model", "zero_quartic", "gT"
    return p


def test_the_header_records_the_conventions_and_the_leave_one_out(grid_file):
    h = nt.load_grid(grid_file).header
    for key, want in (("mu4_over_T", 7.0553), ("mu3_choice", "gT"), ("input_scheme", "zero_quartic"), ("l_max", 70),
                      ("mu4_factors", [0.5, 1.0, 2.0]), ("matching_order", "NLO"), ("purpose", "production"),
                      ("profile_end_fraction", abelian_higgs_3d.TAIL_END_FRACTION)):
        assert h[key] == want, key
    loo = h["leave_one_out"]["1"]
    assert loo["nodes_inside"] == 20 - 4  # all but the four corners of the rectangle
    assert 0.0 < loo["exponent_median"] <= loo["exponent_90"] <= loo["exponent_max"] < 2.0
    # the error of the exponent, the ratio times S_LO, at the worst point the hand can check: a cubic in g at the centre
    # of a uniform grid reproduces this quadratic, so what is left comes from the edge nodes, at S_LO up to 300
    grid = nt.load_grid(grid_file)
    keep = np.arange(len(grid.points)) != 5
    pred = nt._interpolant(grid.points[keep], grid.r[1][keep])(grid.points[5])
    single = float(np.max(np.abs(pred - grid.r[1][5]) * grid.S_nodes))
    assert single <= loo["exponent_max"] and loo["exponent_max"] > float(np.max(np.abs(pred - grid.r[1][5])))
    assert single > 10.0 * float(np.max(np.abs(pred - grid.r[1][5])))


def test_a_grid_is_interpolated_in_the_couplings_and_in_S(grid_file):
    grid = nt.load_grid(grid_file)
    c = grid.correction({"g": 0.65, "y": 0.2}, 1.0)
    for s in S_GRID:
        assert c(1e-3, S3_LO=s) == pytest.approx(exact_r(0.65, 0.2, s) * s, rel=5e-3)
    assert c(1e-3, S3_LO=175.0) == pytest.approx(exact_r(0.65, 0.2, 175.0) * 175.0, rel=5e-2)    # between the S nodes
    # the correction moves with S at a fixed coupling point: one value per S node, not one for all
    assert abs(c(1e-3, S3_LO=280.0) / 280.0 - c(1e-3, S3_LO=50.0) / 50.0) > 1e-3
    # and with the coupling at a fixed S
    far = grid.correction({"g": 0.85, "y": 0.4}, 1.0)
    assert far(1e-3, S3_LO=160.0) == pytest.approx(exact_r(0.85, 0.4, 160.0) * 160.0, rel=5e-3)


def test_the_mu4_factor_selects_its_own_table(grid_file):
    grid = nt.load_grid(grid_file)
    point = {"g": 0.6, "y": 0.15}
    assert grid.correction(point, 2.0)(1e-3, S3_LO=120.0) == pytest.approx(
        1.1 * grid.correction(point, 1.0)(1e-3, S3_LO=120.0), rel=1e-9)
    with pytest.raises(nt.NLOGridError, match="mu4 factor 3"):
        grid.correction(point, 3.0)


def test_outside_the_couplings_is_an_error_naming_the_point_and_the_range(grid_file):
    grid = nt.load_grid(grid_file)
    for point in ({"g": 0.95, "y": 0.2}, {"g": 0.45, "y": 0.2}, {"g": 0.6, "y": 0.5}):
        with pytest.raises(nt.NLOGridError) as e:
            grid.correction(point)
        text = str(e.value)
        assert str(point["g"]) in text and "g in [0.5, 0.9]" in text and "y in [0, 0.45]" in text


def test_outside_the_range_of_S_is_an_error_naming_the_temperature_and_the_range(grid_file):
    c = nt.load_grid(grid_file).correction({"g": 0.65, "y": 0.2})
    for s, word in ((39.0, "below"), (301.0, "above"), (float("nan"), "below"), (0.0, "below"), (-1.0, "below")):
        with pytest.raises(nt.NLOGridError) as e:
            c(1.25e-3, S3_LO=s)
        assert word in str(e.value) and "[40, 300]" in str(e.value) and "1.25e-03" not in str(e.value)
        assert "0.00125" in str(e.value)
    with pytest.raises(nt.NLOGridError, match="pass S3_LO"):
        c(1e-3)


def test_the_continuation_exists_only_where_the_rate_does_not_matter_and_is_smooth(grid_file):
    c = nt.load_grid(grid_file).correction({"g": 0.65, "y": 0.2}, hot_side_negligible=True, cold_side_negligible=True)
    assert c.continued_calls == 0
    c(1e-3, S3_LO=100.0)
    assert c.continued_calls == 0
    for edge in (40.0, 300.0):
        h = 1e-6
        in_side, out_side = (edge * (1 + 2 * h), edge * (1 + h)), (edge * (1 - h), edge * (1 - 2 * h))
        if edge > 100.0:
            in_side, out_side = out_side, in_side
        slope_in = (c(1e-3, S3_LO=in_side[0]) - c(1e-3, S3_LO=in_side[1])) / (in_side[0] - in_side[1])
        slope_out = (c(1e-3, S3_LO=out_side[0]) - c(1e-3, S3_LO=out_side[1])) / (out_side[0] - out_side[1])
        assert slope_in == pytest.approx(slope_out, rel=1e-3)
    assert c.continued_calls > 0
    # bounded far outside
    assert abs(c(1e-3, S3_LO=1e5) / 1e5) < abs(c.r_edges[1]) + abs(c._slopes[1])
    assert abs(c(1e-3, S3_LO=1e-6) / 1e-6) < abs(c.r_edges[0]) + abs(c._slopes[0])


def test_the_model_takes_the_grid_with_determinants_and_no_table(grid_file, pot, monkeypatch):
    conf = pot.config.nucleationConf
    conf.nlo_grid_file, conf.include_determinants, conf.include_dim6 = grid_file, True, False
    monkeypatch.setattr(nt, "build_table", lambda *a, **k: pytest.fail("a run must never build a table"))
    monkeypatch.setattr(nt, "build_node", lambda *a, **k: pytest.fail("a run must never build a table"))
    T = 0.012
    S_LO = nt._bounce(pot, T)[1]
    assert pot.nlo_correction is None
    action = pot.actionNuc(T / pot.conversionFactor, float("nan")) / (T / pot.conversionFactor)
    assert isinstance(pot.nlo_correction, nt.GridCorrection)
    want = nt.load_grid(grid_file).correction({"g": 0.65, "y": 0.2})(T, S3_LO=S_LO)
    assert action == pytest.approx(S_LO + want, rel=1e-8)     # the action comes from the table of the bounce
    assert abs(want) > 1.0


def test_an_attached_table_wins_over_the_grid(grid_file, pot):
    conf = pot.config.nucleationConf
    conf.nlo_grid_file, conf.include_determinants = grid_file, True
    pot.nlo_correction = lambda T, S3_LO=None: 7.0
    T = 0.012
    assert pot.actionNuc(T / pot.conversionFactor, float("nan")) / (T / pot.conversionFactor) == pytest.approx(
        nt._bounce(pot, T)[1] + 7.0, rel=1e-8)


def test_dimension_six_is_chosen_at_run_time(grid_file, pot):
    conf = pot.config.nucleationConf
    conf.nlo_grid_file, conf.include_dim6 = grid_file, False
    plain = nt.correction_for_model(pot)
    conf.include_dim6 = True
    with_dim6 = nt.correction_for_model(pot)
    # S_6 enters the exponent E of eq. (A.2), i.e. eq. (A.4) is solved with it inside the logarithm: from the plain
    # Delta S_0, Delta S = Delta S_0 + S_6 + (3/2) ln[(S + Delta S)/(S + Delta S_0)]; adding S_6 alone misses this by
    # (3/2) S_6/S, about 1% of S_6 here
    S, d0, s6 = 120.0, plain(1e-3, S3_LO=120.0), (1e-3 + 2e-4) * 120.0
    d = d0 + s6
    for _ in range(50):
        d = d0 + s6 + 1.5 * math.log((S + d) / (S + d0))
    assert with_dim6(1e-3, S3_LO=S) == pytest.approx(d, rel=1e-9)
    assert abs(with_dim6(1e-3, S3_LO=S) - (d0 + s6)) > 1e-3 * s6


def test_a_grid_without_the_shifts_refuses_dim6(tmp_path, grid_file):
    grid = nt.load_grid(grid_file)
    data = dict(points=grid.points, S_nodes=grid.S_nodes, r=grid.r, r_dS6=grid.r_dS6 * np.nan, r_dSkin=grid.r_dSkin * np.nan,
                ln_T_over_mu0=grid.ln_T_over_mu0, half_spread=grid.half_spread, S_floor=grid.S_floor)
    out = str(tmp_path / "nodim6.npz")
    nt.write_grid(out, grid.header, data)
    other = nt.load_grid(out)
    assert other.correction({"g": 0.65, "y": 0.2})(1e-3, S3_LO=100.0) > 0.0
    with pytest.raises(nt.NLOGridError, match="no dimension-six"):
        other.correction({"g": 0.65, "y": 0.2}, dim6=True)


@pytest.mark.parametrize("attr, value, word", [
    ("mu4_over_T", 2.0 * math.pi, "mu4_over_T"),
    ("mu4_over_T", 7.0553 * 1.5, "mu4_over_T"),
    ("mu3_choice", "g3sq", "mu3_choice"),
    ("input_scheme", "CW", "input_scheme"),
    ("matching_order", "LO", "matching_order"),
])
def test_a_run_with_other_conventions_is_refused(grid_file, pot, attr, value, word):
    conf = pot.config.nucleationConf
    conf.nlo_grid_file = grid_file
    nt.correction_for_model(pot)
    setattr(conf, attr, value)
    with pytest.raises(nt.NLOGridError, match=word):
        nt.correction_for_model(pot)


def test_a_band_scale_of_the_grid_is_accepted_and_uses_its_table(grid_file, pot):
    conf = pot.config.nucleationConf
    conf.nlo_grid_file, conf.include_dim6 = grid_file, False
    central = nt.correction_for_model(pot)(1e-3, S3_LO=120.0)
    conf.mu4_over_T = 2.0 * 7.0553
    assert nt.correction_for_model(pot)(1e-3, S3_LO=120.0) == pytest.approx(1.1 * central, rel=1e-9)


def test_a_grid_of_another_model_is_refused(grid_file):
    p = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.65, "v_GeV": 1.8}, verbose=False)
    conf = p.config.nucleationConf
    conf.nlo_grid_file, conf.input_scheme, conf.mu3_choice = grid_file, "zero_quartic", "gT"
    with pytest.raises(nt.NLOGridError, match="is for the model"):
        nt.correction_for_model(p)


def test_no_shipped_grid_is_an_error_not_a_build(pot, tmp_path, monkeypatch):
    monkeypatch.setattr(nt, "GRID_DIR", str(tmp_path))
    with pytest.raises(nt.NLOGridError, match="no NLO grid at"):
        nt.correction_for_model(pot)


@pytest.mark.parametrize("name, couplings", [("TL_conformal_u1_no_fermions_3d", {"g": 0.7}), ("TL_conformal_u1_fermions_3d", {"g": 0.7, "y": 0.5})])
def test_the_shipped_grids_serve_both_models(name, couplings):
    p = nt.load_model(name)(dict(couplings, v_GeV=1.5), verbose=False)
    conf = p.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice, conf.mu4_over_T = "model", "zero_quartic", "gT", 7.0553
    conf.matching_order = "NLO"
    corr = nt.correction_for_model(p)
    assert corr.S_range == (5.0, 350.0) and corr.hot_side_negligible


@pytest.mark.parametrize("v, hot, cold", [(1.5, True, True), (1e-30, False, True), (1e17, True, False)])
def test_the_rate_window_follows_v(grid_file, v, hot, cold):
    """At the hot end the rate is below H^4, at the cold end above it, by e^RATE_MARGIN, or the side stays an error."""
    p = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.65, "y": 0.2, "v_GeV": v}, verbose=False)
    conf = p.config.nucleationConf
    conf.nlo_grid_file, conf.input_scheme, conf.mu3_choice = grid_file, "zero_quartic", "gT"
    c = nt.correction_for_model(p)
    assert (c.hot_side_negligible, c.cold_side_negligible) == (hot, cold)


def test_the_rate_window_flips_where_the_rate_formula_says(grid_file):
    """Both edges of the window solved by hand: with H^2 = c v^4 and T = v e^x, ln(rate/H^4) falls with ln v as -4 ln v."""
    def window(v):
        p = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.65, "y": 0.2, "v_GeV": v}, verbose=False)
        conf = p.config.nucleationConf
        conf.nlo_grid_file, conf.input_scheme, conf.mu3_choice = grid_file, "zero_quartic", "gT"
        return p, nt.correction_for_model(p)

    p, c = window(1.0)
    ln_H_at_1 = 0.5 * math.log(p.vacuum_energy_GeV4() / 3.0) - math.log(nt.M_PL_GEV)
    ln_mu0_at_1 = math.log(p.input_scale_GeV())

    def edge(S, r, x, margin):
        total = S * (1.0 + r)
        return (4.0 * (x + ln_mu0_at_1 - ln_H_at_1) + 1.5 * math.log(total / (2.0 * math.pi)) - total - margin) / 4.0

    ln_v_hot = edge(300.0, c.r_edges[1], c.ln_T_over_mu0[1], -nt.RATE_MARGIN)
    ln_v_cold = edge(40.0, c.r_edges[0], c.ln_T_over_mu0[0], nt.RATE_MARGIN)
    assert window(math.exp(ln_v_hot + 0.02))[1].hot_side_negligible is True
    assert window(math.exp(ln_v_hot - 0.02))[1].hot_side_negligible is False
    assert window(math.exp(ln_v_cold - 0.02))[1].cold_side_negligible is True
    assert window(math.exp(ln_v_cold + 0.02))[1].cold_side_negligible is False


def test_the_input_scale_of_both_models():
    import dr_conformal_u1_fermions_matching as dr
    dark, free = model("dark"), model("u1")
    assert dark.input_scale_GeV() == pytest.approx(1.5323027730238088 / dr.vev_from_zero_quartic(0.7, 0.5, 1.0), rel=1e-14)
    assert free.input_scale_GeV() == pytest.approx(0.7 * 1.6833 * math.exp(-1.0 / 6.0), rel=1e-14)
    for pot in (dark, free):
        pot.config.nucleationConf.input_scheme = "CW"
        assert pot.input_scale_GeV() == pytest.approx(dr.input_scale(0.7, pot.v_GeV), rel=1e-14)


def test_the_vacuum_energy_behind_the_hubble_rate():
    """V(0) - V(v) of the zero-quartic potential at v = 1 GeV, and an H of order 1e-20 GeV."""
    p = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.65, "y": 0.2, "v_GeV": 1.0}, verbose=False)
    H2 = p.vacuum_energy_GeV4() / 3.0 / nt.M_PL_GEV ** 2
    assert 1e-41 < H2 < 1e-38
    assert p.vacuum_energy_GeV4() == pytest.approx((3 * 0.65 ** 4 - 0.2 ** 4) / (128 * math.pi ** 2), rel=1e-12)
    free = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.65, "v_GeV": 1.0}, verbose=False)
    assert free.vacuum_energy_GeV4() == pytest.approx(3 * 0.65 ** 4 / (128 * math.pi ** 2), rel=1e-12)


def test_a_table_that_stops_short_of_the_hot_end_is_refused(tmp_path):
    bad = write_node(tmp_path / "a.json", 0.5, 0.1, S_values=(80.0, 120.0, 160.0))
    with pytest.raises(nt.NLOGridError, match="covers S_LO"):
        nt.assemble_grid({1.0: [bad]}, (80.0, 200.0), str(tmp_path / "g.npz"), purpose="example", couplings=("g", "y"))


def test_a_failed_row_inside_a_table_is_refused(tmp_path):
    path = write_node(tmp_path / "a.json", 0.5, 0.1)
    with open(path) as f:
        d = json.load(f)
    d["rows"]["S120"] = dict(target_S=120.0, T_GeV=1e-3, error="ValueError: no barrier")
    with open(path, "w") as f:
        json.dump(d, f)
    with pytest.raises(nt.NLOGridError, match="failed inside the table"):
        nt.assemble_grid({1.0: [path]}, S_GRID, str(tmp_path / "g.npz"), purpose="example", couplings=("g", "y"))


@pytest.fixture(scope="module")
def floored_grid(tmp_path_factory):
    """As grid_file at one factor, with the node (0.7, 0.3) ending at S_LO = 120: its bounce stops there."""
    d = tmp_path_factory.mktemp("floored")
    files = []
    for i, (g, y) in enumerate(NODES):
        files.append(write_node(d / f"n{i}.json", g, y, S_values=S_GRID[2:] if (g, y) == (0.7, 0.3) else S_GRID))
    out = str(d / "grid.npz")
    nt.assemble_grid({1.0: files}, S_GRID, out, purpose="production", couplings=("g", "y"))
    return out


def test_below_the_floor_of_a_neighbouring_node_counts_as_outside(floored_grid):
    grid = nt.load_grid(floored_grid)
    near = grid.correction({"g": 0.68, "y": 0.28})
    assert near.S_range == (120.0, 300.0)
    assert near(1e-3, S3_LO=121.0) == pytest.approx(exact_r(0.68, 0.28, 121.0) * 121.0, rel=5e-2)
    with pytest.raises(nt.NLOGridError, match=r"\[120, 300\]"):
        near(1e-3, S3_LO=119.0)
    assert grid.correction({"g": 0.85, "y": 0.1}).S_range == (40.0, 300.0)


def test_the_leave_one_out_skips_what_a_node_does_not_hold():
    points = np.array([[0.5], [0.6], [0.7], [0.8], [0.9]])
    S_nodes = np.array([40.0, 80.0, 120.0])
    r = np.outer(points[:, 0], np.array([1.0, 0.1, 0.2]))
    r[:, 0] += np.array([3.0, -2.0, 5.0, -4.0, 1.0])         # held values below the floor, not a function of g
    x = np.log(S_nodes) * np.ones((5, 3))
    assert nt.leave_one_out(points, S_nodes, r, x)["exponent_max"] > 10.0
    h = nt.leave_one_out(points, S_nodes, r, x, S_floor=np.full(5, 80.0))
    assert h["exponent_max"] < 1e-6 and h["nodes_inside"] == 3


def test_the_floor_of_one_coupling(tmp_path):
    files = [write_node(tmp_path / f"n{i}.json", g, None, model_name="conformal_u1_no_fermions",
                        S_values=S_GRID[1:] if g == 0.7 else S_GRID) for i, g in enumerate((0.5, 0.6, 0.7, 0.8, 0.9))]
    out = str(tmp_path / "g.npz")
    nt.assemble_grid({1.0: files}, S_GRID, out, purpose="production", couplings=("g",))
    grid = nt.load_grid(out)
    assert grid.correction({"g": 0.65}).S_range[0] == 80.0
    assert grid.correction({"g": 0.7}).S_range[0] == 80.0
    assert grid.correction({"g": 0.85}).S_range[0] == 40.0


def test_the_continuation_keeps_the_slope_of_r_at_the_edge(grid_file):
    c = nt.load_grid(grid_file).correction({"g": 0.65, "y": 0.2}, hot_side_negligible=True)
    assert c._slopes[1] != 0.0
    u = 1e-3
    assert c(1e-3, S3_LO=300.0 * math.exp(u)) / (300.0 * math.exp(u)) - c.r_edges[1] == pytest.approx(c._slopes[1] * u, rel=1e-4)


def test_nodes_with_different_conventions_do_not_assemble(tmp_path):
    a = write_node(tmp_path / "a.json", 0.5, 0.1)
    b = write_node(tmp_path / "b.json", 0.6, 0.1, mu3="g3sq")
    c = write_node(tmp_path / "c.json", 0.6, 0.1, mu4=2.0 * math.pi)
    with pytest.raises(nt.NLOGridError, match="mu3_choice"):
        nt.assemble_grid({1.0: [a, b]}, S_GRID, str(tmp_path / "g.npz"), purpose="example", couplings=("g", "y"))
    with pytest.raises(nt.NLOGridError, match="mu4/T"):
        nt.assemble_grid({1.0: [a, c]}, S_GRID, str(tmp_path / "g.npz"), purpose="example", couplings=("g", "y"))


def test_factors_with_different_nodes_do_not_assemble(tmp_path):
    a = write_node(tmp_path / "a.json", 0.5, 0.1)
    b = write_node(tmp_path / "b.json", 0.6, 0.1, 2.0)
    with pytest.raises(nt.NLOGridError, match="are not those"):
        nt.assemble_grid({1.0: [a], 2.0: [b]}, S_GRID, str(tmp_path / "g.npz"), purpose="example", couplings=("g", "y"))


# ------------------------------------------------- the example grids, from real tables
def test_the_example_grids_say_what_they_are():
    for path, couplings, mu4, factors in ((EXAMPLE_FERMION, ["g", "y"], 2.0 * math.pi, [1.0]),
                                          (EXAMPLE_U1, ["g"], 7.0553, [0.5, 1.0, 2.0])):
        assert "EXAMPLE_ONLY" in path
        h = nt.load_grid(path).header
        assert h["purpose"] == "example" and h["couplings"] == couplings and h["mu4_factors"] == factors
        assert h["mu4_over_T"] == pytest.approx(mu4) and h["mu3_choice"] == "gT"
        assert "example" in nt.load_grid(path).describe()
        for per_factor in h["leave_one_out"].values():
            assert per_factor["exponent_max"] < 1.0


def test_the_example_grid_reproduces_its_own_nodes():
    grid = nt.load_grid(EXAMPLE_FERMION)
    i = 7
    point = dict(zip(grid.couplings, grid.points[i]))
    c = grid.correction(point)
    for j, s in enumerate(grid.S_nodes):
        assert c(1e-3, S3_LO=s) == pytest.approx(grid.r[0, i, j] * s, rel=1e-9)
    # a real number of the shipped table: (0.7, 0.5) is outside, (0.7, 0.2) a node; Delta S there at S_LO = 150
    assert 3.0 < grid.correction({"g": 0.7, "y": 0.2})(1e-3, S3_LO=150.0) < 7.0


def test_the_example_grid_of_one_coupling_raises_outside_it():
    grid = nt.load_grid(EXAMPLE_U1)
    assert grid.correction({"g": 0.75})(1e-3, S3_LO=150.0) > 0.0
    with pytest.raises(nt.NLOGridError, match=r"g in \[0.55, 0.9\]"):
        grid.correction({"g": 0.5})


def test_the_model_refuses_the_narrow_example_grid_at_the_temperatures_outside_it():
    p = nt.load_model("TL_conformal_u1_no_fermions_3d")({"g": 0.7, "v_GeV": 1.6833}, verbose=False)
    conf = p.config.nucleationConf
    conf.rate_mode, conf.include_determinants, conf.nlo_grid_file = "model", True, EXAMPLE_U1
    conf.input_scheme, conf.mu3_choice, conf.mu4_over_T = "zero_quartic", "gT", 7.0553
    T_hot = nt.temperature_at_action(p, 300.0)
    with pytest.raises(nt.NLOGridError, match=r"\[125, 175\]"):
        p.actionNuc(T_hot / p.conversionFactor, float("nan"))


# ------------------------------------------------- the configuration file reaches a run
def test_the_settings_of_a_grid_run_are_allowed_in_the_nucleation_block():
    from transitionlistener.config import NucleationConf
    for key in ("input_scheme", "mu3_choice", "nlo_grid_file", "include_dim6", "include_determinants"):
        assert key in vars(NucleationConf)


def test_a_single_point_gets_the_nucleation_block(monkeypatch):
    from types import SimpleNamespace
    from transitionlistener.interface import single_point

    class Stop(Exception):
        pass

    made = {}

    def compute(params, potential, *a, **k):
        made["pot"] = potential({"g": 0.7, "v_GeV": 1.7}, verbose=False)
        raise Stop

    monkeypatch.setattr(single_point, "_compute_single_point", compute)
    monkeypatch.setattr(single_point, "_build_plot_plan", lambda conf: SimpleNamespace(ordered_keys=[], wants_gw_spectrum=False))
    monkeypatch.setattr(single_point, "PlotExecutor", lambda *a, **k: SimpleNamespace(stage_callback=None))
    monkeypatch.setattr(single_point, "Logger", lambda path: SimpleNamespace(write=lambda *a: None))
    import tempfile
    conf = SimpleNamespace(output_path=tempfile.mkdtemp() + "/", params={"g": 0.7, "v_GeV": 1.7},
                           modelfile=os.path.join(ROOT, "models", "TL_conformal_u1_no_fermions_3d.py"),
                           potential_name="specific_potential", timeout=-1,
                           nucleation={"rate_mode": "model", "mu3_choice": "gT"})
    try:
        single_point.single(conf)
    except Stop:
        pass
    assert made["pot"].config.nucleationConf.rate_mode == "model"
    assert made["pot"].config.nucleationConf.mu3_choice == "gT"


# ------------------------------------------------- the command line
def test_the_command_line_builds_on_the_scale_mu0(tmp_path, monkeypatch):
    """Near y^4 = 3 g^4 a v of 1 GeV would put the temperatures of the table beyond the range of a float."""
    seen = {}

    def fake(pot, S_targets, stem, mu4_factors, l_max, metadata, log):
        seen.update(mu0=pot.input_scale_GeV(), v=pot.v_GeV)
        return []

    monkeypatch.setattr(nt, "build_node", fake)
    y = 0.999 * 3.0 ** 0.25 * 0.6
    assert nlo_cli.main(["build", "--model-file", os.path.join(MODELS, "TL_conformal_u1_fermions_3d.py"), "--param", "g=0.6", "--param", f"y={y}", "--out-stem", str(tmp_path / "n")]) == 0
    assert seen["mu0"] == pytest.approx(nlo_cli.BUILD_MU0_GEV, rel=1e-12) and seen["v"] < 1e-5
    nlo_cli.main(["build", "--model-file", os.path.join(MODELS, "TL_conformal_u1_no_fermions_3d.py"), "--param", "g=0.6", "--param", "v_GeV=3.0", "--out-stem", str(tmp_path / "n")])
    assert seen["v"] == 3.0


def test_the_command_line_builds_assembles_and_reports(tmp_path, monkeypatch, capsys):
    def fake(pot, S_targets, stem, mu4_factors, l_max, metadata, log):
        assert metadata["model"] == "conformal_u1_no_fermions" and metadata["mu3_choice"] == "gT"
        assert pot.config.nucleationConf.input_scheme == "zero_quartic" and l_max == abelian_higgs_3d.MIXING_L_MAX
        assert mu4_factors == [0.5, 2.0]
        return [write_node(tmp_path / "n_mu0.5.json", 0.5, None, model_name="conformal_u1_no_fermions")]

    monkeypatch.setattr(nt, "build_node", fake)
    assert nlo_cli.main(["build", "--model-file", os.path.join(MODELS, "TL_conformal_u1_no_fermions_3d.py"), "--param", "g=0.7", "--out-stem", str(tmp_path / "n"),
                         "--mu4-factors", "0.5,2"]) == 0
    for i, g in enumerate((0.6, 0.7, 0.8)):
        write_node(tmp_path / f"m{i}.json", g, None, model_name="conformal_u1_no_fermions")
    out = str(tmp_path / "g.npz")
    assert nlo_cli.main(["assemble", "--couplings", "g", "--nodes", str(tmp_path / "*.json"), "--out", out,
                         "--S-nodes", ",".join(f"{s:g}" for s in S_GRID)]) == 0
    assert nlo_cli.main(["info", out]) == 0
    assert "4 nodes" in capsys.readouterr().out


@pytest.mark.parametrize("rows, code", [
    ({"S20": {"target_S": 20.0, "error": "no temperature"}}, 1),
    ({"S40": {"target_S": 40.0, "S3_LO": 40.0, "mixing": {}}, "S20": {"target_S": 20.0, "error": "no temperature"}}, 0),
    ({"S40": {"target_S": 40.0, "S3_LO": 40.0, "mixing": {}}, "S80": {"target_S": 80.0, "error": "x"}}, 1),
    ({"S40": {"target_S": 40.0, "S3_LO": 40.0, "mixing": {}}, "S80": {"target_S": 80.0, "mixing_error": "x"}}, 1),
])
def test_the_command_line_fails_for_a_row_lost_above_the_floor(tmp_path, monkeypatch, rows, code):
    def fake(pot, S_targets, stem, mu4_factors, l_max, metadata, log):
        path = tmp_path / "n_mu1.json"
        path.write_text(json.dumps({"rows": rows}))
        return [str(path)]

    monkeypatch.setattr(nt, "build_node", fake)
    assert nlo_cli.main(["build", "--model-file", os.path.join(MODELS, "TL_conformal_u1_no_fermions_3d.py"), "--param", "g=0.7", "--out-stem", str(tmp_path / "n")]) == code


def test_build_node_restores_mu4_and_keeps_the_temperatures_of_a_resumed_node(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(nt, "temperature_at_action",
                        lambda pot, s, memo=None: seen.append(pot.config.nucleationConf.mu4_over_T) or s)
    tables = []

    def build(pot, temps, out, **k):
        tables.append((out, temps, k["metadata"]["mu4_over_T"]))
        with open(out, "w") as f:
            json.dump({"rows": {t[0]: {"T_GeV": t[1]} for t in temps}}, f)

    monkeypatch.setattr(nt, "build_table", build)
    p = model("u1", mu4_over_T=7.0553)
    stem = str(tmp_path / "n")
    files = nt.build_node(p, (30.0, 20.0), stem, mu4_factors=(0.5, 2.0), metadata={})
    assert files == [stem + "_mu0.5.json", stem + "_mu2.json"]
    assert seen == [7.0553 * 0.5] * 2 + [7.0553 * 2.0] * 2
    assert p.config.nucleationConf.mu4_over_T == 7.0553
    assert [t[0] for t in tables[0][1]] == ["S20", "S30"]
    seen.clear()
    nt.build_node(p, (30.0, 20.0), stem, mu4_factors=(0.5,), metadata={})
    assert seen == []


def test_the_search_reaches_a_bounce_colder_than_the_old_end_of_the_range():
    pot = nt.load_model("TL_conformal_u1_fermions_3d")({"g": 0.42, "y": 0.5, "v_GeV": 1.0}, verbose=False)
    conf = pot.config.nucleationConf
    conf.rate_mode, conf.input_scheme, conf.mu3_choice = "model", "zero_quartic", "gT"
    T = nt.temperature_at_action(pot, 20.0)
    assert math.log(T / pot.input_scale_GeV()) < -100.0


def test_a_run_above_the_reach_of_the_expansion_is_warned(monkeypatch):
    grid_stub = types.SimpleNamespace(header={"model": "m"}, describe=lambda: "stub", correction=lambda *a, **k: types.SimpleNamespace(
        S_range=(5.0, 350.0), r_edges=(0.0, 0.0), ln_T_over_mu0=(-5.0, 0.0)))
    monkeypatch.setattr(nt, "load_grid", lambda path: grid_stub)
    monkeypatch.setattr(nt, "_check_conventions", lambda grid, conf: 1.0)
    conf = types.SimpleNamespace(nucleationConf=types.SimpleNamespace(nlo_grid_file="x"))

    def pot(g):
        return types.SimpleNamespace(config=conf, nlo_grid_name="m", nlo_grid_point=lambda: {"g": g},
                                     vacuum_energy_GeV4=lambda: 1.0, input_scale_GeV=lambda: 1.0,
                                     nlo_expansion_limits={"g": 0.85})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nt.correction_for_model(pot(0.85))
    with pytest.warns(UserWarning, match="g > 0.85"):
        nt.correction_for_model(pot(0.86))


def test_the_correction_follows_a_curved_ratio_between_the_action_nodes():
    """Between the nodes in S_LO the interpolation follows the curvature of a smooth Delta S / S_LO to 1e-5 in
    Delta S, as the tables of the shipped models need; a monotone interpolant misses it by of order 1e-2."""
    S_nodes = np.array([5.0, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 165, 180, 200, 220, 245,
                        270, 300, 350])
    def ratio(S):
        return (0.3 - 0.9 * np.log(S) / S - 0.4 * np.exp(-S / 60.0)) / S
    corr = nt.GridCorrection(S_nodes, ratio(S_nodes), np.log(S_nodes / 400.0), {"g": 0.5}, "test", True, True)
    S_mid = np.sqrt(S_nodes[6:-1] * S_nodes[7:])
    err = [abs(corr(1.0, S3_LO=float(S)) - float(S * ratio(S))) for S in S_mid]
    assert max(err) < 1e-5
