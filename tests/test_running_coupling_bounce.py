"""transitionlistener.eft_rate.running_coupling_bounce: the table of S_3/T of the four-dimensional rate with running
couplings, from ladders of bounces to the function a run reads."""
import json
import math

import numpy as np
import pytest

from transitionlistener.eft_rate import running_coupling_bounce as R


def ladder(slope=8.0, offset=400.0, nan_at=None, point=None):
    """A node whose S_3/T rises linearly in ln(T/mu0), the cap above, and optionally one failed temperature."""
    xs = np.arange(-60.0, 25.0, math.log(10.0) / R.PER_DECADE)
    rows = []
    for x in xs:
        S = offset + slope * x
        rows.append((float(x), R.ACTION_CAP if S > 600 else (math.nan if nan_at is not None and abs(x - nan_at) < 1e-9
                                                            else float(S))))
    return dict(rows=rows, point=point or {"g": 0.5}, couplings=["g"], mu_over_piT=1.0, model="m",
                params={"g": 0.5})


class StubModel:
    """A model whose reference action rises linearly in ln(T/mu0), more slowly than the ladders."""

    def __init__(self, params, verbose=False):
        self.g = params["g"]

    def rg_reference_action(self, ln_x):
        return 200.0 + 5.0 * ln_x


def test_a_failed_temperature_is_left_out_of_the_node():
    x_nan = sorted(r[0] for r in ladder()["rows"])[150]
    clean, holed = R.node_on_S(ladder()), R.node_on_S(ladder(nan_at=x_nan))
    assert np.all(np.isfinite(holed)) and np.all(np.diff(holed) > 0)
    assert np.allclose(holed, clean, atol=1e-6, rtol=0)


def test_the_table_returns_the_ladder_it_was_built_from(tmp_path):
    paths = []
    for i, g in enumerate((0.4, 0.5, 0.6, 0.7)):
        p = tmp_path / f"n{i}.json"
        p.write_text(json.dumps(ladder(slope=8.0 + 4 * g, point={"g": g})))
        paths.append(str(p))
    head = R.assemble(paths, str(tmp_path / "t.npz"), StubModel)
    assert head["couplings"] == ["g"] and head["leave_one_out_S"]["nodes_inside"] == 2
    ref = R.reference_ln_x(StubModel({"g": 0.5}).rg_reference_action)
    S_of = R.table_at(str(tmp_path / "t.npz"), {"g": 0.5}, 1.0, ref)
    for x in (-30.0, -20.0, -5.0):
        assert S_of(x) == pytest.approx(400.0 + 10.0 * x, abs=1e-6)
    # colder than the table: the slope continued; hotter: rolled over smoothly to the cap
    assert S_of(-38.6) == pytest.approx(400.0 - 386.0, abs=1e-6)
    assert S_of(-60.0) == pytest.approx(1e-3)
    assert S_of(200.0) == pytest.approx(R.ACTION_CAP, rel=5e-3) and S_of(9.9) == pytest.approx(499.0, abs=0.5)
    assert R.table_at(str(tmp_path / "t.npz"), {"g": 0.5}, 2.0, ref) is None      # another scale factor
    assert R.table_at(str(tmp_path / "t.npz"), {"g": 0.9}, 1.0, ref) is None      # outside the couplings


def test_the_shipped_table_of_the_model_with_fermions_agrees_with_a_direct_bounce():
    """At a node of the shipped table (g, y) = (0.6, 0.3), the action the model reads equals a direct bounce."""
    import os
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
    import TL_conformal_u1_fermions_rg as model
    pot = model.specific_potential({"g": 0.6, "y": 0.3, "v_GeV": 1.0}, verbose=False)
    table = pot._rg_table()
    assert table is not None
    xs = np.linspace(-40.0, 0.0, 4001)
    x = xs[int(np.argmin([abs(table(v) - 120.0) for v in xs]))]
    direct = R.solve_action_over_T(pot, pot.mu0 * math.exp(x))
    assert table(x) == pytest.approx(direct, abs=0.05)
    assert pot.action_over_T(pot.mu0 * math.exp(x)) == pytest.approx(table(x), rel=1e-12)
