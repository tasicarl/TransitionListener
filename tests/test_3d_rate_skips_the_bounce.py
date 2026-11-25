"""With the 3D rate the action comes from the 3D theory, so the 4D bounce must not be solved.

Solving it at every temperature and discarding it would cost about half the bounce time of a run.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "models"))
import dr_conformal_u1_no_fermions_matching as nf  # noqa: E402
import TL_conformal_u1_no_fermions_3d as m3d  # noqa: E402
from transitionlistener import pathDeformation as pd  # noqa: E402
from transitionlistener import bubbledynamics as bd  # noqa: E402
from transitionlistener import bubbledynamics_fixedstep as bdf  # noqa: E402


def potential(rate_mode="model", keep=False):
    p = m3d.specific_potential({"g": 0.7, "v_GeV": nf.vev_from_zero_quartic(0.7, 1.0)}, verbose=False)
    conf = p.config.nucleationConf
    conf.rate_mode, conf.mu4_over_T, conf.input_scheme, conf.keep_4d_action = rate_mode, 7.0553, "zero_quartic", keep
    return p


class Phase:
    def __init__(self, x):
        self.x = np.array([float(x)])

    def valAt(self, T):
        return self.x


@pytest.fixture
def no_solver(monkeypatch):
    calls = []

    def refuse(*a, **k):
        calls.append(a)
        raise AssertionError("the 4D bounce was solved")
    monkeypatch.setattr(pd, "fullTunneling", refuse)
    return calls


@pytest.mark.parametrize("module", [bd, bdf])
def test_the_3d_rate_runs_no_bounce_on_either_solver(module, no_solver):
    p = potential()
    T = 0.05 / p.conversionFactor
    out = {}
    S = module.calcAction(p, T, Phase(0.0), Phase(p.v), out)
    assert no_solver == []
    assert S == pytest.approx(p.actionNuc(T, np.nan), rel=1e-12)
    assert out[T]["trantype"] == 1 and out[T]["instanton"] is None and np.isnan(out[T]["action_4d"])


def test_the_4d_rate_has_no_stand_in():
    assert pd.bounceFreeParams(potential("S3overT_4D")) == {}


def test_the_comparison_switch_brings_the_bounce_back():
    assert pd.bounceFreeParams(potential(keep=True)) == {}


def test_an_object_without_the_hook_solves_the_bounce():
    assert pd.bounceFreeParams(object()) == {}
