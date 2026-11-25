"""A run with the 3D rate reports the validity of its 3D theory at the milestone temperature."""
import math

from transitionlistener import config

from transitionlistener.eft_rate import one_field_rate_3d

from transitionlistener.eft_rate import validity
from transitionlistener.interface import samplers


class _Pot:
    def __init__(self, ratios):
        self.ratios = ratios

    def eft_validity_report(self, T):
        return validity.evaluate(self.ratios, order="NLO")


OK = {"hard": 0.3, "hard_fermion": 0.0, "debye": 0.04, "soft_separation": 0.8, "expansion_3d": 0.31, "dim6": 7e-4}


def test_a_trusted_report_sets_no_warning_and_reports_the_ratios():
    out = one_field_rate_3d.validity_diagnostics(_Pot(OK), 1.0)
    assert out["WARNING:eft3d_outside_validity"] is False
    assert out["DIAG:eft3d_expansion_3d"] == 0.31 and out["DIAG:eft3d_hard"] == 0.3 and out["DIAG:eft3d_dim6"] == 7e-4


def test_a_failed_or_missing_check_sets_the_warning():
    for ratios in (dict(OK, hard=2.0), dict(OK, dim6=None)):
        assert one_field_rate_3d.validity_diagnostics(_Pot(ratios), 1.0)["WARNING:eft3d_outside_validity"] is True


def test_a_model_without_the_3d_rate_keeps_the_defaults():
    out = one_field_rate_3d.validity_diagnostics(object(), 1.0)
    assert out["WARNING:eft3d_outside_validity"] is False
    assert all(math.isnan(out[k]) for k in out if k.startswith("DIAG:"))


def test_the_keys_are_registered_where_the_writer_and_failed_rows_need_them():
    empty = samplers.get_empty_result() if callable(getattr(samplers, "get_empty_result", None)) else {}
    for key in one_field_rate_3d.VALIDITY_KEYS:
        assert key in config.all_observables
        assert key in empty
