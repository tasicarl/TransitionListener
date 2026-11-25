"""g3^2 of the 3D theory is matched at the order of the masses (nucleationConf.matching_order).

At NLO, g3^2 = g^2 T (1 - g^2 (L_b + n L_f) / (48 pi^2)) with n = 1 for the model with fermions and 0 without
(dr_conformal_u1_*_matching.g3_sq_NLO). At mu4 = 4 pi e^-gamma_E T, L_b = 0 and L_f = 4 ln 2, so only the fermion loop is
left. A grid records the order it was built with; a grid that does not record it was built with g3^2 at LO.
"""
import math

import pytest

from transitionlistener.eft_rate import nlo_tables as nt

nt.load_model("TL_conformal_u1_no_fermions_3d")    # puts models/ on the path
import dr_conformal_u1_fermions_matching as fermions        # noqa: E402
import dr_conformal_u1_no_fermions_matching as no_fermions  # noqa: E402

MU4 = 4.0 * math.pi * math.exp(-0.5772156649015329)
G, Y, MU0, T = 0.7, 0.5, 1.0, 0.1


@pytest.mark.parametrize("order", ["NLO", "LO"])
def test_the_matched_g3sq_is_that_of_the_order(order):
    p = fermions.matched_soft_parameters_zero_quartic(G, Y, MU0, T, mu4_over_T=MU4, matching_order=order)
    g = p["g4"]
    lo = g * g * T
    want = lo * (1.0 - g * g * 4.0 * math.log(2.0) / (48.0 * math.pi ** 2)) if order == "NLO" else lo
    assert p["g3sq"] == pytest.approx(want, rel=1e-12)
    q = no_fermions.matched_soft_parameters(G, 0.0, MU0, T, mu4_over_T=MU4, matching_order=order)
    assert q["g3sq"] == pytest.approx(q["g4"] ** 2 * T, rel=1e-12)    # L_b = 0 here: no correction without fermions


def test_without_fermions_the_correction_is_the_boson_log():
    q = no_fermions.matched_soft_parameters(G, 0.0, MU0, T, mu4_over_T=2.0 * MU4, matching_order="NLO")
    g = q["g4"]
    assert q["g3sq"] == pytest.approx(g * g * T * (1.0 - g * g * 2.0 * math.log(2.0) / (48.0 * math.pi ** 2)), rel=1e-12)


class _Grid:
    factors = [1.0]

    def __init__(self, header):
        self.header = header

    def describe(self):
        return "grid"


@pytest.mark.parametrize("recorded,run,refused", [(None, "NLO", True), ("NLO", "NLO", False), ("LO", "NLO", True)])
def test_a_grid_of_the_other_g3sq_order_is_refused(recorded, run, refused):
    header = dict(mu3_choice="gT", input_scheme="zero_quartic", matching_order=run, mu4_over_T=7.0553,
                  temporal_coupling="h3")
    if recorded is not None:
        header["g3sq_order"] = recorded
    conf = type("C", (), dict(mu3_choice="gT", input_scheme="zero_quartic", matching_order=run, mu4_over_T=7.0553,
                              temporal_coupling="h3"))()
    if refused:
        with pytest.raises(nt.NLOGridError, match="g3\\^2 at LO"):
            nt._check_conventions(_Grid(header), conf)
    else:
        assert nt._check_conventions(_Grid(header), conf) == 1.0
