"""The nucleation line belongs on the panel whenever the run found a nucleation temperature.

`plotPercolation` read the nucleation temperature off the transition object alone. That
object does not always carry one: the search that runs before percolation gives up on a
strongly supercooled point and leaves `Tnuc` as None, and the percolation stage afterwards
still solves

    N(T) = 4 pi / 3 v_w^3 int Gamma / H^4 (1 - P) dT / T = 1

and writes `Tnuc_SM_GeV` into the observables. The shipped conformal benchmark is such a
point: its run records a nucleation temperature of 2.5695e-3 GeV against a percolation
temperature of 2.2120e-3 GeV, and the panel drew no nucleation line at all.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import unittest
from types import SimpleNamespace

import numpy as np


def _plotter(derived, conversion_factor=1.0):
    """A plotter with nothing on it but what the temperature lookup reads."""
    from transitionlistener import plots

    plotter = plots.TLPlots.__new__(plots.TLPlots)
    plotter.pot = SimpleNamespace(conversionFactor=conversion_factor)
    plotter.derived_params_dict = derived
    return plotter


class TheReportedNucleationTemperatureTests(unittest.TestCase):

    def test_the_transition_object_is_used_where_it_has_one(self):
        plotter = _plotter({0: {"Tnuc_SM_GeV": 99.0}})
        tr = SimpleNamespace(Tnuc=41.4)
        self.assertEqual(41.4, plotter._reported_nucleation_temperature(0, tr))

    def test_the_observables_are_used_where_the_transition_object_has_none(self):
        """The conformal benchmark: no `Tnuc` on the transition, one in the observables."""
        plotter = _plotter({0: {"Tnuc_SM_GeV": 2.5695e-3}}, conversion_factor=1.0)
        tr = SimpleNamespace(Tnuc=None)
        self.assertAlmostEqual(2.5695e-3,
                               plotter._reported_nucleation_temperature(0, tr))

    def test_the_observable_is_converted_out_of_GeV(self):
        """`Tnuc_SM_GeV` is in GeV and the panel works in the model's internal units."""
        plotter = _plotter({0: {"Tnuc_SM_GeV": 50.0}}, conversion_factor=2.0)
        tr = SimpleNamespace(Tnuc=None)
        self.assertAlmostEqual(25.0, plotter._reported_nucleation_temperature(0, tr))

    def test_a_not_a_number_on_the_transition_falls_through(self):
        plotter = _plotter({0: {"Tnuc_SM_GeV": 7.0}})
        tr = SimpleNamespace(Tnuc=np.nan)
        self.assertAlmostEqual(7.0, plotter._reported_nucleation_temperature(0, tr))

    def test_neither_source_means_no_line(self):
        """A point that never nucleates has to leave the line off rather than invent one."""
        plotter = _plotter({0: {"Tnuc_SM_GeV": np.nan}})
        tr = SimpleNamespace(Tnuc=None)
        self.assertIsNone(plotter._reported_nucleation_temperature(0, tr))

    def test_no_observables_at_all_means_no_line(self):
        plotter = _plotter({})
        tr = SimpleNamespace(Tnuc=None)
        self.assertIsNone(plotter._reported_nucleation_temperature(0, tr))


if __name__ == "__main__":
    unittest.main()
