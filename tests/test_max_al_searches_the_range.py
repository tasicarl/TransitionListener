"""`max_al` must search its range before reporting that the limit is beyond it.

`find_vw` calls `max_al(cb2, cs2, psiN, 100)` and treats the answer as a physical strength:
`alN > max_al` is the runaway test. The old code decided from the top end alone --
`if func(upper_limit) < 0: return upper_limit` -- which is right only while `func` is
monotonic. It is not always.

Measured on the 2HDM line at lambda3 = 5.6868686869, where a one-point dip in the
symmetric-phase sound speed leaves cs2 at 0.30116 instead of ~0.334: `func` is positive at
alN = 1 and negative from alN = 10, so the early return fired, `max_al` came back as the
search bound 100, `alN > max_al` was false for every physical strength, and the wall velocity
was computed as a hybrid 0.7475 instead of the runaway 1 that all its neighbours get. That
moved the LISA SNR from ~0.7 to 11.27, a visible spike in the paper figure.

The inputs below are the measured ones, so these are the real numbers, not a construction.
"""
import unittest


SPIKE = dict(cb2=0.3255800, cs2=0.3011600, psiN=0.9525040)   # lambda3 = 5.6868686869
NEIGHBOUR = dict(cb2=0.3253300, cs2=0.3300800, psiN=0.9523320)  # lambda3 = 5.6843434343
ALN_SPIKE = 0.07473


def _hydro():
    """`max_al` and `find_vJ` use only their arguments, so no potential is needed."""
    from transitionlistener.hydrodynamics import Hydrodynamics
    return Hydrodynamics.__new__(Hydrodynamics)


class MaxAlRangeSearchTests(unittest.TestCase):
    def test_the_root_is_found_and_not_reported_as_the_bound(self):
        value = _hydro().max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 100)
        self.assertLess(value, 1.0, f"max_al returned the search bound, not a root: {value}")
        self.assertAlmostEqual(value, 0.06707, places=4)

    def test_the_runaway_verdict_survives(self):
        """alN = 0.07473 exceeds the real limit, so this point must stay a runaway."""
        value = _hydro().max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 100)
        self.assertGreater(ALN_SPIKE, value,
                           "the runaway test would not fire, and v_wall becomes a hybrid")

    def test_the_answer_does_not_depend_on_the_search_bound(self):
        """A search parameter must not decide a physical verdict."""
        h = _hydro()
        values = [h.max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], ul)
                  for ul in (1.0, 2.0, 5.0, 10.0, 100.0)]
        for v in values[1:]:
            self.assertAlmostEqual(v, values[0], places=6,
                                   msg=f"max_al varies with the bound: {values}")

    def test_the_ordinary_monotonic_case_is_unchanged(self):
        """A neighbour, where `func` is monotonic: the old code already got this right."""
        value = _hydro().max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)
        self.assertAlmostEqual(value, 0.03088, places=4)

    def test_a_range_with_no_crossing_still_reports_the_bound(self):
        """Where subsonic solutions survive the whole range, the bound is the right answer."""
        h = _hydro()
        value = h.max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 0.05)
        self.assertAlmostEqual(value, 0.05, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class BranchAmbiguityIsFlaggedTests(unittest.TestCase):
    """A non-monotonic residual means the branch was chosen, not determined.

    `arxiv:2303.10171` brackets the wall velocity in [0, v_J] because the solution there is
    unique; WallGo (`arxiv:2411.04970`) states the same assumption -- `solveWall` "assumes
    pressure increases with v_w below v_J" -- and notes it can fail for strong transitions,
    flagging such an outcome as uncertain rather than resolving it silently. This follows that
    practice: searching the range recovers the root, and the flag says the structure was not
    the one the method assumes.
    """

    def test_the_non_monotonic_case_is_flagged(self):
        h = _hydro()
        h.max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 100)
        self.assertIs(h.max_al_non_monotonic, True)

    def test_the_ordinary_case_is_not_flagged(self):
        h = _hydro()
        h.max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)
        self.assertIs(h.max_al_non_monotonic, False)

    def test_the_flag_does_not_leak_between_calls(self):
        """One ambiguous point must not mark every later one on the same instance."""
        h = _hydro()
        h.max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 100)
        self.assertIs(h.max_al_non_monotonic, True)
        h.max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)
        self.assertIs(h.max_al_non_monotonic, False, "a stale flag leaked into the next call")

    def test_the_flag_is_a_registered_observable(self):
        from transitionlistener import config
        from transitionlistener.interface import samplers, output_schema
        self.assertIn("WARNING:v_wall_branch_ambiguous", config.all_observables)
        self.assertIn("WARNING:v_wall_branch_ambiguous", samplers.get_empty_result())
        self.assertIn("WARNING:v_wall_branch_ambiguous", output_schema.WARNING_COLUMN_ORDER)
