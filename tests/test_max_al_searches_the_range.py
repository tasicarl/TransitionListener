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
moved the LISA SNR from 0.8076 to 11.269, a factor 14 and a visible spike in the paper
figure.

The inputs below are the measured ones, so these are the real numbers, not a construction.
"""
import unittest

from transitionlistener import errors


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

    def test_a_range_with_no_subsonic_branch_reports_the_bound_instead_of_raising(self):
        """The other no-crossing case, where the old code died with an untyped error.

        With the broken phase softer than the symmetric one the residual can stay positive
        across the whole range, so there is no subsonic branch to bound. The old code then
        handed `root_scalar` a bracket with equal signs at both ends and raised a bare
        `ValueError: f(a) and f(b) must have different signs`, which nothing upstream types,
        so the point was reported as error -1. Sweeping 15 x 15 x 11 triples with `c_b^2`
        and `c_s^2` in [0.05, 1/3] and `psi_N` in [0.9, 1.1], 895 of the 2475 have a finite
        non-negative residual at both ends of the old bracket, which is the condition that
        made `root_scalar` raise; 227 of those have `psi_N` below one, and this is one.

        Reporting the bound leaves the verdict to `find_vw`, which still decides between the
        subsonic solve and the runaway branch on its own `shooting(v_J) < 0` test. The
        branch flag stays down, because no crossing is absence of a subsonic branch rather
        than a residual with more than one.
        """
        h = _hydro()
        value = h.max_al(0.05, 0.131, 0.905, 100)
        self.assertAlmostEqual(value, 100.0, places=9)
        self.assertIs(h.max_al_non_monotonic, False)


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


class DegenerateRangesAndGapsTests(unittest.TestCase):
    """The grid the search builds has to exist, and a hole in it must not decide anything.

    Two situations the measured fixtures above cannot reach. `psiN` arrives from
    `w_broken / w_symmetric`, so it is one wherever the two enthalpies agree and the lower
    end of the range is then exactly zero, which `numpy.geomspace` refuses. And a residual
    that cannot be evaluated at some strengths leaves a hole: `find_vJ` takes a square root
    of `3 alN (1 - c_b^2 + 3 c_b^2 alN)` and the matching conditions divide by quantities
    that can vanish, so the hole is a real possibility rather than a hypothetical one. It
    is produced here by making `find_vJ` refuse, over a band of `alN` or on one chosen
    evaluation, because no combination of the three physical arguments was found that
    produces one: a sweep of
    2475 triples over `c_b^2`, `c_s^2` in [0.05, 1/3] and eleven values of `psiN` gave no
    grid with both a non-finite residual and a crossing.
    """

    @staticmethod
    def _refusing_band(hydro, low, high):
        """Make the residual non-evaluable for `low < alN < high`."""
        original = hydro.find_vJ

        def find_vJ(alN, cb2):
            if low < float(alN) < high:
                raise ValueError("no Jouguet velocity here")
            return original(alN, cb2)

        hydro.find_vJ = find_vJ

    @staticmethod
    def _failing_sample(hydro, call, error):
        """Make the `call`-th evaluation of the residual raise `error`."""
        original = hydro.find_vJ
        state = {"calls": 0}

        def find_vJ(alN, cb2):
            state["calls"] += 1
            if state["calls"] == call:
                raise error
            return original(alN, cb2)

        hydro.find_vJ = find_vJ

    def test_an_enthalpy_ratio_of_one_still_searches(self):
        """psiN = 1 puts the lower end of the range at exactly zero."""
        h = _hydro()
        value = h.max_al(0.33, 0.33, 1.0, 100)
        self.assertTrue(value == value and abs(value) < float("inf"),
                        f"max_al returned {value} where the lower limit is zero")
        self.assertIs(h.max_al_non_monotonic, False)

    def test_a_hole_in_the_residual_does_not_move_the_root(self):
        """The crossing is outside the hole, so the answer must be the measured one."""
        h = _hydro()
        self._refusing_band(h, 0.0175, 0.0215)
        value = h.max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)
        self.assertAlmostEqual(value, 0.03088, places=4)

    def test_a_hole_in_the_residual_is_not_counted_as_a_crossing(self):
        """A gap is absence of information, not a sign change.

        `nan < 0.0` is False, so a hole inside the negative stretch looks like two sign
        changes to any test that does not drop it first, and this monotonic point would be
        reported as branch-ambiguous on the strength of a missing sample.
        """
        h = _hydro()
        self._refusing_band(h, 0.0175, 0.0215)
        h.max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)
        self.assertIs(h.max_al_non_monotonic, False,
                      "a gap in the residual was counted as non-monotonicity")

    def test_a_hole_between_opposite_signs_is_not_counted_as_a_crossing(self):
        """The other hole: one that separates a negative residual from a positive one.

        The SPIKE residual changes sign exactly twice on the search grid, between
        alN = 0.0635 and 0.0730 and between alN = 6.216 and 7.142, which is why it is
        flagged. Refusing the band around the second one removes both of its samples, so no
        bracket spans that stretch any more and the search in `max_al` would not use it.
        Dropping the holes and comparing what is left instead makes the surviving `+` at
        alN = 5.42 and `-` at alN = 8.20 neighbours, and the point stays flagged on the
        strength of two samples that never touch. The count has to agree with the search.

        The first crossing is untouched, so the root must still be the measured one.
        """
        h = _hydro()
        self._refusing_band(h, 6.0, 7.5)
        value = h.max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 100)
        self.assertAlmostEqual(value, 0.06707, places=4,
                               msg="the hole moved the root it does not span")
        self.assertIs(h.max_al_non_monotonic, False,
                      "a hole between opposite signs was counted as a crossing")

    def test_the_same_point_is_flagged_when_that_crossing_is_visible(self):
        """Without the hole the second crossing is real evidence, so the flag must fire.

        Otherwise the test above would pass on a flag that is never set at all.
        """
        h = _hydro()
        h.max_al(SPIKE["cb2"], SPIKE["cs2"], SPIKE["psiN"], 100)
        self.assertIs(h.max_al_non_monotonic, True)

    def test_a_timeout_while_sampling_is_not_turned_into_a_hole(self):
        """`errors.Timeout` subclasses `Exception`, so the blanket handler would absorb it."""
        h = _hydro()
        self._failing_sample(h, 5, errors.Timeout("deadline reached while sampling"))
        with self.assertRaises(errors.Timeout):
            h.max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)

    def test_an_ordinary_failure_at_the_same_sample_is_absorbed(self):
        """The counterpart: same sample, a numerical failure instead of a deadline.

        That one is absorbed, and the sample it costs is the lower end of the only bracket
        NEIGHBOUR has: alN = 0.027689, against the crossing at 0.030878. No bracket spans
        the crossing any more, so the search reports its bound and leaves the verdict to
        `find_vw` and its own `shooting(v_J) < 0` test. Nothing reaches the caller.

        This is also what an absorbed deadline would return, and why absorbing one is worse
        than leaving a hole: `max_al = 100` is not a missing answer but a wrong one, and
        `alN > max_al` is then false for every physical strength -- the measured failure in
        this file's opening paragraph, reached a second way.
        """
        h = _hydro()
        self._failing_sample(h, 5, ValueError("no Jouguet velocity here"))
        value = h.max_al(NEIGHBOUR["cb2"], NEIGHBOUR["cs2"], NEIGHBOUR["psiN"], 100)
        self.assertEqual(float(value), 100.0,
                         "an ordinary failure at the bracket did not fall back to the bound")
        self.assertIs(h.max_al_non_monotonic, False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
