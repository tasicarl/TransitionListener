"""Diagnostics must not claim something the run did not do.

Two unrelated messages said the wrong thing on every run that reached them.

`DIAG:runtime_s` is in the output schema, so the completeness check in
`computeTransitionObservables` looks for it, but `interface/pipeline.py` assigns it afterwards,
once the elapsed time is known. The check could therefore never find it and every completed run
printed "It might be that this variable still needs to be implemented" about a column that is
implemented and correct in the same output file.

`transitions.py` raised `TooMuchSupercoolingError` as a bare class when no candidate transition
below the current phase was viable, so the user was shown the class default, "too supercooled
to allow for the nucleation of one bubble per Hubble volume until today". On the 2HDM benchmark
at `lambda3` = 5.8888889 to 5.9015152 that was not supercooling: the candidate pair was
rejected because the starting phase is higher in potential at `tmax` and exists only to
121.3 GeV against the target's 615.5 GeV.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import math
import types
import unittest

import numpy as np
from unittest import mock

from transitionlistener import transitions
from transitionlistener import transitionObservables as tobs
from transitionlistener import transitionObservables_fixedstep as tobs_fixed
from transitionlistener.interface import output_schema


class PipelineFilledNamesAreNotReportedMissingTests(unittest.TestCase):
    def test_the_runtime_diagnostic_is_in_the_schema(self):
        """If it were not, the completeness check would never look for it and this is moot."""
        self.assertIn("DIAG:runtime_s", set(output_schema.DIAGNOSTIC_COLUMN_ORDER))

    def test_both_backends_skip_the_same_names(self):
        """The two modules carry one copy of this loop each; a fix to one must reach the other."""
        self.assertEqual(tobs._FILLED_BY_THE_PIPELINE, tobs_fixed._FILLED_BY_THE_PIPELINE)

    def test_the_runtime_diagnostic_is_skipped(self):
        self.assertIn("DIAG:runtime_s", tobs._FILLED_BY_THE_PIPELINE)
        self.assertIn("DIAG:runtime_s", tobs_fixed._FILLED_BY_THE_PIPELINE)

    def test_the_loop_warns_selectively_and_fills_unconditionally(self):
        """Both halves of the completeness check, driven rather than grepped.

        `_finalize_results` reads only `ctx.derived_params` and `ctx.derived_param_names` and
        touches no instance state, so it runs unbound. Every other branch in it is guarded by a
        name being in `derived_param_names`, and neither name below is one of those, so what
        executes is the completeness loop.

        The placeholder half matters as much as the warning half: `printDerivedParams` indexes
        every configured name directly, and `interface/pipeline.py` assigns `DIAG:runtime_s`
        only to `strongestTransitionObservables`, so a per-transition dictionary missing the key
        raises `KeyError` when the run prints its table. The first version of this change
        skipped the assignment along with the message and would have crashed there.
        """
        for module in (tobs, tobs_fixed):
            with self.subTest(module=module.__name__):
                names = ["DIAG:runtime_s", "DIAG:not_implemented_anywhere"]
                ctx = types.SimpleNamespace(derived_params={}, derived_param_names=names)
                buffer = io.StringIO()
                with contextlib.redirect_stdout(buffer):
                    derived = module.TransitionObservables._finalize_results(None, ctx)
                printed = " ".join(buffer.getvalue().split())

                for name in names:
                    self.assertIn(name, derived,
                                  "printDerivedParams indexes this name directly")
                    self.assertTrue(math.isnan(derived[name]))
                self.assertNotIn("DIAG:runtime_s not found", printed,
                                 "the pipeline fills this one later; no warning is due")
                self.assertIn("DIAG:not_implemented_anywhere not found", printed,
                              "a genuinely missing name must still be reported")


class SupercoolingErrorNamesItsReasonTests(unittest.TestCase):
    def test_no_raise_relies_on_the_class_default(self):
        """Both the bare class and empty parentheses show the default, which asserts
        supercooling. Checked on the parse tree, because a string search for one spelling
        misses the other, which is how the first version of this test passed while a
        message-less raise was still in the file."""
        import ast as _ast
        tree = _ast.parse(inspect.getsource(transitions))
        bare = []
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.Raise) or node.exc is None:
                continue
            exc = node.exc
            name = (exc.attr if isinstance(exc, _ast.Attribute)
                    else exc.id if isinstance(exc, _ast.Name) else None)
            if name in ("TooMuchSupercoolingError", "NoViablePhasePairError"):
                bare.append(node.lineno)          # the class itself, no call
            elif isinstance(exc, _ast.Call):
                fn = exc.func
                fname = (fn.attr if isinstance(fn, _ast.Attribute)
                         else fn.id if isinstance(fn, _ast.Name) else None)
                if (fname in ("TooMuchSupercoolingError", "NoViablePhasePairError")
                        and not exc.args and not exc.keywords):
                    bare.append(node.lineno)
        self.assertEqual(bare, [],
                         f"raises at lines {bare} rely on the class default, which claims the "
                         "transition is too supercooled to nucleate")


class ChildPhaseCountTests(unittest.TestCase):
    """The count must be taken before the helper empties the list.

    `getUniqueTunnelingPhases` empties the list it is given: its loop is
    `while childPhases != []: p1 = childPhases.pop()`. The walk hands it
    `startPhase.childPhases` itself, so the phase's child list comes back empty and a count
    taken after the call reads zero. The count is therefore taken before it.

    The consumption is left in place, because it is what terminates this loop on a cyclic
    phase graph: a revisited phase has an empty child list and leaves through the
    `startPhase.childPhases == []` exit. Handing over a copy removes that guard, which needs
    an explicit replacement and does not belong in a change about diagnostics.

    Every other test in this file stubs `getUniqueTunnelingPhases`, which is why none of them
    saw the count was wrong. This one calls the real method.
    """

    @staticmethod
    def _phase(key, value):
        return types.SimpleNamespace(
            key=key, X=np.array([[value], [value]], dtype=float),
            T=np.array([10.0, 200.0]))

    def test_the_real_method_consumes_the_list_it_is_given(self):
        """Pinned as a property of that method, so the walk's copy is not cargo cult."""
        phases = {"a": self._phase("a", 1.0), "b": self._phase("b", 2.0),
                  "root": self._phase("root", 0.0)}
        stub = types.SimpleNamespace()
        children = ["a", "b"]
        with contextlib.redirect_stdout(io.StringIO()):
            transitions.Transitions.getUniqueTunnelingPhases(
                stub, lambda x, T: float(np.squeeze(x)), phases, "root", children)
        self.assertEqual(children, [],
                         "if this ever stops emptying its argument, the copy in the walk can "
                         "go, but until then it is load-bearing")

    def test_the_count_survives_the_helper_emptying_the_list(self):
        """The count is taken first, so it describes the graph that was encountered."""
        phases = _Phases({
            "1": types.SimpleNamespace(key="1", Tmax=615.5, Tmin=300.0, childPhases=["0"],
                                       low_trans={}, valAt=lambda T: 2.0,
                                       X=np.array([[2.0], [2.0]]), T=np.array([300.0, 615.5])),
            "0": types.SimpleNamespace(key="0", Tmax=200.0, Tmin=0.0, childPhases=[],
                                       low_trans={}, valAt=lambda T: 1.0,
                                       X=np.array([[1.0], [1.0]]), T=np.array([0.0, 200.0])),
        })
        stub = types.SimpleNamespace(
            phases=phases, V=lambda x, T: float(np.squeeze(x)), conversionFactor=1.0,
            tunnelFromPhase_args={},
            pot=types.SimpleNamespace(config=types.SimpleNamespace(
                percolationConf=types.SimpleNamespace(algorithm_mode="adaptive_step_size"))))
        stub.getUniqueTunnelingPhases = (
            transitions.Transitions.getUniqueTunnelingPhases.__get__(stub))
        stub.seedlessFirstOrderCandidate = (
            transitions.Transitions.seedlessFirstOrderCandidate.__get__(stub))
        with self.assertRaises(transitions.errors.NoViablePhasePairError) as caught:
            with contextlib.redirect_stdout(io.StringIO()):
                transitions.Transitions.findTransitionPathFromTnuc(stub, verbose=False)
        message = " ".join(str(caught.exception).split())
        self.assertIn("1 child phase(s) listed below phase 1", message,
                      "counted before the helper emptied the list, so not zero")
        # This fixture's phases never coexist: phase 1 exists only above 300 GeV and phase 0
        # only below 200, so the `tmin >= tmax` branch skips the pair without evaluating it.
        self.assertIn("0 evaluated as a candidate", message)
        self.assertEqual(phases["1"].childPhases, [],
                         "the helper consumed the list, which is the behaviour this branch "
                         "leaves alone")


class SeedlessSkipMessageTests(unittest.TestCase):
    """Drive the branch and read what it prints, rather than grepping the source.

    The branch tests ``high_phase.Tmax <= low_phase.Tmax``, so the starting phase exists to a
    LOWER temperature than the target. The message used to say the opposite, that the starting
    phase "exists to at least as high a temperature". Asserting only that the old wording is
    absent would pass if the message were deleted or replaced by a different wrong one, which
    is what this version fixes.
    """

    @staticmethod
    def _phase(tmax, value):
        return types.SimpleNamespace(Tmax=tmax, key="p", valAt=lambda T, v=value: v)

    def _run(self, high_tmax, low_tmax):
        """Reach the third rejection branch: dv_min >= 0 and dv_max > 0.

        `V` returns the field value itself, so the potential difference is
        ``high - low = 1.0`` at every temperature, which is positive at both ends. The method
        is called unbound; the only attribute it reads off `self` past the branch under test is
        `conversionFactor`, used to print the reference temperature in GeV.
        """
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            result = transitions.Transitions.seedlessFirstOrderCandidate(
                types.SimpleNamespace(conversionFactor=1.0),
                lambda x, T: float(x),
                self._phase(high_tmax, 2.0),
                self._phase(low_tmax, 1.0),
                10.0, 100.0, verbose=True,
            )
        # `transitionlistener.print` is `rich.console.Console.print`, which word-wraps to the
        # console width, 80 columns when stdout is not a terminal as it is here. The message
        # therefore arrives with newlines inside it, so collapse the whitespace before
        # matching on the wording.
        return result, " ".join(buffer.getvalue().split())

    def test_the_message_says_the_starting_phase_stops_at_a_lower_temperature(self):
        result, printed = self._run(high_tmax=121.3, low_tmax=615.5)
        self.assertIsNone(result, "this branch rejects the candidate")
        self.assertIn("does not exist to a higher temperature", printed)
        self.assertIn("no window to tunnel from", printed)
        self.assertNotIn("exists to at least as high a", printed,
                         "the old wording contradicted the condition above it")

    def test_the_message_covers_two_phases_ending_at_the_same_temperature(self):
        """The branch tests `<=`, so equal Tmax takes it too.

        "does not exist to as high a temperature" was false in that case: it exists to exactly
        as high a one. The wording now says "to a higher temperature than", which is what the
        condition actually means.
        """
        result, printed = self._run(high_tmax=300.0, low_tmax=300.0)
        self.assertIsNone(result, "equal Tmax is still refused by this branch")
        self.assertIn("does not exist to a higher temperature", printed)
        self.assertIn("same temperature", printed,
                      "the equal case is worth naming, since it reads as a contradiction")

    def test_the_message_is_not_printed_when_the_starting_phase_reaches_higher(self):
        """Pins the message to its condition: swap the two Tmax values and it must not fire.

        The absence of the message is only evidence if the swapped case still walks past the
        branch, so the candidate it then builds and the line it prints instead are asserted
        too. Without them the test would pass just as happily on an earlier `return None`.
        """
        result, printed = self._run(high_tmax=615.5, low_tmax=121.3)
        self.assertIsNotNone(result, "the swapped case must reach the end and build the "
                                     "candidate, or the branch under test was never passed")
        self.assertIn("Adding seedless first-order candidate", printed)
        self.assertNotIn("does not exist to a higher temperature", printed)



class _Phases(dict):
    """The phase mapping the walk is handed, with the two keys it reads off it."""

    rootPhase = "1"
    T0Phase = "0"


class CyclicPhaseGraphTests(unittest.TestCase):
    """The chain walk must say how many transitions it is throwing away.

    Measured on the 2HDM at `lambda3` = 5.8888888889, 5.8939393939 and 5.9015151515, the phase
    graph contains a cycle: the zero-temperature phase lists the high-temperature root phase
    among its children. The walk therefore reaches the end of the chain, asks the final phase
    for one more transition, gets the cycle's closing link, and `seedlessFirstOrderCandidate`
    rightly refuses it because the starting phase reaches 121.3 GeV against the target's
    615.5 GeV. `childTransitions` is then empty and the walk raises, discarding the two
    transitions it had already found. All three couplings report CLI error 8 and no transitions.

    The error is not wrong to be raised here -- deciding whether the chain should instead end
    successfully is a change to the walk and not to a message -- but it must not hide that it
    found a chain and dropped it. This pins that.

    The fixture accepts both of its first two steps through the `high_phase.Tmax >
    low_phase.Tmax` branch, which is the simplest way to reach the third step; on the benchmark
    the second step is accepted through the critical-temperature branch instead. The claim here
    is about the walk's bookkeeping, not about which branch accepts a pair.
    """

    @staticmethod
    def _phase(key, tmax, tmin, children, value):
        return types.SimpleNamespace(key=key, Tmax=tmax, Tmin=tmin, childPhases=children,
                                     low_trans={}, valAt=lambda T, v=value: v)

    def _walk(self):
        phases = _Phases({
            "1": self._phase("1", 615.5, 50.0, ["2"], 3.0),
            "2": self._phase("2", 300.0, 40.0, ["0"], 2.0),
            "0": self._phase("0", 200.0, 0.0, ["1"], 1.0),
        })
        stub = types.SimpleNamespace(
            phases=phases,
            V=lambda x, T: float(x),
            conversionFactor=1.0,
            pot=types.SimpleNamespace(
                config=types.SimpleNamespace(
                    percolationConf=types.SimpleNamespace(
                        algorithm_mode="adaptive_step_size"))),
            getUniqueTunnelingPhases=lambda V, phases, startPhase, childPhases: childPhases,
            # read once at the top of the walk and only used on the non-seedless path
            tunnelFromPhase_args={},
        )
        stub.seedlessFirstOrderCandidate = (
            transitions.Transitions.seedlessFirstOrderCandidate.__get__(stub))
        with contextlib.redirect_stdout(io.StringIO()):
            return transitions.Transitions.findTransitionPathFromTnuc(stub, verbose=False)

    def test_the_error_says_how_many_transitions_it_discards(self):
        with self.assertRaises(transitions.errors.NoViablePhasePairError) as caught:
            self._walk()
        message = " ".join(str(caught.exception).split())
        self.assertIn("2 transition(s) had already been found", message,
                      "the walk found two transitions before the cycle closed and must say so")
        self.assertIn("1 child phase(s) listed below phase 0, 1 evaluated as a candidate",
                      message,
                      "the message must say what was offered and what was actually tried")
        self.assertIn("0 of those ended in a nucleation solve that did not succeed", message,
                      "a candidate can also disappear because the solver failed, which is "
                      "neither a rejection nor an overlap problem")
        self.assertIn("nothing about the rate was measured at this step", message,
                      "no candidate reached a nucleation solve below this phase")
        self.assertIn("The claim is about this step alone", message,
                      "the claim is about one step of the chain and has to say so")
        self.assertIn("counted below", message,
                      "the transitions found earlier are reported, not disowned")
        for overclaim in ("solved normally", "were solved", "solved earlier"):
            self.assertNotIn(
                overclaim, message,
                f"{overclaim!r} claims a nucleation solve succeeded for the earlier steps. "
                "A second-order step appends a transition with no solve at all "
                "(transitions.py, the low_trans branch), and in the adaptive backend "
                "seedlessFirstOrderCandidate builds a candidate without computing Tnuc, so "
                "nothing below this phase had been solved when this error was raised")
        self.assertIn("below phase 0", message,
                      "the phase the claim is scoped to has to be named in it")
        self.assertNotIn("every candidate was rejected", message,
                         "a child can be skipped without ever being evaluated: "
                         "getUniqueTunnelingPhases drops duplicates and the tmin >= tmax "
                         "branch skips a pair with no temperature overlap")

    def test_a_candidate_that_reached_the_solver_is_code_8_not_code_19(self):
        """The discriminator between the two errors, which is the point of having two.

        A pair that never reaches a nucleation solve says nothing about the rate, so it raises
        `NoViablePhasePairError`, code 19. A pair that does reach one and fails it has had the
        reached the solver: that is `TooMuchSupercoolingError`, code 8. Code 8 does not claim
        the transition is supercooled, only that a solve was attempted and did not succeed,
        which may be supercooling or a numerical failure, and the message says so.

        Driven on the fixed-step path, where `computeNucleationTemperature` is what raises
        `NucleationError` and the candidate is dropped.
        """
        phases = _Phases({
            "1": self._phase("1", 615.5, 50.0, ["0"], 3.0),
            "0": self._phase("0", 200.0, 0.0, [], 1.0),
        })
        stub = types.SimpleNamespace(
            phases=phases, V=lambda x, T: float(x), dV=None, d2V=None,
            conversionFactor=1.0, tunnelFromPhase_args={},
            pot=types.SimpleNamespace(config=types.SimpleNamespace(
                percolationConf=types.SimpleNamespace(algorithm_mode="fixed_step_size"))),
            getUniqueTunnelingPhases=lambda V, p, sp, cp: cp)

        def failing_solve(*a, **k):
            raise transitions.errors.NucleationError("no finite nucleation criterion")

        with mock.patch.object(transitions, "computeNucleationTemperature", failing_solve):
            with self.assertRaises(transitions.errors.TooMuchSupercoolingError) as caught:
                with contextlib.redirect_stdout(io.StringIO()):
                    transitions.Transitions.findTransitionPathFromTnuc(stub, verbose=False)
        message = " ".join(str(caught.exception).split())
        self.assertEqual(caught.exception.errorcode, 8)
        self.assertIn("1 of those ended in a nucleation solve that did not succeed", message)
        self.assertIn("failed numerically", message,
                      "the solve was attempted, so a numerical failure is a live possibility")
        self.assertNotIsInstance(caught.exception, transitions.errors.NoViablePhasePairError)

    def test_a_child_skipped_without_evaluation_is_counted_as_such(self):
        """The case that made "every candidate was rejected" false.

        A child phase is never turned into a candidate when the two phases do not overlap in
        temperature: the `tmin >= tmax` branch skips it without calling
        `seedlessFirstOrderCandidate`, so nothing is rejected and verbose mode shows no
        rejection either. `getUniqueTunnelingPhases` dropping a duplicate child has the same
        effect. The message must not claim a rejection that never happened.
        """
        phases = _Phases({
            "1": self._phase("1", 615.5, 300.0, ["0"], 2.0),   # exists only above 300
            "0": self._phase("0", 200.0, 0.0, [], 1.0),        # exists only below 200
        })
        stub = types.SimpleNamespace(
            phases=phases, V=lambda x, T: float(x), conversionFactor=1.0,
            tunnelFromPhase_args={},
            pot=types.SimpleNamespace(config=types.SimpleNamespace(
                percolationConf=types.SimpleNamespace(algorithm_mode="adaptive_step_size"))),
            getUniqueTunnelingPhases=lambda V, p, sp, cp: cp)
        stub.seedlessFirstOrderCandidate = (
            transitions.Transitions.seedlessFirstOrderCandidate.__get__(stub))
        with self.assertRaises(transitions.errors.NoViablePhasePairError) as caught:
            with contextlib.redirect_stdout(io.StringIO()):
                transitions.Transitions.findTransitionPathFromTnuc(stub, verbose=False)
        message = " ".join(str(caught.exception).split())
        self.assertIn("1 child phase(s) listed below phase 1, 0 evaluated as a candidate",
                      message,
                      "the pair has no temperature overlap, so nothing was evaluated")
        self.assertNotIn("every candidate was rejected", message)

    def test_the_error_does_not_claim_supercooling_on_its_own(self):
        """Not merely a better message: a different error, carrying CLI code 19.

        Supercooling is a claim about the physics, that the rate never reached one bubble per
        Hubble volume. It can only be made where a nucleation solve was attempted. These
        phases never produced a candidate, so nothing about the rate was measured.
        """
        with self.assertRaises(transitions.errors.NoViablePhasePairError) as caught:
            self._walk()
        self.assertNotIn("until today", str(caught.exception),
                         "the class default is a claim about the physics, not about this walk")
        self.assertEqual(caught.exception.errorcode, 19)
        self.assertNotIsInstance(caught.exception, transitions.errors.TooMuchSupercoolingError,
                                 "no nucleation solve was attempted, so supercooling is not "
                                 "something this run can report")


if __name__ == "__main__":
    unittest.main()
