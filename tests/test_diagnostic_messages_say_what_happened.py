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
import types
import unittest

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

    def test_the_check_consults_the_skip_set(self):
        for module in (tobs, tobs_fixed):
            source = inspect.getsource(module)
            self.assertIn("if name in _FILLED_BY_THE_PIPELINE:", source,
                          f"{module.__name__} does not consult the skip set, so it will still "
                          "report a pipeline-filled name as unimplemented")


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
            if name == "TooMuchSupercoolingError":
                bare.append(node.lineno)          # the class itself, no call
            elif isinstance(exc, _ast.Call):
                fn = exc.func
                fname = (fn.attr if isinstance(fn, _ast.Attribute)
                         else fn.id if isinstance(fn, _ast.Name) else None)
                if fname == "TooMuchSupercoolingError" and not exc.args and not exc.keywords:
                    bare.append(node.lineno)
        self.assertEqual(bare, [],
                         f"raises at lines {bare} rely on the class default, which claims the "
                         "transition is too supercooled to nucleate")


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
        self.assertIn("does not exist to as high a temperature", printed)
        self.assertIn("no window to tunnel from", printed)
        self.assertNotIn("exists to at least as high a", printed,
                         "the old wording contradicted the condition above it")

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
        self.assertNotIn("does not exist to as high a temperature", printed)


if __name__ == "__main__":
    unittest.main()


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
        with self.assertRaises(transitions.errors.TooMuchSupercoolingError) as caught:
            self._walk()
        message = " ".join(str(caught.exception).split())
        self.assertIn("2 transition(s) had already been found", message,
                      "the walk found two transitions before the cycle closed and must say so")
        self.assertIn("zero-temperature phase", message,
                      "the final phase is the T0 phase, which is why the chain had ended")
        self.assertIn("phases may not overlap", message,
                      "the reason must stay hedged; this is not necessarily supercooling")

    def test_the_error_does_not_claim_supercooling_on_its_own(self):
        """The class default claims one bubble per Hubble volume was never reached."""
        with self.assertRaises(transitions.errors.TooMuchSupercoolingError) as caught:
            self._walk()
        self.assertNotIn("until today", str(caught.exception),
                         "the class default is a claim about the physics, not about this walk")
