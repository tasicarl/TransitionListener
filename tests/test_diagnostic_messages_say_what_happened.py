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
        # TL wraps stdout at about 75 columns, so the message arrives with newlines inside
        # it. Collapse the whitespace before matching on the wording.
        return result, " ".join(buffer.getvalue().split())

    def test_the_message_says_the_starting_phase_stops_at_a_lower_temperature(self):
        result, printed = self._run(high_tmax=121.3, low_tmax=615.5)
        self.assertIsNone(result, "this branch rejects the candidate")
        self.assertIn("does not exist to as high a temperature", printed)
        self.assertIn("no window to tunnel from", printed)
        self.assertNotIn("exists to at least as high a", printed,
                         "the old wording contradicted the condition above it")

    def test_the_message_is_not_printed_when_the_starting_phase_reaches_higher(self):
        """Pins the message to its condition: swap the two Tmax values and it must not fire."""
        _result, printed = self._run(high_tmax=615.5, low_tmax=121.3)
        self.assertNotIn("does not exist to as high a temperature", printed)


if __name__ == "__main__":
    unittest.main()
