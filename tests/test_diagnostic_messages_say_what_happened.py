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

import inspect
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

    def test_the_seedless_skip_message_matches_its_condition(self):
        """The branch tests high.Tmax <= low.Tmax, so the starting phase exists to a LOWER
        temperature. The message used to say the opposite."""
        source = inspect.getsource(transitions.TransitionsFinder.seedlessFirstOrderCandidate
                                   if hasattr(transitions, "TransitionsFinder")
                                   else transitions)
        self.assertNotIn("exists to at least as high a", source,
                         "the message contradicts the condition above it")


if __name__ == "__main__":
    unittest.main()
