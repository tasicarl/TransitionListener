"""The accuracy verdict must be visible in the output, not only inside the solver.

Change 24 records whether the accepted percolation profile met acc_tperc, acc_tfinal and
acc_rh. The targets are advisory -- the controller accepts either way, because by the time it
estimates, refinement has nothing left to try -- so this verdict is the only record of whether
they were honoured, and a profile whose RH missed its 1% target was otherwise indistinguishable
in the output from one that met it (measured p90 spread: 27.6%).

Until these four names were registered the verdict reached `PercolationDiagnostics` and stopped
there: absent from `config.all_observables`, from `samplers.get_empty_result()` and from the
scan JSON, so only a caller passing `return_metadata=True` could see it.

The tri-state survives without a tri-state column: warning false with finite uncertainties
means the targets were met; warning true means they could not be certified, which covers a
target genuinely missed as well as a profile accepted with no comparable second estimate to
measure convergence against; warning false with nan means no verdict was formed at all, which
is the fixed-step backend and the adaptive controller's iteration-limit exit.
"""
import math
import unittest

NAMES = (
    "WARNING:accuracy_targets_missed",
    "DIAG:accuracy_tperc_uncertainty",
    "DIAG:accuracy_tfinal_uncertainty",
    "DIAG:accuracy_rh_uncertainty",
)


class AccuracyObservablesRegisteredTests(unittest.TestCase):
    def test_all_four_are_declared(self):
        from transitionlistener import config
        for name in NAMES:
            self.assertIn(name, config.all_observables)

    def test_every_declared_observable_has_an_empty_result_entry(self):
        """The invariant the branch's own warning test encodes, across all four at once.

        A failed scan point builds its row from `get_empty_result()`; a name present in one and
        not the other produces a row that does not line up with a successful point's.
        """
        from transitionlistener import config
        from transitionlistener.interface import samplers
        missing = set(config.all_observables) - set(samplers.get_empty_result())
        self.assertEqual(missing, set(), f"declared but with no empty-result entry: {missing}")

    def test_the_warning_has_a_column_position(self):
        from transitionlistener.interface import output_schema
        self.assertIn("WARNING:accuracy_targets_missed", output_schema.WARNING_COLUMN_ORDER)

    def test_both_backends_carry_the_defaults(self):
        """The fixed-step backend never forms the verdict, so it must still report the
        default rather than omit the key."""
        from transitionlistener import transitionObservables, transitionObservables_fixedstep
        for module in (transitionObservables, transitionObservables_fixedstep):
            source = open(module.__file__).read()
            for name in NAMES:
                self.assertIn(f'"{name}"', source,
                              f"{name} missing from {module.__name__}'s template")

    def test_the_defaults_encode_no_verdict(self):
        """An unassessed point must read as "not missed, unknown by how much"."""
        from transitionlistener.interface import samplers
        empty = samplers.get_empty_result()
        self.assertIs(empty["WARNING:accuracy_targets_missed"], False)
        for name in NAMES[1:]:
            self.assertTrue(math.isnan(float(empty[name])),
                            f"{name} should default to nan, got {empty[name]}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
