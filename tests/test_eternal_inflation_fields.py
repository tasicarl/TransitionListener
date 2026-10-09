"""Change 27: a refusal to finish must carry the percolation numbers it already had.

`calcTf` raises EternalInflationError when Pint never reaches f_final. On a strongly
supercooled line that routinely happens *after* percolation -- all 14 such points on the
2HDM edge line reached Pint between 0.3215 and 0.7640, above f_perc = 0.28957 -- so Tperc
is determined at the moment the point is refused. The scan output records only the error
code, so the message is the surviving record.
"""
import unittest


class EternalInflationFieldsTests(unittest.TestCase):
    def setUp(self):
        from transitionlistener import errors
        self.errors = errors

    def test_fields_are_carried(self):
        err = self.errors.EternalInflationError(
            "msg", Pint_max=0.764, Tperc=24.06, Tlow=3.95, f_final=0.99)
        self.assertEqual(err.errorcode, 15)
        self.assertAlmostEqual(err.Pint_max, 0.764)
        self.assertAlmostEqual(err.Tperc, 24.06)
        self.assertAlmostEqual(err.Tlow, 3.95)
        self.assertAlmostEqual(err.f_final, 0.99)

    def test_default_construction_still_works(self):
        """The eternal-inflation caller passes no fields and must keep working."""
        err = self.errors.EternalInflationError()
        self.assertEqual(err.errorcode, 15)
        self.assertIsNone(err.Pint_max)
        self.assertIsNone(err.Tperc)

    def test_message_only_construction_still_works(self):
        err = self.errors.EternalInflationError("just a message")
        self.assertEqual(err.errorcode, 15)
        self.assertIn("just a message", str(err))
        self.assertIsNone(err.Tperc)


if __name__ == "__main__":
    unittest.main(verbosity=2)


class CalcTfPopulatesTheFieldsTests(unittest.TestCase):
    """The production raise sites must pass the numbers, not just be able to.

    Testing only `errors.py` is vacuous for this change: reverting both `calcTf` copies to
    `raise EternalInflationError(msg)` leaves such a test green, because the class would still
    accept the keywords nobody passes.
    """

    def _pot(self, f_final=0.99):
        import types
        return types.SimpleNamespace(
            config=types.SimpleNamespace(percolationConf=types.SimpleNamespace(f_final=f_final)))

    def _call(self, module_name):
        import importlib
        from transitionlistener import errors
        mod = importlib.import_module(f"transitionlistener.{module_name}")
        # Pint never reaches f_final, which is the refusal this change annotates.
        with self.assertRaises(errors.EternalInflationError) as cm:
            mod.calcTf(24.06, 3.95, lambda T: 0.7639687269359464, self._pot(), verbose=False)
        return cm.exception

    def test_adaptive_backend_reports_the_numbers(self):
        err = self._call("bubbledynamics")
        self.assertAlmostEqual(err.Pint_max, 0.7639687269359464)
        self.assertAlmostEqual(err.Tperc, 24.06)
        self.assertAlmostEqual(err.Tlow, 3.95)
        self.assertAlmostEqual(err.f_final, 0.99)

    def test_fixed_step_backend_reports_them_too(self):
        err = self._call("bubbledynamics_fixedstep")
        self.assertAlmostEqual(err.Pint_max, 0.7639687269359464)
        self.assertAlmostEqual(err.Tperc, 24.06)

    def test_the_message_names_the_percolation_temperature(self):
        """The scan output keeps only the error code, so the message is the surviving record."""
        err = self._call("bubbledynamics")
        self.assertIn("24.06", str(err))
        self.assertIn("0.7639687269359464", str(err))
