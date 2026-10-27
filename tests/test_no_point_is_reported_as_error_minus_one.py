"""No failing point is reported under error code -1, and every one carries its cause.

Code -1 named no cause and was documented nowhere. It came from two places: the per-point
handler in `interface/pipeline.py`, whose `getattr(error, "errorcode", -1)` turned any
exception without an `errorcode` into it, and the `else` branch of `call_by_sampler` in
`interface/samplers.py`, which wrote it literally. Both now report `UnexpectedError`, code
999, which is what the package already calls an unclassified failure, and both name the
original exception type in the message so a genuine 999 stays distinguishable from a stray
`ValueError` that reached the handler.

The second half is that the cause reached the user at all. `errormsg` is a registered base
result column, filled on the random-scan path and left empty by the per-point handler, so the
message went to the error log and never to the structured output: a point failed, the user saw
a number, and the reason was in a file beside it.

Measured case behind this: the 2HDM at `lambda4 = -2.0807692307692305` raised a bare
`ValueError` from the step-3 ordering guard, and came out of the scan as error -1 with no
message. That guard is classified now, but any other uncoded exception would have done the
same, so the class is closed here rather than one exception at a time.
"""

from __future__ import annotations

import ast
import pathlib
import unittest
from unittest import mock

import numpy as np

from transitionlistener import errors
from transitionlistener.interface import pipeline


class ALogger:
    """Collects the lines the handler logs, in place of a file on disk."""

    def __init__(self):
        self.lines: list[str] = []

    def log(self, message: str):
        self.lines.append(message)


def handle(error):
    """Run the per-point handler on `error` with an otherwise empty point."""
    logger = ALogger()
    with mock.patch.object(pipeline, "console"):
        result = pipeline._handle_single_point_error(
            error, {"lambda4": -2.0807692307692305}, logger, {})
    return result, logger


class AnUncodedExceptionIsNotReportedAsMinusOneTests(unittest.TestCase):
    """The three kinds of exception that can reach the per-point handler."""

    def test_a_bare_value_error_is_reported_as_the_unclassified_code(self):
        """What the step-3 ordering guard used to raise."""
        result, _ = handle(ValueError("T_bro below T_sym"))
        self.assertEqual(int(result["error"][0]), errors.UnexpectedError.ERRORCODE)

    def test_a_bare_value_error_is_not_reported_as_minus_one(self):
        result, _ = handle(ValueError("T_bro below T_sym"))
        self.assertNotEqual(int(result["error"][0]), -1)

    def test_a_coded_transitionlistener_error_keeps_its_own_code(self):
        """Closing the -1 class must not flatten the codes that do name a cause."""
        result, _ = handle(errors.PercolationError("no percolation temperature"))
        self.assertEqual(int(result["error"][0]), errors.PercolationError().errorcode)

    def test_a_genuine_unexpected_error_keeps_its_code_too(self):
        result, _ = handle(errors.UnexpectedError("something unclassified"))
        self.assertEqual(int(result["error"][0]), errors.UnexpectedError.ERRORCODE)

    def test_an_exception_with_no_message_attribute_still_gets_a_code(self):
        result, _ = handle(ZeroDivisionError())
        self.assertEqual(int(result["error"][0]), errors.UnexpectedError.ERRORCODE)


class TheCauseReachesTheStructuredOutputTests(unittest.TestCase):
    """`errormsg` is a registered column, and the per-point handler left it empty."""

    def test_the_handler_fills_the_errormsg_column(self):
        result, _ = handle(ValueError("T_bro below T_sym"))
        self.assertIn("errormsg", result)
        self.assertTrue(result["errormsg"])

    def test_the_message_names_the_exception_type_that_was_not_classified(self):
        """Otherwise a stray `ValueError` is indistinguishable from a real `UnexpectedError`."""
        result, _ = handle(ValueError("T_bro below T_sym"))
        self.assertIn("ValueError", result["errormsg"])

    def test_the_message_carries_what_the_exception_said(self):
        result, _ = handle(ValueError("T_bro below T_sym"))
        self.assertIn("T_bro below T_sym", result["errormsg"])

    def test_a_coded_error_is_not_prefixed_with_its_type(self):
        """Its code already names the cause, so the prefix would only add noise."""
        result, _ = handle(errors.PercolationError("no percolation temperature"))
        self.assertNotIn("PercolationError:", result["errormsg"])

    def test_the_logged_line_and_the_column_say_the_same_thing(self):
        result, logger = handle(ValueError("T_bro below T_sym"))
        self.assertEqual([line.rstrip("\n") for line in logger.lines],
                         [result["errormsg"]])

    def test_the_input_parameters_are_in_the_message(self):
        result, _ = handle(ValueError("T_bro below T_sym"))
        self.assertIn("lambda4", result["errormsg"])


class TheErrormsgColumnIsRegisteredTests(unittest.TestCase):
    """A result key that is not a registered column never reaches the output."""

    def test_errormsg_is_a_base_result_column(self):
        from transitionlistener.interface import output_schema
        self.assertIn("errormsg", output_schema.BASE_RESULT_COLUMNS)


class NoSourceStillWritesMinusOneTests(unittest.TestCase):
    """Both producers of the code, read from the source rather than exercised."""

    PRODUCERS = ("interface/pipeline.py", "interface/samplers.py")

    def source_of(self, relative):
        root = pathlib.Path(pipeline.__file__).parent.parent
        return (root / relative).read_text()

    def test_no_producer_still_defaults_an_errorcode_to_minus_one(self):
        for relative in self.PRODUCERS:
            with self.subTest(file=relative):
                self.assertNotIn('getattr(error, "errorcode", -1)', self.source_of(relative))
                self.assertNotIn("getattr(err, 'errorcode', -1)", self.source_of(relative))

    def test_no_producer_assigns_minus_one_to_an_error_field(self):
        """Only comments explaining the old behaviour may still contain the number."""
        for relative in self.PRODUCERS:
            with self.subTest(file=relative):
                tree = ast.parse(self.source_of(relative))
                for node in ast.walk(tree):
                    if not isinstance(node, ast.UnaryOp):
                        continue
                    if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
                        self.assertNotEqual(node.operand.value, 1,
                                            f"{relative} still has a literal -1")

    def test_the_sampler_branch_reports_the_unclassified_code_by_name(self):
        source = self.source_of("interface/samplers.py")
        self.assertIn("errors.UnexpectedError.ERRORCODE", source)

    def test_the_sampler_branch_names_the_exception_type(self):
        source = self.source_of("interface/samplers.py")
        self.assertIn("type(err).__name__", source)


class TheUnclassifiedCodeIsAClassConstantTests(unittest.TestCase):
    """The CLI needs the code before an instance of the exception exists."""

    def test_the_constant_and_the_instance_agree(self):
        self.assertEqual(errors.UnexpectedError.ERRORCODE,
                         errors.UnexpectedError().errorcode)

    def test_the_constant_is_the_documented_unclassified_code(self):
        self.assertEqual(errors.UnexpectedError.ERRORCODE, 999)

    def test_the_instance_reads_the_constant_rather_than_repeating_it(self):
        """One source of truth, so the two cannot drift apart later."""
        with mock.patch.object(errors.UnexpectedError, "ERRORCODE", 1234):
            self.assertEqual(errors.UnexpectedError().errorcode, 1234)


if __name__ == "__main__":
    unittest.main()
