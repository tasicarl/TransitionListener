"""Every CLI error code a run can report must appear in the error reference.

The reference documented four of the thirty exception classes, the four whose docstrings had
carried long prose before pull request #55 moved it out, while the output's ``error`` column can
hold any of twenty codes. A user reading a scan has the integer and nothing else, so a code the
reference omits is a point whose failure cannot be looked up.

What this asserts is the direction that rots: a code added to ``errors.py`` without a row in
``docs/source/error_reference.rst``. The reverse, a row for a code that no longer exists, is
asserted too, because a reference that names a code the package cannot raise sends the reader
looking for a cause that is not there.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import inspect
import re
import unittest
from pathlib import Path

from transitionlistener import errors

REFERENCE = Path(__file__).resolve().parents[1] / "docs" / "source" / "error_reference.rst"


def _codes_in_the_package():
    """Map every code the CLI can write to the exception classes that carry it.

    The code is read the way `interface/pipeline.py` reads it, from the instance attribute
    `errorcode`, with the class constant and then the source as fallbacks, because some classes
    set it in `__init__` and one exposes it as a constant for the handler that has no instance.
    """
    found = {}
    for name, obj in vars(errors).items():
        if not (inspect.isclass(obj) and issubclass(obj, BaseException)):
            continue
        if obj.__module__ != errors.__name__:
            continue
        code = getattr(obj, "errorcode", None)
        if code is None:
            code = getattr(obj, "ERRORCODE", None)
        if code is None:
            match = re.search(r"errorcode\s*=\s*(\d+)", inspect.getsource(obj))
            code = int(match.group(1)) if match else None
        if code is not None:
            found.setdefault(int(code), set()).add(name)
    return found


def _rows_in_the_reference():
    """Return the codes the reference's list-table gives a row to."""
    text = REFERENCE.read_text(encoding="utf-8")
    table = text.split(".. list-table:: CLI error codes", 1)
    assert len(table) == 2, "the reference no longer carries a table titled 'CLI error codes'"
    body = table[1].split("\nCodes ", 1)[0]
    return {int(m) for m in re.findall(r"^   \* - (\d+)$", body, flags=re.MULTILINE)}


class TheErrorReferenceIsCompleteTests(unittest.TestCase):

    def test_the_reference_exists_where_the_test_looks_for_it(self):
        self.assertTrue(REFERENCE.is_file(), f"no error reference at {REFERENCE}")

    def test_every_code_the_package_can_raise_has_a_row(self):
        missing = sorted(set(_codes_in_the_package()) - _rows_in_the_reference())
        self.assertEqual(
            missing, [],
            "error codes with no row in docs/source/error_reference.rst: "
            + ", ".join(f"{c} ({'/'.join(sorted(_codes_in_the_package()[c]))})" for c in missing))

    def test_the_reference_names_no_code_the_package_cannot_raise(self):
        extra = sorted(_rows_in_the_reference() - set(_codes_in_the_package()))
        self.assertEqual(extra, [],
                         f"rows for codes no exception carries: {extra}")

    def test_every_row_names_its_exception_class(self):
        """A row whose code is right and whose class is wrong is worse than a missing row."""
        text = REFERENCE.read_text(encoding="utf-8")
        for code, names in sorted(_codes_in_the_package().items()):
            with self.subTest(code=code):
                self.assertTrue(
                    any(f"``{n}``" in text for n in names),
                    f"code {code} has a row but the reference never names {sorted(names)}")

    def test_the_uncoded_classes_are_listed_under_999(self):
        """Ten classes carry no code and are reported as 999; the reader needs their names."""
        uncoded = sorted(
            name for name, obj in vars(errors).items()
            if inspect.isclass(obj) and issubclass(obj, BaseException)
            and obj.__module__ == errors.__name__
            and getattr(obj, "errorcode", getattr(obj, "ERRORCODE", None)) is None
            and not re.search(r"errorcode\s*=\s*\d+", inspect.getsource(obj)))
        text = REFERENCE.read_text(encoding="utf-8")
        unlisted = [n for n in uncoded if f"``{n}``" not in text]
        self.assertEqual(unlisted, [],
                         f"uncoded exceptions the reference does not name: {unlisted}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
