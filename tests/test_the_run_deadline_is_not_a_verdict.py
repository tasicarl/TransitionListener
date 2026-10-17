"""The run deadline must not be reported as a physics verdict.

`errors.Timeout` carries CLI error code 16 and subclasses `Exception`. It is raised from the
`SIGALRM` handler in `interface/state.py`, armed by `interface/pipeline.py` with
`signal.alarm(timeout)` before the potential is built, so it can land on any bytecode executed
from there on.

A bare `except Exception` that converts what it catches into a TransitionListener error class
therefore turns an aborted run into a finished one with a verdict. `nucleation.py` reported it as
`NucleationError`, which reaches the user as CLI error code 8. `bubbledynamics.py` reported it as
`PercolationError` with a message attributing the failure to strong supercooling. `gwfopt.py`
reported it as "Missing necessary GW parameters", a statement about the input.

`tunneling1D.initialConditions` and `pathDeformation._fit_spline_path` already guarded their own
broad handlers for this reason. This test makes the pattern a rule rather than two examples.
"""

from __future__ import annotations

import ast
import pathlib
import unittest

import numpy as np
from unittest import mock

from transitionlistener import errors
from transitionlistener import gwfopt
from transitionlistener import nucleation


def _src_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1] / "src"


def _enclosing_function(tree: ast.AST, lineno: int) -> str:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end:
                if best is None or node.lineno > best.lineno:
                    best = node
    return best.name if best else "<module>"


def _converting_handlers():
    """Every broad handler that turns what it catches into an error class.

    Returns (path, function, raised, guarded) per handler. "Broad" means `except Exception` or
    `except BaseException`; "converting" means the body raises something whose name ends in
    Error. A handler that re-raises the same object with `raise err` is not converting, so it
    is not listed: a `Timeout` passing through it keeps its own `errorcode`.
    """
    found = []
    for path in sorted(_src_root().rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Try):
                continue
            for index, handler in enumerate(node.handlers):
                kind = handler.type
                if not (isinstance(kind, ast.Name)
                        and kind.id in ("Exception", "BaseException")):
                    continue
                raised = set()
                for inner in ast.walk(handler):
                    if isinstance(inner, ast.Raise) and inner.exc is not None:
                        call = inner.exc.func if isinstance(inner.exc, ast.Call) else inner.exc
                        name = (call.attr if isinstance(call, ast.Attribute)
                                else call.id if isinstance(call, ast.Name) else "?")
                        if name.endswith("Error"):
                            raised.add(name)
                if not raised:
                    continue
                guarded = any(
                    (isinstance(earlier.type, ast.Attribute)
                     and earlier.type.attr == "Timeout")
                    or (isinstance(earlier.type, ast.Name) and earlier.type.id == "Timeout")
                    for earlier in node.handlers[:index])
                found.append((path.name, _enclosing_function(tree, handler.lineno),
                              tuple(sorted(raised)), guarded))
    return found


# Empty, and the staleness test below keeps it that way. It briefly held the two `transitions.py`
# handlers while pull request #37 was in review, because that branch touched the same lines. #37
# merged without its deadline fix, so those two came into `main` unguarded and are fixed here
# instead, which is why nothing is waived any more.
KNOWN_UNGUARDED: set[tuple[str, str]] = set()


class EveryConvertingHandlerLetsTheDeadlineThroughTests(unittest.TestCase):

    def test_no_unwaived_handler_converts_the_deadline(self):
        offenders = [(f, fn, raised) for f, fn, raised, guarded in _converting_handlers()
                     if not guarded and (f, fn) not in KNOWN_UNGUARDED]
        self.assertEqual(
            offenders, [],
            "these broad handlers convert errors.Timeout into a physics error class, so an "
            "aborted run is reported as a finished one: "
            + "; ".join(f"{f}:{fn} -> {','.join(r)}" for f, fn, r in offenders)
            + ". Add `except errors.Timeout: raise` before each")

    def test_no_waiver_is_stale(self):
        """A waiver that is no longer needed hides the next real one."""
        unguarded = {(f, fn) for f, fn, _, guarded in _converting_handlers() if not guarded}
        stale = sorted(KNOWN_UNGUARDED - unguarded)
        self.assertEqual(
            stale, [],
            f"{stale} are waived but already guarded; remove them from KNOWN_UNGUARDED")

    def test_the_sweep_can_see_a_guarded_handler(self):
        """A check that cannot tell the two apart would pass on an empty tree."""
        handlers = _converting_handlers()
        self.assertTrue(any(guarded for *_, guarded in handlers),
                        "the sweep found no guarded handler, so it is not discriminating")
        self.assertIn(("tunneling1D.py", "initialConditions"),
                      {(f, fn) for f, fn, _, guarded in handlers if guarded},
                      "tunneling1D guarded its broad handler before this change and must "
                      "still be seen as guarded")


class TheDeadlineReachesTheCallerTests(unittest.TestCase):
    """The sweep above proves the shape. These drive the real functions."""

    def test_the_deadline_escapes_the_u_shaped_nucleation_search(self):
        """Otherwise `findTminFromUShapedNucl` reports it as `NucleationError`, code 8."""
        def times_out(*args, **kwargs):
            raise errors.Timeout()

        with mock.patch.object(nucleation, "evalNucleationCriterion", times_out):
            with self.assertRaises(errors.Timeout) as caught:
                nucleation.findTminFromUShapedNucl(
                    Tmin=10.0, Tmax=100.0, args=(), start_phase=None, end_phase=None,
                    outdict={}, Ttol=1e-6, maxiter=100, CF=1.0, verbose=False,
                    nuclCriterion=None)
        self.assertEqual(caught.exception.errorcode, 16)

    def test_the_deadline_escapes_the_spectrum_constructor(self):
        """Otherwise it is reported as "Missing necessary GW parameters", about the input."""
        class _TimesOut(dict):
            def __getitem__(self, key):
                raise errors.Timeout()

        with self.assertRaises(errors.Timeout) as caught:
            gwfopt.FOPTspectrum(_TimesOut())
        self.assertEqual(caught.exception.errorcode, 16)


if __name__ == "__main__":
    unittest.main()
