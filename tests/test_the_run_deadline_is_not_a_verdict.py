"""The run deadline must not be reported as a physics verdict.

`errors.Timeout` carries CLI error code 16 and subclasses `Exception`. It is raised from the
`SIGALRM` handler in `interface/state.py`, armed by `interface/pipeline.py` with
`signal.alarm(timeout)` before the potential is built, so it can land on any bytecode executed
from there on.

A bare `except Exception` that converts what it catches into a TransitionListener error class
therefore turns an aborted run into a finished one with a verdict. `nucleation.py` reported it as
`NucleationError`, which reaches the user as CLI error code 2. `bubbledynamics.py` reported it as
`PercolationError`, code 10, with a message attributing the failure to strong supercooling.
`gwfopt.py` reported it as "Missing necessary GW parameters", a statement about the input.

`tunneling1D.initialConditions` and `pathDeformation._fit_spline_path` already guarded their own
broad handlers for this reason. This test makes the pattern a rule rather than two examples.
"""

from __future__ import annotations

import ast
import contextlib
import io
import pathlib
import types
import unittest

from unittest import mock

from transitionlistener import errors
from transitionlistener import gwfopt
from transitionlistener import nucleation
from transitionlistener.interface import pipeline
from transitionlistener.interface import state as tl_state


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


def _is_bare_timeout_reraise(handler) -> bool:
    """A guard only counts if it catches the deadline **and lets it out unchanged**.

    Checking merely that an `except ...Timeout` precedes the broad handler tests the shape and
    not the contract: `except errors.Timeout: raise errors.NucleationError(...)`, or a plain
    `pass`, would satisfy it while the deadline is still relabelled or dropped. A reviewer
    pointed that out on PR #41, and the runtime witnesses below cover only a few of the sites,
    so the static rule has to carry the rest.

    The body must therefore be exactly one bare `raise`. Comments do not appear in the parse
    tree, so the explanatory comment above each `raise` costs nothing here. A tuple handler such
    as `except (errors.Timeout, ValueError):` is deliberately **not** accepted: re-raising there
    would re-raise the other types too, which is a different decision and should be made
    explicitly. There are none in `src/` today.
    """
    kind = handler.type
    named = ((isinstance(kind, ast.Attribute) and kind.attr == "Timeout")
             or (isinstance(kind, ast.Name) and kind.id == "Timeout"))
    if not named:
        return False
    return (len(handler.body) == 1
            and isinstance(handler.body[0], ast.Raise)
            and handler.body[0].exc is None)


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
                guarded = any(_is_bare_timeout_reraise(earlier)
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


class TheOptionalStagesDoNotAbsorbTheDeadlineTests(unittest.TestCase):
    """Re-raising inside a module is not enough; the caller has to let it out too.

    `_compute_single_point` wraps the spectrum stage so that a point which already has
    transition observables still returns them when only the spectrum fails. That fallback falls
    through to `signal.alarm(0)` and `return context, None`, a success. So the `gwfopt` guard on
    its own moved the deadline one level up into a handler that dropped it completely -- not a
    wrong code, no error at all. A reviewer found this on PR #41 after the guard was added.

    The outer handler of the same function is deliberately **not** guarded: it is the terminus
    that returns the exception as the point's error, which `error_code = getattr(error,
    "errorcode", -1)` then maps. Guarding it would break the mapping. So "every broad handler
    must re-raise" is the wrong rule, and this tests the contract instead: a deadline in an
    optional stage comes back as the point's error.
    """

    def test_a_deadline_in_the_spectrum_stage_is_the_points_error(self):
        pot = types.SimpleNamespace(config=types.SimpleNamespace(
            percolationConf=types.SimpleNamespace(algorithm_mode="adaptive_step_size")))
        observables = types.SimpleNamespace(transitionObservables={0: {"alpha": 0.1}})

        def times_out(*args, **kwargs):
            raise errors.Timeout()

        with mock.patch.object(pipeline, "Phases", lambda *a, **k: "phases"), \
             mock.patch.object(pipeline, "Transitions", lambda *a, **k: "transitions"), \
             mock.patch.object(pipeline, "_select_observables_class",
                               lambda p: lambda *a, **k: observables), \
             mock.patch.object(pipeline, "_find_strongest_transition",
                               lambda d: {"alpha": 0.1}), \
             mock.patch.object(pipeline, "FOPTspectrum", times_out), \
             mock.patch.object(tl_state, "DEBUGMODE", False):
            with contextlib.redirect_stdout(io.StringIO()):
                context, error = pipeline._compute_single_point(
                    inputparams_dict={}, potential=lambda *a, **k: pot,
                    timeout=0, verbose=False, include_smbhb=False, max_stage=4)

        self.assertIsNotNone(
            error,
            "the deadline was absorbed as an optional-stage failure and the point came back "
            "successful, so nothing downstream can report CLI code 16")
        self.assertIsInstance(error, errors.Timeout)
        self.assertEqual(getattr(error, "errorcode", None), 16)

    def test_the_terminal_handler_still_returns_other_errors(self):
        """The guard must not turn the terminus into a re-raiser for ordinary failures."""
        pot = types.SimpleNamespace(config=types.SimpleNamespace(
            percolationConf=types.SimpleNamespace(algorithm_mode="adaptive_step_size")))

        def ordinary(*args, **kwargs):
            raise errors.NucleationError("no solution")

        with mock.patch.object(pipeline, "Phases", ordinary), \
             mock.patch.object(tl_state, "DEBUGMODE", False):
            with contextlib.redirect_stdout(io.StringIO()):
                context, error = pipeline._compute_single_point(
                    inputparams_dict={}, potential=lambda *a, **k: pot,
                    timeout=0, verbose=False, include_smbhb=False, max_stage=4)
        self.assertIsInstance(error, errors.NucleationError)
        self.assertEqual(getattr(error, "errorcode", None), 2)


class TheDocumentedCodesAreTheRealCodesTests(unittest.TestCase):
    """The prose in this file and in the guards names CLI codes. Pin them to `errors.py`.

    The first version of this branch said a deadline rewrapped as `NucleationError` reached the
    user as code 8. It reaches it as code 2; code 8 is `TooMuchSupercoolingError`. A reviewer
    caught it, in a docstring, where nothing executes and no assertion looks. Renumbering any of
    these classes now fails here, which is the signal to re-read the prose that quotes them.
    """

    EXPECTED = {
        "Timeout": 16,
        "NucleationError": 2,
        "TooMuchSupercoolingError": 8,
        "NoViablePhasePairError": 19,
        "PercolationError": 10,
        "PercolationApproximation1Error": 7,
        "WrongT0MinimumError": 3,
    }

    def test_each_class_still_carries_the_code_the_prose_quotes(self):
        for name, code in sorted(self.EXPECTED.items()):
            cls = getattr(errors, name)
            with self.subTest(error=name):
                self.assertEqual(
                    getattr(cls(), "errorcode", None), code,
                    f"{name} no longer carries CLI code {code}. Every docstring and comment in "
                    "this file and in tests/test_diagnostic_messages_say_what_happened.py that "
                    "quotes a code has to be re-read before this number is changed")

    def test_the_gw_error_still_carries_no_code(self):
        """`gwfopt` reports a deadline as `GWParamsError`, which has no code of its own.

        So the prose must not give it one. If it gains an `errorcode`, the claim that the
        deadline is simply lost there becomes a claim about a specific code instead.
        """
        self.assertIsNone(getattr(errors.GWParamsError(), "errorcode", None))


class TheDeadlineReachesTheCallerTests(unittest.TestCase):
    """The sweep above proves the shape. These drive the real functions."""

    def test_the_deadline_escapes_the_u_shaped_nucleation_search(self):
        """Otherwise `findTminFromUShapedNucl` reports it as `NucleationError`, CLI code 2."""
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
