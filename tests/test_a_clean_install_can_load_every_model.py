"""Every shipped model must load with nothing but the declared dependencies installed.

`models/TL_2HDM.py` could not. It imports `counterterms/twohdm_solver.py`, which imports SymPy
at module level, and SymPy was not in `pyproject.toml`. A clean `pip install -e .` followed by
the first thing the README suggests therefore worked, because the example uses the conformal
model, while the 2HDM raised `ModuleNotFoundError` at load. The import alone was enough to
fail, and the cache does not make SymPy optional afterwards: building the potential from the
shipped tensors calls `counterterms.tensors._require_sympy` 894 times, because the cached
expressions are stored as strings and are parsed back with SymPy.

The check here is the one that fails if a new model, or a new module a model reaches, imports
something the metadata does not promise: the imports are taken from the source and compared with
the declared dependencies, rather than relying on the environment the tests happen to run in,
which has everything installed and so can never show the defect.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import ast
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: Distributions whose import name differs from the name pip installs.
IMPORT_NAME = {"pillow": "pil", "pyyaml": "yaml"}

#: What an optional dependency has to look like to be one: every import of it is inside a
#: `try`, so a missing install degrades instead of raising. `observability` imports the PTA
#: stack that way and reports the PTA columns as unavailable.
#:
#: The exemption is read off the source rather than written out here. A list has to be
#: maintained, and a stale entry, `ceffyl` was one, exempts a module nothing imports and so
#: would wave through a future unguarded import of it. The names are named below only as a
#: floor: the PTA stack has to stay guarded, whatever else is.
GUARDED_FLOOR = {"ptarcade", "enterprise"}


def _declared():
    """The distributions pyproject.toml promises, as import names."""
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies = \[(.*?)^\]", text, re.S | re.M)
    assert block, "pyproject.toml has no dependencies list"
    names = re.findall(r'^\s*"([A-Za-z0-9_.\-]+)', block.group(1), re.M)
    out = set()
    for n in names:
        low = n.lower()
        out.add(IMPORT_NAME.get(low, low).replace("-", "_"))
    return out


def _third_party_imports(paths):
    """Top-level third-party modules imported anywhere under `paths`."""
    found = {}
    for root in paths:
        for p in root.rglob("*.py"):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module]
                else:
                    continue
                for m in mods:
                    top = m.split(".")[0]
                    if top in sys.stdlib_module_names or top == "transitionlistener":
                        continue
                    found.setdefault(top.lower(), set()).add(
                        str(p.relative_to(ROOT)))
    return found


def _imports_by_guard(paths):
    """Third-party modules under `paths`, split by whether every import of one is guarded.

    Returns `(guarded, unguarded)`, two sets of top-level module names. A module lands in
    `guarded` only if no import of it anywhere under `paths` sits outside a `try`, so one
    guarded import somewhere does not excuse an unguarded one elsewhere.
    """
    guarded, unguarded = set(), set()
    for root in paths:
        for p in root.rglob("*.py"):
            try:
                tree = ast.parse(p.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            inside_try = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    for child in ast.walk(node):
                        if isinstance(child, (ast.Import, ast.ImportFrom)):
                            inside_try.add(id(child))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    mods = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    mods = [node.module]
                else:
                    continue
                for m in mods:
                    top = m.split(".")[0]
                    if top in sys.stdlib_module_names or top == "transitionlistener":
                        continue
                    (guarded if id(node) in inside_try else unguarded).add(top.lower())
    return guarded - unguarded, unguarded


class ACleanInstallHasWhatTheCodeImportsTests(unittest.TestCase):

    def test_the_package_declares_everything_it_imports(self):
        declared = _declared()
        found = _third_party_imports([ROOT / "src"])
        guarded, _ = _imports_by_guard([ROOT / "src"])
        missing = {m: f for m, f in found.items()
                   if m not in declared and m not in guarded}
        self.assertEqual(
            {}, missing,
            "imported by src/ but not declared in pyproject.toml: "
            + "; ".join(f"{m} ({sorted(f)[0]})" for m, f in sorted(missing.items())))

    def test_the_shipped_models_declare_everything_they_reach(self):
        """A model is shipped, so what it reaches has to be installed too."""
        declared = _declared()
        found = _third_party_imports([ROOT / "models"])
        guarded, _ = _imports_by_guard([ROOT / "src", ROOT / "models"])
        missing = {m: f for m, f in found.items()
                   if m not in declared and m not in guarded}
        self.assertEqual(
            {}, missing,
            "imported by models/ but not declared: "
            + "; ".join(f"{m} ({sorted(f)[0]})" for m, f in sorted(missing.items())))

    def test_sympy_is_declared_because_the_2hdm_model_needs_it(self):
        """Named on its own, because this is the one that was missing."""
        self.assertIn("sympy", _declared())

    def test_a_cached_run_still_needs_sympy(self):
        """SymPy is needed to use the 2HDM, not only to import it.

        The shipped tensors are cached as strings and parsed back with SymPy, so a run that
        regenerates nothing still calls into it. Counted rather than asserted in the
        abstract: a stand-in for `_require_sympy` records how often the model reaches it
        while the potential is built from the cache.
        """
        import contextlib
        import importlib.util
        import io as _io

        from transitionlistener.counterterms import tensors

        calls = []
        real = tensors._require_sympy
        spec = importlib.util.spec_from_file_location("TL_2HDM", ROOT / "models" / "TL_2HDM.py")
        model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(model)
        try:
            tensors._require_sympy = lambda: (calls.append(1), real())[1]
            with contextlib.redirect_stdout(_io.StringIO()):
                model.R2HDM(dict(lambda1=0.006, lambda2=0.25, lambda3=8.27, lambda4=-2.55,
                                 lambda5=0.76, m12_sq_GeV2=14186.7, tan_beta=17.7,
                                 yukawa_type=1, v_GeV=246.21965079413735), verbose=False)
        finally:
            tensors._require_sympy = real
        self.assertGreater(len(calls), 0,
                           "the cached path never reached SymPy, so the dependency claim "
                           "in this file is wrong")

    def test_the_pta_stack_is_imported_behind_a_guard(self):
        """The one optional dependency the package has stays optional.

        The two tests above exempt whatever is imported inside a `try`. That exemption is
        only honest while the guard is there, so the PTA stack is named here: if an import of
        it moves out of the `try`, a clean install without tempo2 stops being able to run,
        and the exemption would have hidden it.
        """
        guarded, unguarded = _imports_by_guard([ROOT / "src"])
        for name in sorted(GUARDED_FLOOR):
            with self.subTest(module=name):
                self.assertIn(name, guarded,
                              "%s is not imported behind a guard anywhere in src/" % name)
                self.assertNotIn(name, unguarded)
        src = (ROOT / "src" / "transitionlistener" / "observability.py").read_text(
            encoding="utf-8")
        self.assertRegex(src, r"except Exception as exc:",
                         "observability imports the PTA stack without catching the failure")

    def test_the_2hdm_reaches_sympy_through_the_solver_module(self):
        """The module the dependency comment names is the one the model loads.

        A static check of both files, not a run: it reads the imports of `models/TL_2HDM.py`
        and of the solver it names. That the import really goes that way at run time is
        measured by the test below, which loads the model and counts the SymPy calls.
        """
        model = ast.parse((ROOT / "models" / "TL_2HDM.py").read_text(encoding="utf-8"))
        reached = {node.module for node in ast.walk(model)
                   if isinstance(node, ast.ImportFrom) and node.module}
        self.assertIn("transitionlistener.counterterms.twohdm_solver", reached)

        solver = ast.parse((ROOT / "src" / "transitionlistener" / "counterterms"
                            / "twohdm_solver.py").read_text(encoding="utf-8"))
        at_module_level = [node for node in solver.body if isinstance(node, ast.Import)]
        self.assertIn("sympy", {alias.name for node in at_module_level
                                for alias in node.names})


if __name__ == "__main__":
    unittest.main(verbosity=2)
