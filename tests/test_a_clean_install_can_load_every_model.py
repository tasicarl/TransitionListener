"""Every shipped model must load with nothing but the declared dependencies installed.

`models/TL_2HDM.py` could not. It reaches `counterterms/twohdm.py`, which imports SymPy at
module level, and SymPy was not in `pyproject.toml`. A clean `pip install -e .` followed by the
first thing the README suggests therefore worked, because the example uses the conformal model,
while the 2HDM raised `ModuleNotFoundError` at load. The tensors are cached, so the run would
never have evaluated anything with SymPy; the import alone was enough.

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

#: Imported behind a guard that degrades gracefully, so not required to be declared.
#: `observability` catches the import error and reports the PTA columns as unavailable.
OPTIONAL = {"ptarcade", "enterprise", "ceffyl"}


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


class ACleanInstallHasWhatTheCodeImportsTests(unittest.TestCase):

    def test_the_package_declares_everything_it_imports(self):
        declared = _declared()
        found = _third_party_imports([ROOT / "src"])
        missing = {m: f for m, f in found.items()
                   if m not in declared and m not in OPTIONAL}
        self.assertEqual(
            {}, missing,
            "imported by src/ but not declared in pyproject.toml: "
            + "; ".join(f"{m} ({sorted(f)[0]})" for m, f in sorted(missing.items())))

    def test_the_shipped_models_declare_everything_they_reach(self):
        """A model is shipped, so what it reaches has to be installed too."""
        declared = _declared()
        found = _third_party_imports([ROOT / "models"])
        missing = {m: f for m, f in found.items()
                   if m not in declared and m not in OPTIONAL}
        self.assertEqual(
            {}, missing,
            "imported by models/ but not declared: "
            + "; ".join(f"{m} ({sorted(f)[0]})" for m, f in sorted(missing.items())))

    def test_sympy_is_declared_because_the_2hdm_model_needs_it(self):
        """Named on its own, because this is the one that was missing."""
        self.assertIn("sympy", _declared())

    def test_the_optional_imports_are_actually_guarded(self):
        """An optional dependency is only optional if its import error is caught."""
        src = (ROOT / "src" / "transitionlistener" / "observability.py").read_text(
            encoding="utf-8")
        self.assertRegex(src, r"except Exception as exc:",
                         "observability imports the PTA stack without catching the failure")
        for name in ("ptarcade", "enterprise"):
            with self.subTest(module=name):
                self.assertIn(name, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
