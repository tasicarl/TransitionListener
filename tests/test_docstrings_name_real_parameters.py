"""No docstring may list a parameter its function does not take.

Ten functions did. Most were arguments that had been renamed while the documentation was not:
`calcTf` documented `phase_sym` and `Sint` and took neither, `energyDensity` documented `pot`
and is a method of it, `findAllTransitions` documented three arguments and takes only `self`.
Two were worse than stale. `checkNucleationPossibility` documented one argument it does not
take and none of the eleven it does, and promised a `tuple[bool, dict]` while every `return` in
it is a bare bool. `deformPath` documented `maxstep`, which belongs to `step`, instead of
`minstep`, which it actually takes.

This test is the guard. It walks every function in `src/` and `models/` and compares the
Parameters block against the signature. KNOWN is the set still outstanding, each of which is
fixed on the release branch rather than here so the two do not collide. The test fails both
when a new one appears and when a known one is fixed, so the list cannot rot in either
direction.
"""

from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

SECTION_END = ("Returns", "Raises", "Yields", "Notes", "Examples", "See Also")

# (file, function, documented name) still outstanding; fixed on chore/release-v2.2 (2767339)
KNOWN = {
    ("bubbledynamics.py", "g_eff_DS", "T"),
    ("bubbledynamics.py", "h_eff_DS", "T"),
    ("bubbledynamics_fixedstep.py", "g_eff_DS", "T"),
    ("bubbledynamics_fixedstep.py", "h_eff_DS", "T"),
    ("bubbledynamics_fixedstep.py", "calcMeanBubbleSeparationwExp", "SRFint"),
    ("generic_potential.py", "makePrettyDictionaryPrint", "input_dict"),
}


def _ghosts():
    root = Path(__file__).resolve().parents[1]
    out = set()
    for f in sorted((root / "src").rglob("*.py")) + sorted((root / "models").rglob("*.py")):
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            doc = ast.get_docstring(node)
            if not doc or "Parameters" not in doc:
                continue
            a = node.args
            sig = {x.arg for x in a.posonlyargs + a.args + a.kwonlyargs}
            if a.vararg:
                sig.add(a.vararg.arg)
            if a.kwarg:
                sig.add(a.kwarg.arg)
            section = doc.split("Parameters", 1)[1]
            for stop in SECTION_END:
                section = section.split(stop, 1)[0]
            # "X : type" and "startstep, minstep : type" are both parameter headers; a
            # continuation line starting "word:" inside a description is not, which is why
            # the match is anchored at the start of the line with no leading whitespace left
            # after dedent.
            for m in re.finditer(r"^([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*:", section, re.M):
                for name in (x.strip() for x in m.group(1).split(",")):
                    if name not in sig and name not in ("self", "cls"):
                        out.add((f.name, node.name, name))
    return out


class DocstringParametersTests(unittest.TestCase):
    def test_no_docstring_names_a_parameter_that_does_not_exist(self):
        found = _ghosts()
        new = found - KNOWN
        self.assertEqual(
            new, set(),
            "these docstrings name parameters their function does not take:\n  "
            + "\n  ".join(f"{f}:{fn}() documents {n}" for f, fn, n in sorted(new)))

    def test_the_known_list_has_not_gone_stale(self):
        """When the release branch lands, these are fixed and must leave the list."""
        found = _ghosts()
        fixed = KNOWN - found
        self.assertEqual(
            fixed, set(),
            "these are fixed now and should be removed from KNOWN:\n  "
            + "\n  ".join(f"{f}:{fn}() documents {n}" for f, fn, n in sorted(fixed)))


if __name__ == "__main__":
    unittest.main()
