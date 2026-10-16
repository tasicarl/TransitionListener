"""No docstring may list a parameter its function does not take.

Eleven docstrings did. Measured on the base commit, 16 functions carried at least one such
parameter; this corrects 11 docstrings and leaves 6 functions with one, each of them pinned in
KNOWN below. Most were arguments that had been renamed while the documentation was not:
`calcTf` documented `phase_sym` and `Sint` and took neither, `energyDensity` documented `pot`
and is a method of it, `findAllTransitions` documented three arguments and takes only `self`.
Two were worse than stale. `checkNucleationPossibility` documented one argument it does not
take and none of the eleven it does, and both its docstring and its signature promised a
`tuple[bool, dict]` while every `return` in it is a bare bool. `deformPath` documented
`maxstep`, which belongs to `Deformation_Spline.step`, a different class, instead of `minstep`,
which it actually takes; the method it calls has no upper bound at all.

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

# Anchored to a line of its own. Splitting on the bare word truncates the Parameters block at
# the first description that happens to contain it, and the rest of the block then goes
# unchecked: on the base commit that hid `tunneling1D.findRscaleNew` documenting `ndim` behind
# "Returns the potential at V(phi)" in its `V` entry, 21 findings instead of 22.
SECTION_START = re.compile(r"^Parameters\s*$", re.M)
SECTION_END = re.compile(
    r"^(?:Returns|Raises|Yields|Notes|Examples|See Also)\s*:?\s*$", re.M)

# (file, function, documented name) still outstanding. These six are corrected by the release
# preparation rather than here, so that the two changes do not touch the same lines. The test
# below fails once they are, which is the signal to empty this set rather than to grow it: a
# ghost found later belongs in a fix, not in a waiver.
KNOWN = {
    ("bubbledynamics.py", "g_eff_DS", "T"),
    ("bubbledynamics.py", "h_eff_DS", "T"),
    ("bubbledynamics_fixedstep.py", "g_eff_DS", "T"),
    ("bubbledynamics_fixedstep.py", "h_eff_DS", "T"),
    ("bubbledynamics_fixedstep.py", "calcMeanBubbleSeparationwExp", "SRFint"),
    ("generic_potential.py", "makePrettyDictionaryPrint", "input_dict"),
}


def _unparsable():
    """Files the sweep could not read, which it must report rather than skip.

    Swallowing a `SyntaxError` would let a file that no longer parses pass this guard
    silently, and a documentation regression inside it would never be seen.
    """
    root = Path(__file__).resolve().parents[1]
    bad = []
    for f in sorted((root / "src").rglob("*.py")) + sorted((root / "models").rglob("*.py")):
        try:
            ast.parse(f.read_text(encoding="utf-8"))
        except SyntaxError as err:
            bad.append(f"{f.relative_to(root)}: {err}")
        except OSError as err:
            bad.append(f"{f.relative_to(root)}: {err}")
    return bad


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
            # "Parameters" is anchored to its own line too, for the same reason the stop
            # words are: a docstring that mentions the word in prose before the heading would
            # otherwise have everything before the real heading treated as the block.
            head = SECTION_START.split(doc, maxsplit=1)
            if len(head) < 2:
                continue
            section = SECTION_END.split(head[1], maxsplit=1)[0]
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
    def test_every_file_in_the_sweep_parses(self):
        """The sweep is only a guard over the files it could read."""
        bad = _unparsable()
        self.assertEqual(bad, [],
                         "these files could not be parsed, so the sweep skipped them "
                         "entirely:\n  " + "\n  ".join(bad))

    def test_no_docstring_names_a_parameter_that_does_not_exist(self):
        found = _ghosts()
        new = found - KNOWN
        self.assertEqual(
            new, set(),
            "these docstrings name parameters their function does not take:\n  "
            + "\n  ".join(f"{f}:{fn}() documents {n}" for f, fn, n in sorted(new)))

    def test_prose_mentioning_parameters_is_not_read_as_a_block(self):
        """Why the heading is anchored to its own line.

        Splitting on the bare word takes everything after the first mention, so a docstring
        with no Parameters block at all but the word in prose has its prose scanned for
        `name : description` lines. A sentence like "Note : the caller owns the object" then
        reads as a parameter called `Note`, and the sweep reports a function that documents
        nothing wrongly. Measured: with the split unanchored this raises, with it anchored it
        does not.
        """
        doc = ("Explains how Parameters are handled elsewhere in the pipeline.\n\n"
               "    Note : the caller owns the lifetime of the object passed in.\n")
        head = SECTION_START.split(doc, maxsplit=1)
        self.assertEqual(len(head), 1,
                         "a prose mention is not a Parameters heading, so there is no block "
                         "to scan and no name to report")

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
