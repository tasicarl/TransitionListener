"""The documentation is held to the house rules for prose, maths and figures.

The rules themselves are in ``docs/check_docs_style.py`` and ``docs/check_figure_style.py``,
each with the reason it exists. They are here as a test so that a page or a figure that breaks
one fails the suite, rather than being found by a reader of the rendered page.
"""
import pathlib
import subprocess
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"


class TestDocumentationStyle(unittest.TestCase):
    def test_prose_and_maths_follow_the_house_rules(self):
        """Every page passes ``docs/check_docs_style.py``."""
        sys.path.insert(0, str(DOCS))
        import check_docs_style as checker

        paths = sorted((DOCS / "source").rglob("*.rst"))
        paths += sorted((DOCS / "source").rglob("*.md"))
        paths = [p for p in paths if "api/" not in str(p)]
        self.assertTrue(paths, "no documentation pages were found to check")
        problems = checker.check(paths)
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_the_checker_catches_what_it_is_for(self):
        """A rule that cannot fail is not a rule.

        Each rule is given a line that breaks it, and the checker has to say so. Without this
        a mistake in a pattern leaves the rule silently switched off, and the suite goes on
        passing.
        """
        sys.path.insert(0, str(DOCS))
        import check_docs_style as checker

        cases = {
            "text in maths": r"The step is :math:`\Delta T = \text{step}` wide.",
            "double nabla": r"It solves :math:`\left(\nabla\nabla V\right)^{-1}` for it.",
            "italic differential": r"The slope is :math:`d\phi/dT` there.",
            "two semicolons": "One thing; another thing; a third thing.",
            "em dash": "The step is small — smaller than the tolerance.",
            "filler": "Crucially, the step has to be small.",
        }
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            for name, line in cases.items():
                page = tmp / "page.rst"
                page.write_text("Title\n=====\n\n%s\n" % line, encoding="utf-8")
                found = checker.check([page])
                self.assertTrue(found, "the checker missed %s: %r" % (name, line))


if __name__ == "__main__":
    unittest.main()
