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
            "bold label": "**A trial ends in one of three ways.** The field turns around.",
            "a path that does not exist": "The files are in ``config/``.",
            "one semicolon": "The step is small; the tolerance is smaller.",
            "the verb give": "A model gives an effective potential.",
            "the same verb in the past": "The potential gave a sound speed of one.",
            "an invented name": "The converted fraction reaches one at the end.",
            "a short superlative": "A jump is the commonest signature.",
            "a module named as a literal": "The stages are run by ``pipeline``.",
            "a one-word bold label": "**Turbulence**, from the vortical motion.",
            "a bold label of several words": "**The first pass** locates the range.",
            "a word written twice": "The file is in the the output folder.",
        }
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            tmp = pathlib.Path(d)
            for name, line in cases.items():
                page = tmp / "page.rst"
                page.write_text("Title\n=====\n\n%s\n" % line, encoding="utf-8")
                found = checker.check([page])
                self.assertTrue(found, "the checker missed %s: %r" % (name, line))


class TestTitlesAndFigures(unittest.TestCase):
    def test_a_lower_case_section_title_is_caught(self):
        """Titles start with a capital letter, on the pages as in the panels."""
        sys.path.insert(0, str(DOCS))
        import check_docs_style as checker
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            page = pathlib.Path(d) / "page.rst"
            page.write_text("Title\n=====\n\nA sentence.\n\nthe plasma\n----------\n\n"
                            "Another sentence.\n", encoding="utf-8")
            found = checker.check([page])
            self.assertTrue(any("lower case" in f for f in found), found)
            page.write_text("Title\n=====\n\nA sentence.\n\nThe plasma\n----------\n\n"
                            "Another sentence.\n", encoding="utf-8")
            self.assertEqual([], checker.check([page]))

    def test_a_lower_case_panel_title_is_caught(self):
        """The same rule, read off a figure rather than off a page."""
        sys.path.insert(0, str(DOCS))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import check_figure_style as figures

        fig, ax = plt.subplots()
        try:
            ax.set_title("the tabulated bath")
            self.assertTrue(any("lower case" in m
                                for m in figures.check_figure_wording(fig, "probe")))
            ax.set_title("The tabulated bath")
            self.assertEqual([], figures.check_figure_wording(fig, "probe"))
        finally:
            plt.close(fig)

    def test_the_house_style_pads_no_space_beside_a_curve(self):
        """Every curve of a shipped plot reaches both ends of its panel.

        Matplotlib pads an axis by five per cent of the data range on each side, and a
        percolation history and a bubble profile both ended short of the frame because of it.
        A reader cannot tell that padding from a curve that stopped early, which is the one
        thing those plots are looked at for.
        """
        from transitionlistener.plot_settings import plot_settings

        self.assertEqual(0.0, plot_settings["axes.xmargin"])


class TestThePlotsPageIsReproducible(unittest.TestCase):
    def test_every_setting_the_page_quotes_is_the_setting_that_drew_the_figure(self):
        """The plots page says each figure was drawn by the block beside it. It has to be.

        The figures and the page drifted apart twice: a potential drawn at one temperature
        and a block quoting another, and a temperature range widened for one figure and left
        behind on the page. The configuration that drew them is checked in beside the page,
        and this compares the two.
        """
        import re

        import yaml

        page = (DOCS / "source" / "algorithms" / "the_plots.rst").read_text(encoding="utf-8")
        config = yaml.safe_load((DOCS / "docs_gallery.yaml").read_text(encoding="utf-8"))
        plots = config["additional_plots"]

        quoted = {}
        for block in re.findall(r"\.\. code-block:: yaml\n\n((?:(?: {3,}.*)?\n)+)", page):
            lines = [l for l in block.splitlines() if l.strip()]
            if not lines:
                continue
            indent = min(len(l) - len(l.lstrip()) for l in lines)
            text = "\n".join(l[indent:] for l in lines)
            try:
                parsed = yaml.safe_load(text)
            except yaml.YAMLError:
                continue
            if isinstance(parsed, dict):
                # The first block on the page carries the ``additional_plots`` key itself.
                quoted.update(parsed.get("additional_plots", parsed))

        self.assertTrue(quoted, "the page quotes no settings at all")
        for key, settings in quoted.items():
            self.assertIn(key, plots, "the page quotes a %r block the configuration has not"
                          % key)
            if not isinstance(settings, dict):
                continue
            for name, value in settings.items():
                self.assertEqual(
                    plots[key].get(name), value,
                    "the page says %s.%s is %r, the configuration that drew the figure says "
                    "%r" % (key, name, value, plots[key].get(name)))


class TestDocumentationMaths(unittest.TestCase):
    def test_every_equation_renders(self):
        """Every equation on every page compiles under LaTeX.

        MathJax drops an equation it cannot parse without saying so, and the reader sees a
        gap. LaTeX accepts more than MathJax does, so an equation LaTeX refuses is one the
        page will certainly not show.
        """
        sys.path.insert(0, str(DOCS))
        import check_docs_math as checker

        if not checker.pages():
            self.skipTest("no documentation pages were found")
        problems = checker.violations()
        self.assertEqual(problems, [], "\n" + "\n".join(problems))

    def test_an_equation_that_cannot_render_is_caught(self):
        """A rule that cannot fail is not a rule."""
        import shutil
        import tempfile

        sys.path.insert(0, str(DOCS))
        import check_docs_math as checker

        if shutil.which("pdflatex") is None:
            self.skipTest("pdflatex is not installed")
        with tempfile.TemporaryDirectory() as tmp:
            work = pathlib.Path(tmp)
            self.assertTrue(checker._compile([r"\frac{1}{2}"], work))
            for broken in (r"\frac{1}{2", r"\notacommand{x}", r"S_3 = \left( x"):
                self.assertFalse(checker._compile([broken], work),
                                 "LaTeX accepted %r, so the rule cannot fail" % broken)

    def test_the_equations_of_a_page_are_found(self):
        """The reader of the pages sees the same equations the checker does."""
        sys.path.insert(0, str(DOCS))
        import check_docs_math as checker

        page = """Some prose with :math:`a_1` in it.

.. math::
   x = y + z

.. math::
   \\alpha = 1

   \\beta = 2
"""
        found = [body for _, body in checker.equations(page)]
        self.assertEqual(found, ["a_1", "x = y + z", r"\alpha = 1", r"\beta = 2"])


class TestShippedImages(unittest.TestCase):
    def test_every_full_width_figure_is_sharp(self):
        """A figure shown at the width of the text column has to carry the pixels for it.

        Only the figures a page shows at full width, which is what a ``.. figure::`` directive
        does here. A thumbnail in a table links to its own full-size file and is meant to be
        small.
        """
        import re

        sys.path.insert(0, str(DOCS))
        import check_figure_style as cfs

        source = DOCS / "source"
        problems = []
        for page in sorted(source.rglob("*.rst")):
            text = page.read_text(encoding="utf-8")
            # Both directives, because a plot shown at full width is as coarse under
            # ``image`` as under ``figure``. A thumbnail in a gallery is exempt: it is drawn
            # at a third of the column and links to the full-size file behind it.
            for m in re.finditer(r"\.\.\s+(figure|image)::\s+(\S+)"
                                 r"((?:\n\s+:\w[^\n]*)*)", text):
                ref = m.group(2)
                # A badge is served from a URL and has no file to measure.
                if ref.startswith(("http://", "https://")):
                    continue
                if m.group(1) == "image":
                    width = re.search(r":width:\s*(\d+)\s*%", m.group(3))
                    if width is None or int(width.group(1)) < 50:
                        continue
                target = (page.parent / ref).resolve()
                if not target.exists():
                    problems.append("%s refers to %s, which is not there"
                                    % (page.name, ref))
                    continue
                problems += cfs.check_image_resolution(target, "%s: %s" % (page.name, ref))
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


class TestDocumentationPaths(unittest.TestCase):
    def test_every_documentation_path_a_test_names_exists(self):
        """A test that points at a page has to point at where the page is.

        Moving a page is cheap and breaks these silently: the suite then fails on five tests
        that have nothing to do with what was changed. This says so in one line instead.
        """
        import re

        problems = []
        for test in sorted((ROOT / "tests").glob("*.py")):
            text = test.read_text(encoding="utf-8")
            for m in re.finditer(r'"(docs)"((?:\s*/\s*"[^"]+")+)', text):
                parts = re.findall(r'"([^"]+)"', m.group(0))
                target = ROOT.joinpath(*parts)
                if not target.exists():
                    problems.append("%s names %s, which is not there"
                                    % (test.name, "/".join(parts)))
        self.assertEqual(problems, [], "\n" + "\n".join(problems))


class TestFigureStyle(unittest.TestCase):
    """The figure rules, each given a figure that breaks it.

    Three of these are here because the checker once passed them: a curve running through a
    label, a label whose box was taken to include its own leader, and a legend drawn over a
    curve. A rule that cannot fail is not a rule.
    """

    def _checker(self):
        sys.path.insert(0, str(DOCS))
        import matplotlib

        matplotlib.use("Agg")
        import check_figure_style as cfs

        return cfs

    def test_a_banned_phrasing_in_a_panel_is_caught(self):
        """The prose rules reach the words inside a figure, not only the page around it.

        A phrasing struck out of the pages survived a review round inside a panel, because
        nothing read the panels.
        """
        import matplotlib.pyplot as plt

        sys.path.insert(0, str(DOCS))
        import check_figure_style as cfs

        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "the verdict\nis settled here")
        self.assertTrue(cfs.check_figure_wording(fig, "probe"))
        plt.close(fig)

        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "the field turns back here")
        self.assertEqual(cfs.check_figure_wording(fig, "probe"), [])
        plt.close(fig)

    def test_a_curve_through_a_label_is_caught(self):
        cfs = self._checker()
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        x = np.linspace(0.0, 1.0, 200)
        ax.plot(x, x, lw=2)
        ax.text(0.5, 0.5, "straight through this label", ha="center", va="center")
        self.assertTrue([m for m in cfs.check_figure(fig, "t") if "sits on ink" in m])
        plt.close(fig)

    def test_a_label_is_measured_without_its_own_leader(self):
        """The label is in clear space; only its leader reaches the curve."""
        cfs = self._checker()
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.95, 0.95], lw=2)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ax.annotate("a label in clear space", xy=(0.5, 0.92), xytext=(0.5, 0.45),
                    ha="center", va="center",
                    arrowprops=dict(arrowstyle="->"))
        found = [m for m in cfs.check_figure(fig, "t") if "sits on ink" in m]
        self.assertEqual(found, [], found)
        plt.close(fig)

    def test_a_legend_over_a_curve_is_caught(self):
        cfs = self._checker()
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        x = np.linspace(0.0, 1.0, 200)
        ax.plot(x, 0.5 + 0.0 * x, lw=2, label="a curve")
        ax.legend(loc="center")
        self.assertTrue([m for m in cfs.check_figure(fig, "t") if "legend sits" in m])
        plt.close(fig)

    def test_a_shallow_leader_is_caught(self):
        cfs = self._checker()
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        x = np.linspace(0.0, 1.0, 200)
        ax.plot(x, 0.5 + 0.02 * x, lw=1.5)
        ax.annotate("x", xy=(0.9, 0.52), xytext=(0.1, 0.48),
                    arrowprops=dict(arrowstyle="->"))
        self.assertTrue([m for m in cfs.check_figure(fig, "t") if "degrees" in m])
        plt.close(fig)

    def test_an_elbow_leader_is_followed_along_its_own_path(self):
        """A leader drawn as an elbow is tested as the two pieces it is drawn as.

        Tested as the diagonal between its ends instead, it appears to cross curves it never
        goes near, and the figure is rejected for a crossing that is not on the page.
        """
        cfs = self._checker()
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.8, 0.8], lw=2, label="a curve")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        # Out along a low row, then straight up on to the curve: neither piece runs along it.
        ax.annotate("label", xy=(0.8, 0.8), xytext=(0.1, 0.2),
                    ha="right", va="center",
                    arrowprops=dict(arrowstyle="->",
                                    connectionstyle="angle,angleA=0,angleB=90,rad=0"))
        found = [m for m in cfs.check_figure(fig, "t") if "degrees" in m]
        self.assertEqual(found, [], found)
        plt.close(fig)

    def test_a_label_outside_its_panel_is_caught(self):
        """A label anchored beyond the axis limits floats in the margin between panels."""
        cfs = self._checker()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0])
        ax.set_ylim(0.0, 1.0)
        ax.text(0.5, 1.6, "above the panel", ha="center", va="center")
        self.assertTrue([m for m in cfs.check_figure(fig, "t") if "outside its panel" in m])
        plt.close(fig)

    def test_a_legend_outside_its_panel_is_caught(self):
        cfs = self._checker()
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([0.0, 1.0], [0.0, 1.0], label="a curve")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15))
        self.assertTrue([m for m in cfs.check_figure(fig, "t")
                         if "legend is not inside" in m])
        plt.close(fig)

    def test_a_curve_that_stops_inside_the_panel_is_caught(self):
        cfs = self._checker()
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax = plt.subplots()
        x = np.linspace(0.3, 0.7, 50)
        ax.plot(x, x, label="stops inside")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        self.assertTrue(cfs.check_curves_span(ax, "panel"))
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
