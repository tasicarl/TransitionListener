"""The GW spectrum figure must be complete in every output format.

GitHub issue: on Windows the GW spectrum PDF could not be opened, and the PNG
had no hatching in the predicted-sensitivity regions and an empty legend entry
for them. None of these tests needs Windows.
"""

from __future__ import annotations

import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transitionlistener import plots

GW_PARAMS = dict(alpha=0.5, RH=1e-2, Treh_SM_GeV=0.05, g_eff_tot_reh=10.0, h_eff_tot_reh=10.0,
                 kappa_phi=0.05, kappa_sw=0.5, kappa_turb=0.05, v_wall=1.0, betaH=100.0)


def assert_complete_pdf(testcase, data: bytes):
    """Header, trailer and every xref offset must be consistent (breaks on CRLF translation or truncation)."""
    testcase.assertTrue(data.startswith(b"%PDF-"))
    m = re.search(rb"startxref\s+(\d+)\s+%%EOF\s*$", data)
    testcase.assertIsNotNone(m, "no startxref/%%EOF trailer: truncated PDF")
    xref = int(m.group(1))
    testcase.assertEqual(data[xref:xref + 4], b"xref")
    lines = data[xref:].split(b"trailer")[0].split(b"\n")
    first, count = map(int, lines[1].split())
    for i, entry in enumerate(lines[2:2 + count]):
        offset, _, kind = entry.split()[:3]
        if kind == b"n":
            testcase.assertTrue(data[int(offset):].startswith(b"%d 0 obj" % (first + i)))


class GWSpectrumPlotTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)  # the NG15 violins are sampled

    def test_pdf_is_complete_and_saved_from_a_path(self):
        seen = []
        real_savefig = matplotlib.figure.Figure.savefig

        def spy(fig, fname, *args, **kwargs):
            seen.append(fname)
            return real_savefig(fig, fname, *args, **kwargs)

        with tempfile.TemporaryDirectory(prefix="TL out µ ") as tmp, \
                mock.patch.object(matplotlib.figure.Figure, "savefig", spy):
            plots.plotGWSpectrum(GW_PARAMS, showplot=False, foldername=tmp + "/")
            data = (Path(tmp) / "GW_spectrum.pdf").read_bytes()
            self.assertEqual(sorted(p.name for p in Path(tmp).iterdir()), ["GW_spectrum.pdf"])
        # savefig must get a path, never a (text-mode) file object
        self.assertTrue(all(isinstance(f, (str, Path)) for f in seen), seen)
        assert_complete_pdf(self, data)

    def test_failed_pdf_save_leaves_no_truncated_file(self):
        def boom(*args, **kwargs):
            raise RuntimeError("simulated PDF backend failure")

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch("matplotlib.backends.backend_pdf.RendererPdf.draw_image", boom):
            with self.assertRaises(RuntimeError):
                plots.plotGWSpectrum(GW_PARAMS, showplot=False, foldername=tmp + "/")
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertEqual(plt.get_fignums(), [])

    def test_axes_without_figure_are_saved(self):
        fig, ax = plt.subplots()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                plots.plotGWSpectrum(GW_PARAMS, showplot=False, foldername=tmp + "/", ax=ax)
                assert_complete_pdf(self, (Path(tmp) / "GW_spectrum.pdf").read_bytes())
        finally:
            plt.close("all")

    def test_line_and_grid_plots_leave_no_truncated_file(self):
        from transitionlistener import gridplots, lineplots

        def boom(*args, **kwargs):
            raise RuntimeError("simulated PDF backend failure")

        def line_plot(path):
            fig, ax = plt.subplots()
            ax.plot([0.0, 1.0], [0.0, 1.0])
            lineplots._finish_plot(fig, ax, "x", "lin", np.array([0.0, 1.0]), "title", path)

        def grid_plot(path):
            fig, ax = plt.subplots()
            im = ax.pcolormesh(np.arange(3.0), np.arange(3.0), np.ones((2, 2)))
            gridplots._finish_plot(fig, ax, ("x", "y"), im, "linlin", "title", path)

        for label, make in (("line", line_plot), ("grid", grid_plot)):
            with self.subTest(plot=label), tempfile.TemporaryDirectory() as tmp:
                path = str(Path(tmp) / "plot.pdf")
                make(path)
                assert_complete_pdf(self, Path(path).read_bytes())
                Path(path).unlink()
                with mock.patch("matplotlib.backends.backend_pdf.RendererPdf.draw_path", boom):
                    with self.assertRaises(RuntimeError):
                        make(path)
                self.assertEqual(list(Path(tmp).iterdir()), [])
            plt.close("all")

    def test_every_hatch_is_visible_in_raster_output(self):
        # Agg draws a hatch in the edge colour including its alpha, whereas the
        # PDF backend ignores that alpha: alpha=0 hid the predicted-sensitivity
        # hatching (and its legend swatches) only in PNG output.
        fig, ax = plots.plotGWSpectrum(GW_PARAMS, showplot=False, saveplot=False)
        try:
            def render():
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=100)
                buf.seek(0)
                return plt.imread(buf)

            hatched = [c for c in ax.collections if c.get_hatch()]
            hatched += [h for h in ax.get_legend().legend_handles
                        if getattr(h, "get_hatch", lambda: None)()]
            self.assertGreaterEqual(len(hatched), 2)
            reference = render()
            for artist in hatched:
                hatch = artist.get_hatch()
                artist.set_hatch(None)
                try:
                    with self.subTest(artist=artist.get_label() or repr(artist)):
                        self.assertFalse(np.array_equal(render(), reference),
                                         "hatch is invisible in the PNG output")
                finally:
                    artist.set_hatch(hatch)
        finally:
            plt.close(fig)

if __name__ == "__main__":
    unittest.main()
