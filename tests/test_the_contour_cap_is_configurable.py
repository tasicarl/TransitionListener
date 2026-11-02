"""The fraction of the potential a two-field panel draws contours over is a setting.

`plot2d` draws contours from the minimum of the potential up to
``min(V) + cfrac * (max(V) - min(V))``, so at the default of 0.8 the top fifth of the range
carries none. On a benchmark with a deep broken minimum that fifth is the region around the
origin, which is where the tunnelling path starts, and the panel then shows the path leaving
a blank area. The cap stays at 0.8, because it is what resolves the minima on most points,
and a configuration file can raise it.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import inspect
import unittest
from unittest import mock

import numpy as np


class TheDefaultIsUnchangedTests(unittest.TestCase):

    def test_both_panels_still_default_to_four_fifths(self):
        from transitionlistener import plots

        for name in ("plotProfileV", "plotPotential"):
            with self.subTest(method=name):
                signature = inspect.signature(getattr(plots.TLPlots, name))
                self.assertEqual(0.8, signature.parameters["cfrac"].default)

    def test_plot2d_still_defaults_to_four_fifths(self):
        from transitionlistener import plots

        self.assertEqual(0.8, inspect.signature(plots.plot2d).parameters["cfrac"].default)


class TheSettingReachesTheContoursTests(unittest.TestCase):
    """Read off the call rather than off a figure, which would cost a whole point."""

    def _dispatch(self, spec):
        from transitionlistener.interface import single_point

        executor = single_point.PlotExecutor.__new__(single_point.PlotExecutor)
        executor._outpath = ""
        executor._verbose = False
        plotter = mock.MagicMock()
        with mock.patch.object(single_point.plots, "TLPlots", return_value=plotter):
            executor._conf = mock.MagicMock()
            executor._input_params = {}
            executor._run_additional_plot_impl(spec["key"], spec["spec"], {"pot": None})
        return plotter

    def test_a_configured_cap_reaches_the_path_panel(self):
        plotter = self._dispatch({"key": "profileV",
                                  "spec": {"field_index_1": 0, "field_index_2": 1,
                                           "cfrac": 1.0}})
        self.assertEqual(1.0, plotter.plotProfileV.call_args.kwargs["cfrac"])

    def test_the_path_panel_keeps_the_default_when_nothing_is_set(self):
        plotter = self._dispatch({"key": "profileV",
                                  "spec": {"field_index_1": 0, "field_index_2": 1}})
        self.assertEqual(0.8, plotter.plotProfileV.call_args.kwargs["cfrac"])

    def test_a_configured_cap_reaches_the_potential_panel(self):
        plotter = self._dispatch({"key": "potential",
                                  "spec": {"T_GeV": 10.0, "phi_ranges_GeV": [0, 1],
                                           "cfrac": 1.0}})
        self.assertEqual(1.0, plotter.plotPotential.call_args.kwargs["cfrac"])

    def test_the_potential_panel_keeps_the_default_when_nothing_is_set(self):
        plotter = self._dispatch({"key": "potential",
                                  "spec": {"T_GeV": 10.0, "phi_ranges_GeV": [0, 1]}})
        self.assertEqual(0.8, plotter.plotPotential.call_args.kwargs["cfrac"])


class TheCapChangesWhatIsDrawnTests(unittest.TestCase):
    """A setting that reaches the call but not the contours would pass the tests above."""

    def test_raising_the_cap_adds_levels_above_the_old_ceiling(self):
        from transitionlistener import plots

        class _Flat:
            Ndim = 2
            conversionFactor = 1.0

            def DVtot(self, X, T):
                return np.asarray(X)[..., 1]

        levels = {}

        def _record(*args, **kwargs):
            levels[kwargs.get("tag")] = np.asarray(args[3], dtype=float)
            return mock.MagicMock()

        for tag, cfrac in (("low", 0.8), ("high", 1.0)):
            with mock.patch.object(plots.plt, "figure"), \
                 mock.patch.object(plots.plt, "subplots"):
                ax = mock.MagicMock()
                ax.contour.side_effect = lambda *a, **k: _record(*a, tag=tag)
                ax.figure = mock.MagicMock()
                plots.plot2d(_Flat(), (0.0, 1.0, 0.0, 10.0), ax, T=1.0, cfrac=cfrac,
                             clevs=20)

        self.assertIn("low", levels)
        self.assertIn("high", levels)
        self.assertGreater(float(levels["high"].max()), float(levels["low"].max()),
                           "raising the cap drew no contour above the old ceiling")
        self.assertAlmostEqual(float(levels["high"].max()), 10.0, places=6)
        self.assertAlmostEqual(float(levels["low"].max()), 8.0, places=6)


if __name__ == "__main__":
    unittest.main()
