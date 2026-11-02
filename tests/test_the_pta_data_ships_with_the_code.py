"""The PTA likelihoods must come from data this package ships, not from a download.

Until v2.2 `observability` built its likelihoods with `ptarcade.signal_builder.ceffyl_builder`,
which resolves a data set from the configuration and then fetches it from Zenodo at run time.
Two things follow that these tests hold shut. A released code could not produce a PTA column
without a network, and the record it fetched was the 2023 common-uncorrelated one whatever the
configuration said, so every column rested on that product and nothing in the output recorded
which product it was.

The log-likelihood values asserted here were measured once, with the shipped builder, and
match `ceffyl_builder` on the same products to every digit printed. They are regression values:
a change in them means the assembly changed, not that the data did.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
import os
import unittest

import numpy as np

from transitionlistener import observability as pfs
from transitionlistener.observability import PTA_LIKELIHOOD_SETTINGS

#: The seven files a ceffyl data set is made of.
FILES = ("bandwidths.npy", "density.npy", "freqs.npy", "log10rhogrid.npy",
         "log10rholabels.txt", "log.txt", "pulsar_list.txt")

#: (product, bins) -> ln L of one fixed spectrum, measured with the shipped builder.
RECORDED = {
    ("ng15_hd_v2", 14): -193.7891169493,
    ("ng15_cp_v1", 14): -165.1357821988,
    ("ipta2_cp_v1", 13): -164.7416849102,
}

#: Which label each product carries a reference for.
SERVES = {
    "ng15_hd_v2": ("NG15_14bins", "NG15_5bins"),
    "ng15_cp_v2": ("NG15_14bins", "NG15_5bins"),
    "ng15_hdall_v2": ("NG15_14bins", "NG15_5bins"),
    "ng15_hd_v1": ("NG15_14bins", "NG15_5bins"),
    "ng15_cp_v1": ("NG15_14bins", "NG15_5bins"),
    "ng15_hd_dmgp_v1": ("NG15_14bins", "NG15_5bins"),
    "ng12p5_cp_v1": ("NG12_5bins",),
    "ipta2_cp_v1": ("IPTA2_13bins",),
}


def _fixed_spectrum():
    """One spectrum, fixed, so the recorded values mean the assembly and not the physics."""
    from transitionlistener.gwfopt import FOPTspectrum
    return FOPTspectrum(dict(
        alpha=0.5, RH=0.05, Treh_SM_GeV=0.5, g_eff_tot_reh=10.75, h_eff_tot_reh=10.75,
        kappa_phi=0.0, kappa_sw=0.3, kappa_turb=0.0, v_wall=1.0, c_s=1 / np.sqrt(3)))


class TheProductsShipTests(unittest.TestCase):

    def test_every_registered_product_is_installed_whole(self):
        """A registry entry whose files are missing is a broken package, not a missing feature."""
        for name in pfs.PTA_FREE_SPECTRA:
            with self.subTest(product=name):
                path = pfs.free_spectrum_path(name)
                for f in FILES:
                    self.assertTrue(os.path.isfile(os.path.join(path, f)),
                                    f"{name} is missing {f}")

    def test_the_default_is_a_registered_product(self):
        self.assertIn(pfs.DEFAULT_PTA_FREE_SPECTRUM, pfs.PTA_FREE_SPECTRA)

    def test_an_unknown_product_is_refused_by_name(self):
        with self.assertRaises(ValueError) as ctx:
            pfs.free_spectrum_path("not_a_product")
        self.assertIn("not_a_product", str(ctx.exception))

    def test_a_registered_product_with_no_directory_is_refused(self):
        """The failure an incomplete package-data entry produces, named rather than silent."""
        pfs.PTA_FREE_SPECTRA["phantom_for_this_test"] = {"array": "none"}
        try:
            with self.assertRaises(ValueError) as ctx:
                pfs.free_spectrum_path("phantom_for_this_test")
            self.assertIn("named but not installed", str(ctx.exception))
        finally:
            del pfs.PTA_FREE_SPECTRA["phantom_for_this_test"]

    def test_every_product_has_a_reference_for_each_label_it_serves(self):
        """A product whose reference is missing reports nan, so the gap must be deliberate."""
        for name, labels in SERVES.items():
            for label in labels:
                with self.subTest(product=name, label=label):
                    self.assertIsNotNone(
                        pfs.bestfit_lnL(name, label),
                        f"{name} serves {label} but carries no reference for it")

    def test_a_label_a_product_does_not_serve_has_no_reference(self):
        """Better nan than a distance from a best fit obtained on different data."""
        self.assertIsNone(pfs.bestfit_lnL("ng15_hd_v2", "IPTA2_13bins"))
        self.assertIsNone(pfs.bestfit_lnL("ipta2_cp_v1", "NG15_14bins"))

    def test_each_label_is_scored_on_its_own_array(self):
        """A label is a data set and a bin count, so the data has to be that data set.

        Every label was built from one product, so `NG12_5bins` and `IPTA2_13bins` were
        scored against NANOGrav 15 yr free spectra and their log-likelihoods belonged to no
        measurement. The run's choice now applies to the labels whose array it is.
        """
        self.assertEqual("ng12p5_cp_v1", pfs.free_spectrum_for_label("NG12_5bins"))
        self.assertEqual("ipta2_cp_v1", pfs.free_spectrum_for_label("IPTA2_13bins"))
        for label in ("NG15_14bins", "NG15_5bins"):
            with self.subTest(label=label):
                self.assertEqual(pfs.DEFAULT_PTA_FREE_SPECTRUM,
                                 pfs.free_spectrum_for_label(label))
                # A NANOGrav 15 yr product the run selected is used for those labels.
                self.assertEqual("ng15_cp_v1",
                                 pfs.free_spectrum_for_label(label, "ng15_cp_v1"))
        # and selecting one does not drag the other two arrays onto it
        self.assertEqual("ipta2_cp_v1",
                         pfs.free_spectrum_for_label("IPTA2_13bins", "ng15_cp_v1"))

    def test_each_label_finds_the_reference_of_the_product_it_is_scored_on(self):
        """The pairing the references are filed under is the pairing that is used."""
        for label in PTA_LIKELIHOOD_SETTINGS:
            with self.subTest(label=label):
                product = pfs.free_spectrum_for_label(label)
                self.assertIsNotNone(pfs.bestfit_lnL(product, label),
                                     "%s has no reference on %s" % (label, product))

    def test_an_unknown_product_has_no_reference(self):
        self.assertIsNone(pfs.bestfit_lnL("not_a_product", "NG15_14bins"))


class TheBuilderReproducesItsRecordedLikelihoodTests(unittest.TestCase):

    def setUp(self):
        """Skip where the PTA stack is missing, which the import alone does not show.

        `observability` catches a failed import of ceffyl and ptarcade at module level and
        leaves the names bound to None, so importing the builder from it succeeds in an
        environment that cannot run it, and the test then failed on an attribute of None
        instead of being skipped.
        """
        from transitionlistener import observability

        if observability.Ceffyl is None or observability.models_utils is None:
            self.skipTest("PTA stack unavailable: %s" % observability._PTA_IMPORT_ERROR)

    def test_the_shipped_builder_gives_the_recorded_likelihood(self):
        from transitionlistener.observability import _build_ceffyl, PTAModel
        spec = _fixed_spectrum()
        for (product, nbins), expected in RECORDED.items():
            with self.subTest(product=product, nbins=nbins):
                pta = _build_ceffyl(pfs.free_spectrum_path(product), nbins,
                                    PTAModel(spec.gwspec_dict["h2OmegaGW_fun"]))
                got = float(pta.ln_likelihood(np.array([])))
                self.assertAlmostEqual(got, expected, places=6)

    def test_two_products_do_not_give_the_same_likelihood(self):
        """Guards the whole point: if the product did not reach the likelihood, these agree."""
        from transitionlistener.observability import _build_ceffyl, PTAModel
        spec = _fixed_spectrum()
        values = []
        for product in ("ng15_hd_v2", "ng15_cp_v1"):
            pta = _build_ceffyl(pfs.free_spectrum_path(product), 14,
                                PTAModel(spec.gwspec_dict["h2OmegaGW_fun"]))
            values.append(float(pta.ln_likelihood(np.array([]))))
        self.assertGreater(abs(values[0] - values[1]), 1.0,
                           "the two products gave the same likelihood, so the selection "
                           "never reached the data")


class AConfigFileChoosesTheProductTests(unittest.TestCase):
    """The product has to be reachable from a configuration file, not only from Python.

    The point of shipping the data is that a PTA column can be reproduced later. A run can
    only be reproduced from the file that configured it, so the product the likelihood used
    has to be settable there and has to reach `Observability`.
    """

    def test_the_default_product_is_the_registry_default(self):
        from transitionlistener.config import GWConf

        self.assertEqual(pfs.DEFAULT_PTA_FREE_SPECTRUM, GWConf().pta_free_spectrum)

    def test_a_configured_product_reaches_the_configuration(self):
        from transitionlistener.config import GWConf
        from transitionlistener.runtime_options import apply_gw_overrides

        conf = GWConf()
        apply_gw_overrides(conf, {"gw_pta_free_spectrum": "ng15_cp_v1"})
        self.assertEqual("ng15_cp_v1", conf.pta_free_spectrum)

    def test_an_unknown_product_is_refused_before_the_run(self):
        from transitionlistener.config import GWConf
        from transitionlistener.runtime_options import apply_gw_overrides

        with self.assertRaises(ValueError) as caught:
            apply_gw_overrides(GWConf(), {"gw_pta_free_spectrum": "not_a_product"})
        self.assertIn("not_a_product", str(caught.exception))

    def test_the_pipeline_hands_the_configured_product_to_observability(self):
        """Read off the call rather than off a run, which would cost a whole point."""
        import ast
        import pathlib

        src = (pathlib.Path(pfs.__file__).parent / "interface" / "pipeline.py").read_text(
            encoding="utf-8")
        calls = [node for node in ast.walk(ast.parse(src))
                 if isinstance(node, ast.Call)
                 and getattr(node.func, "id", None) == "Observability"]
        self.assertTrue(calls, "the pipeline no longer builds an Observability")
        for call in calls:
            self.assertIn("pta_free_spectrum", [kw.arg for kw in call.keywords])


class ALabelWithNoReferenceReportsNothingTests(unittest.TestCase):
    """A product that carries no reference for a label is not a point that fits badly."""

    def test_the_distance_and_the_verdict_are_nan(self):
        from transitionlistener.observability import Observability

        delta = Observability._compute_delta_lnL(-100.0, np.nan)
        self.assertTrue(np.isnan(delta))
        self.assertTrue(np.isnan(Observability._delta_to_sigma(delta)))
        self.assertIsNone(Observability._within_three_sigma(delta))

    def test_no_reference_and_no_likelihood_is_still_nothing_rather_than_a_verdict(self):
        """Both missing is the case that read as a point scored and ruled out.

        `_compute_delta_lnL` tested the likelihood before the reference, so a label with
        neither returned infinity, which `_within_three_sigma` turns into False.
        """
        from transitionlistener.observability import Observability

        delta = Observability._compute_delta_lnL(np.nan, np.nan)
        self.assertTrue(np.isnan(delta))
        self.assertTrue(np.isnan(Observability._delta_to_sigma(delta)))
        self.assertIsNone(Observability._within_three_sigma(delta))

    def test_the_figure_draws_the_product_the_run_scored_against(self):
        """Both violin callers hand `add_violins` the product, not the default."""
        import types
        from unittest import mock

        from transitionlistener import plots

        gw = mock.MagicMock()
        gw.pta_free_spectrum = "ipta2_cp_v1"
        gw.det = {}
        gw.det_names = ()
        for name in ("plotSensitivities", "plotSensitivitiesPTA"):
            with self.subTest(function=name):
                # `save_figure` as well: the mock stands in for the figure's path too, and
                # an unpatched save writes a file named after the mock into the repository.
                with mock.patch.object(plots, "add_violins") as violins, \
                     mock.patch.object(plots, "add_NeffHatch"), \
                     mock.patch.object(plots, "save_figure"), \
                     mock.patch.object(plots.plt, "savefig"), \
                     mock.patch.object(plots.plt, "show"):
                    try:
                        getattr(plots, name)(gw, showplot=False, nfreq=5)
                    except Exception:
                        # Tolerated after the violins are drawn: the rest of the figure is
                        # built from a stand-in and is not what this test is about. Before
                        # them it is the reason the test would fail, so it is raised.
                        if not violins.called:
                            raise
                self.assertTrue(violins.called, "%s never drew the violins" % name)
                self.assertEqual("ipta2_cp_v1", violins.call_args.kwargs.get("product"))

    def test_a_likelihood_that_could_not_be_evaluated_still_reports_infinity(self):
        from transitionlistener.observability import Observability

        delta = Observability._compute_delta_lnL(np.nan, -100.0)
        self.assertTrue(np.isinf(delta))
        self.assertTrue(np.isinf(Observability._delta_to_sigma(delta)))
        self.assertIs(False, Observability._within_three_sigma(delta))


if __name__ == "__main__":
    unittest.main(verbosity=2)
