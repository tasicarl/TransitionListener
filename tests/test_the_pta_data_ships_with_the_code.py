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

from transitionlistener import pta_free_spectra as pfs

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

    def test_an_unknown_product_has_no_reference(self):
        self.assertIsNone(pfs.bestfit_lnL("not_a_product", "NG15_14bins"))


class TheBuilderReproducesItsRecordedLikelihoodTests(unittest.TestCase):

    def setUp(self):
        try:
            from transitionlistener.observability import _build_ceffyl, PTAModel  # noqa: F401
        except Exception as exc:  # pragma: no cover - optional dependency
            self.skipTest(f"PTA stack unavailable: {exc}")

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
