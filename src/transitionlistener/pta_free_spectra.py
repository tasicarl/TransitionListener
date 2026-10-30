"""The pulsar-timing free-spectrum products TransitionListener ships, and how to load one.

Until v2.2 the PTA likelihoods did not use any data this package ships. They were built by
``ptarcade.signal_builder.ceffyl_builder``, which resolves a data set from the configuration
and then downloads it from Zenodo at run time. A released code that needs a network fetch to
produce a likelihood column cannot be reproduced offline, and the records it fetched are the
2023 ones, so every likelihood column rested on the common-uncorrelated product of that year
whatever the configuration said.

The products below are shipped instead, under ``tab_data/pta_free_spectra``. Each is a kernel
density representation of a free-spectrum posterior in the format ``ceffyl`` reads, seven files
of about 2.4 MB. Six are the versions compared in the appendix of the binary-foreground paper;
the other two are the data sets PTArcade would have downloaded for NANOGrav 12.5 yr and for
IPTA DR2, and are byte-identical to them, so nothing is fetched any more.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import os
from collections import OrderedDict

#: Directory holding one subdirectory per product.
PTA_FREE_SPECTRUM_DIR = os.path.join(
    os.path.dirname(__file__), "tab_data", "pta_free_spectra")

#: The product used when nothing names one.
DEFAULT_PTA_FREE_SPECTRUM = "ng15_hd_v2"

#: Every shipped product: what it is, when it was made, and which chain it came from.
#:
#: ``array`` is the pulsar timing array, ``correlation`` the overlap reduction function the
#: free spectrum was inferred with, and ``chain`` the posterior sample the kernel density was
#: built from, as the product's own ``log.txt`` records it. "ptswapfix" marks the 2026 re-runs
#: made after the correction to the sampler's parallel-tempering swaps.
PTA_FREE_SPECTRA = OrderedDict([
    ("ng15_hd_v2", {
        "array": "NANOGrav 15 yr", "correlation": "Hellings-Downs",
        "created": "2026-04-06", "chain": "model_hd_30f_ptswapfix",
        "note": "the 2026 re-run; the default",
    }),
    ("ng15_cp_v2", {
        "array": "NANOGrav 15 yr", "correlation": "common uncorrelated red noise",
        "created": "2026-04-06", "chain": "model_crn_30f_ptswapfix",
        "note": "the 2026 re-run",
    }),
    ("ng15_hdall_v2", {
        "array": "NANOGrav 15 yr",
        "correlation": "Hellings-Downs with a monopole, a dipole and a common process",
        "created": "2025-12-15", "chain": "model_cpmpdphd_30f_ptswapfix",
        "note": "the most conservative set of components",
    }),
    ("ng15_hd_v1", {
        "array": "NANOGrav 15 yr", "correlation": "Hellings-Downs",
        "created": "2023-12-10", "chain": "30fCP_30fiRN_3A",
        "note": "before the 2026 correction to the sampler",
    }),
    ("ng15_cp_v1", {
        "array": "NANOGrav 15 yr", "correlation": "common uncorrelated red noise",
        "created": "2023-12-10", "chain": "30fCP_30fiRN_2A_2",
        "note": "what PTArcade downloads, and what every TransitionListener PTA column "
                "rested on before v2.2",
    }),
    ("ng15_hd_dmgp_v1", {
        "array": "NANOGrav 15 yr",
        "correlation": "Hellings-Downs, dispersion measure as a Gaussian process",
        "created": "2023-05-15", "chain": "fshd_dmgp",
        "note": "the set this package already shipped, for the free spectrum drawn behind "
                "the spectrum figure",
    }),
    ("ng12p5_cp_v1", {
        "array": "NANOGrav 12.5 yr", "correlation": "common uncorrelated red noise",
        "created": "2024-01-16", "chain": "30fCP_30fiRN_2A_ext",
        "note": "what PTArcade downloads for NANOGrav 12.5 yr",
    }),
    ("ipta2_cp_v1", {
        "array": "IPTA DR2", "correlation": "common uncorrelated red noise",
        "created": "2024-01-07", "chain": "ipta_dr2_fs_crn",
        "note": "what PTArcade downloads for IPTA DR2",
    }),
])


def free_spectrum_path(name: str | None = None) -> str:
    """Return the directory of one shipped product.

    Parameters
    ----------
    name : str, optional
        A key of :data:`PTA_FREE_SPECTRA`. ``None`` selects
        :data:`DEFAULT_PTA_FREE_SPECTRUM`.

    Raises
    ------
    ValueError
        If the name is not a shipped product, or its directory is missing from the
        installation, which an incomplete package data entry would cause.
    """
    key = DEFAULT_PTA_FREE_SPECTRUM if name is None else str(name)
    if key not in PTA_FREE_SPECTRA:
        raise ValueError(
            f"Unknown PTA free-spectrum product {key!r}. "
            f"Shipped: {', '.join(PTA_FREE_SPECTRA)}."
        )
    path = os.path.join(PTA_FREE_SPECTRUM_DIR, key)
    if not os.path.isdir(path):
        raise ValueError(
            f"The free-spectrum product {key!r} is named but not installed; "
            f"expected it at {path}."
        )
    return path


#: Reference log-likelihood per product and per likelihood label.
#:
#: Each ``sigma`` the code reports is ``bestfit_lnL - lnL``, so the reference has to belong to
#: the product the likelihood was evaluated on. These were obtained with ultranest over the
#: transition strength, the inverse duration and the reheating temperature of the sound-wave
#: template, flat priors throughout, the efficiency factor from
#: :func:`hydrodynamics.kappa_sw` at a wall velocity of one.
#:
#: Three of the four references released with v2.1 are reproduced by that calculation on the
#: product they were made on, ``ng15_cp_v1``, to between 0.029 and 0.045 in log-likelihood.
#: The fourth, ``IPTA2_13bins``, was released as -105.66903 and is not reachable: a nested
#: sampler cannot overshoot a maximum, and adding a binary foreground with both parameters
#: free, which can only raise it, still reaches only -105.854. More live points, and every bin
#: count from 5 to 30, were ruled out by measurement. The value below is the reproducible one.
PTA_BESTFIT_LNL = {
    "ng15_hd_v2": {"NG15_14bins": -105.65005153, "NG15_5bins": -33.89607940},
    "ng15_cp_v2": {"NG15_14bins": -104.88702140, "NG15_5bins": -33.86097277},
    "ng15_hdall_v2": {"NG15_14bins": -108.98115566, "NG15_5bins": -37.42190053},
    "ng15_hd_v1": {"NG15_14bins": -103.47695026, "NG15_5bins": -29.33904777},
    "ng15_cp_v1": {"NG15_14bins": -103.08982451, "NG15_5bins": -29.50838035},
    "ng15_hd_dmgp_v1": {"NG15_14bins": -112.13013487, "NG15_5bins": -35.61306571},
    "ng12p5_cp_v1": {"NG12_5bins": -36.86269137},
    "ipta2_cp_v1": {"IPTA2_13bins": -105.88364710},
}


def bestfit_lnL(product: str, label: str) -> float | None:
    """The reference log-likelihood for one label on one product, or None if there is none.

    A missing entry is reported as ``None`` rather than guessed, so that a label evaluated on a
    product it has no reference for comes out as not-a-number rather than as a distance from
    somebody else's best fit.
    """
    return PTA_BESTFIT_LNL.get(str(product), {}).get(str(label))
