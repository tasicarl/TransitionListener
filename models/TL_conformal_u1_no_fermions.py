"""Conformal U(1) with a dark photon and a dark Higgs, and no fermions.

The special case of ``TL_conformal_u1_fermions`` without its fermions: the
gauge-plus-scalar model of arXiv:2511.02910 and 2609.12098 (dark sector only),
for comparing with them like for like. The two Weyl fermions of the parent
carry gauge charge, so setting their Yukawa coupling to zero is not this model
(the Debye mass differs by 25%).

Inputs are (g, v); the quartic follows from radiative symmetry breaking as in
the parent. A 3D-rate version is ``TL_conformal_u1_no_fermions_3d``.
"""
import numpy as np

import TL_conformal_u1_fermions as parent


class specific_potential(parent.specific_potential):
    """Conformal U(1): dark photon and dark Higgs only."""

    fermion_charges = ()

    model_parameters = {
        "g": {
            "default": 0.7,
            "plotname": r"$g$",
            "min": 0,
            "max": 4 * np.pi},
        "v_GeV": {
            "default": 1,
            "plotname": r"$v$ [GeV]",
            "min": 0,
            "max": np.inf},
    }
