"""transitionlistener.spectrum.gw_spectrum, the entry point of an external scan: the model file, its parameters, the
modelling switches as input keys and the settings of the rate as ``nucleation``."""
import os

import numpy as np
import pytest

from transitionlistener.spectrum import gw_spectrum

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models",
                     "TL_conformal_u1_fermions_3d.py")
POINT = dict(g=0.5, y=0.3, v_GeV=1.0)
F = np.geomspace(1e-9, 1e-7, 5)


def test_the_spectrum_of_a_point_with_the_switches_of_the_paper():
    s = gw_spectrum(MODEL, gw_source_model="LewickiVaskonen22", gw_expansion_suppression=True,
                    gw_expansion_speed="betaH_Gamma", **POINT)
    assert np.all(np.isfinite(s(F))) and np.all(s(F) > 0)
    assert s.observables["betaH_RH"] == pytest.approx(14.57, rel=2e-3)


def test_the_rate_settings_reach_the_model():
    nlo = gw_spectrum(MODEL, **POINT)
    lo = gw_spectrum(MODEL, nucleation={"include_determinants": False}, **POINT)
    assert lo.observables["Tperc_SM_GeV"] != pytest.approx(nlo.observables["Tperc_SM_GeV"], rel=1e-3)
    with pytest.raises(ValueError, match="unknown nucleationConf"):
        gw_spectrum(MODEL, nucleation={"no_such_key": 1}, **POINT)
