"""``gwConf.use_mean_bubble_separation``: the speed of the spectrum from the bubble separation, or from the slope of
the action.

With False the spectrum takes (beta/H)_S3 in every factor, the expansion factors of arXiv:2511.15687 included, and
the separation follows from it, R_* H = (8 pi / f_perc)^{1/3} max(v_w, c_s) / (beta/H)_S3. A spectrum with
(beta/H)_S3 = b is then the spectrum of a transition whose bubble separation corresponds to b.
"""
from __future__ import annotations

import numpy as np
import pytest

from transitionlistener import errors, runtime_options
from transitionlistener.config import Configuration
from transitionlistener.gwfopt import FOPTspectrum

F_PERC = 0.28957
F_HZ = np.geomspace(1e-10, 1e-6, 40)


def params(betaH_R, betaH_S3=np.nan, betaH_Gamma=np.nan):
    return dict(alpha=1e10, RH=(8 * np.pi / F_PERC) ** (1 / 3) / betaH_R, Treh_SM_GeV=0.15, g_eff_tot_reh=20.0,
                h_eff_tot_reh=20.0, kappa_phi=0.5, kappa_sw=0.5, kappa_turb=0.05, v_wall=1.0, c_s=1 / np.sqrt(3),
                D=1.0, alpha_hyd=1e10, betaH_S3=betaH_S3, betaH_Gamma=betaH_Gamma)


def spectrum(p, **choices):
    conf = Configuration()
    for key, value in choices.items():
        setattr(conf.gwConf, key, value)
    return FOPTspectrum(p, config=conf)


@pytest.mark.parametrize("choices", [dict(source_model="LewickiVaskonen22", expansion_suppression=True),
                                     dict(source_model="LewickiVaskonen22", expansion_suppression=False),
                                     dict(source_model="LISA-pipeline24"), dict(source_model="LISA-update2019")])
def test_the_action_slope_replaces_the_separation_in_every_factor(choices):
    from_slope = spectrum(params(10.0, betaH_S3=25.0, betaH_Gamma=7.0), use_mean_bubble_separation=False,
                          expansion_speed="betaH_Gamma", **choices)
    from_separation = spectrum(params(25.0), expansion_speed="betaH_RH", **choices)
    a, b = from_slope.h2Omega_0_sum(F_HZ), from_separation.h2Omega_0_sum(F_HZ)
    assert np.allclose(a, b, rtol=1e-12, atol=0.0)
    # and the switch is what moves it: the default takes the separation of beta/H = 10
    assert not np.allclose(spectrum(params(10.0, betaH_S3=25.0)).h2Omega_0_sum(F_HZ), a, rtol=1e-3, atol=0.0)


@pytest.mark.parametrize("bad", [np.nan, -3.0, 0.0])
def test_a_slope_that_is_not_positive_is_refused(bad):
    with pytest.raises(errors.GWParamsError, match="betaH_S3"):
        spectrum(params(10.0, betaH_S3=bad), use_mean_bubble_separation=False)


def test_the_input_key_sets_it_and_the_spectrum_records_it():
    conf = Configuration()
    runtime_options.apply_gw_overrides(conf.gwConf, {"gw_use_mean_bubble_separation": "False"})
    assert conf.gwConf.use_mean_bubble_separation is False
    s = FOPTspectrum(params(10.0, betaH_S3=25.0), config=conf)
    assert s.modelling_choices()["use_mean_bubble_separation"] is False
    assert s.betaH == pytest.approx(25.0, rel=1e-12)
