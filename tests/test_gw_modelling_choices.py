"""The modelling choices of the gravitational wave spectrum are configuration
switches: the percolation fraction f_perc, the kinetic-energy efficiency of the
sound waves, the form of their lifetime factor, the template and the cosmic
expansion during the transition. These tests pin their meaning and check that
the defaults reproduce the historical numbers.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for path in (REPO_ROOT, SRC_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from transitionlistener import errors  # noqa: E402
from transitionlistener import runtime_options  # noqa: E402
from transitionlistener.config import Configuration, GWConf, PercolationConf  # noqa: E402
from transitionlistener.gwfopt import FOPTspectrum  # noqa: E402


# A strongly supercooled dark transition: alpha of order one, R_* H = 0.12,
# reheating at 100 MeV, no wall-collision source, sound waves with turbulence.
PARAMS = dict(
    alpha=3.0, RH=0.12, Treh_SM_GeV=0.1, g_eff_tot_reh=20.0, h_eff_tot_reh=20.0,
    kappa_phi=0.0, kappa_sw=0.9, kappa_turb=0.09, v_wall=1.0, c_s=1 / np.sqrt(3),
    D=1.0, alpha_hyd=3.0,
)
FREQ = np.logspace(-10, -5, 400)
H_P0 = 1.65e-5 * (0.1 / 100) * (20 / 100) ** 0.5 * (100 / 20) ** (1 / 3)   # redshifted Hubble rate, Hz
RH2 = 2.473e-5 * (3.9309363 / 20) ** (4 / 3) * 20 / 2                       # redshift of the amplitude


def _spectrum(params=PARAMS, **gw_choices):
    conf = Configuration()
    for key, value in gw_choices.items():
        setattr(conf.gwConf, key, value)
    return FOPTspectrum(params, config=conf)


def _peak(spec, source="all", f=FREQ):
    return float(np.max(spec.h2Omega_0_sum(f, source)))


def test_defaults_are_the_historical_choices():
    """No config object means the historical prescription: the LISA pipeline
    template in flat space, the 0.6 kinetic-energy factor, Upsilon = min(1, H tau_sh)
    and the percolation fraction 0.28957 in the bubble-separation scale."""
    spec = FOPTspectrum(PARAMS)
    assert spec.f_perc == pytest.approx(PercolationConf.f_perc)
    assert spec.kinetic_energy_efficiency == pytest.approx(0.6)
    assert spec.source_model == "LISA-pipeline24"
    assert spec.expansion_suppression is False
    assert spec.sw_lifetime_factor == "min"
    spec.h2Omega_0_sum(FREQ)
    expected = (8 * np.pi / 0.28957) ** (1 / 3) * max(PARAMS["v_wall"], PARAMS["c_s"]) / PARAMS["RH"]
    assert spec.betaH == pytest.approx(expected)
    assert spec.sources == ["Wall collisions", "Sound waves", "Turbulence"]
    # the number the code produced before the switches existed, 8.271e-9, times f_perc^{2/3}:
    # the sound waves now receive the separation in the convention of their fits, f_perc^{1/3} R_*, which
    # lowers a source that lives less than a Hubble time by f_perc^{2/3}
    assert _peak(spec) == pytest.approx(8.271e-9 * 0.28957 ** (2 / 3), rel=2e-3)


def test_f_perc_sets_the_bubble_separation_scale():
    conf = Configuration()
    conf.percolationConf.f_perc = 0.5
    spec = FOPTspectrum(PARAMS, config=conf)
    spec.h2Omega_0_sum(FREQ)
    default = FOPTspectrum(PARAMS)
    default.h2Omega_0_sum(FREQ)
    assert spec.betaH / default.betaH == pytest.approx((0.28957 / 0.5) ** (1 / 3))


def test_kinetic_energy_efficiency_scales_the_sound_waves():
    """With Upsilon = min(1, H tau_sh) and H tau_sh < 1 the sound-wave amplitude
    goes as K^2 Upsilon with Upsilon = R_* H / U_f and U_f proportional to
    sqrt(K), so the peak scales as efficiency^(3/2)."""
    base = _peak(_spectrum(), "Sound waves")
    high = _peak(_spectrum(kinetic_energy_efficiency=0.8), "Sound waves")
    assert high / base == pytest.approx((0.8 / 0.6) ** 1.5, rel=1e-6)


def test_lifetime_factor_forms():
    """The three Upsilon forms differ by the known factors at the same H tau_sh,
    and Upsilon is always applied."""
    eff = 0.6
    K = eff * PARAMS["kappa_sw"] * PARAMS["alpha_hyd"] / (1 + PARAMS["alpha_hyd"])
    # the template's separation is its own convention, f_perc^{1/3} RH
    Htau = PARAMS["RH"] * 0.28957 ** (1 / 3) / np.sqrt(0.75 * K)
    assert Htau < 1
    minimum = _peak(_spectrum(sw_lifetime_factor="min"), "Sound waves")
    guo = _peak(_spectrum(sw_lifetime_factor="guo"), "Sound waves")
    capped = _peak(_spectrum(sw_lifetime_factor="guo_capped"), "Sound waves")
    assert guo / minimum == pytest.approx((1 - 1 / np.sqrt(1 + 2 * Htau)) / Htau, rel=1e-6)
    assert capped == pytest.approx(guo)   # the cap only acts for H tau_sh > 1
    # the same switch acts on the LISA-update2019 template
    old_min = _peak(_spectrum(source_model="LISA-update2019", sw_lifetime_factor="min"), "Sound waves")
    old_guo = _peak(_spectrum(source_model="LISA-update2019", sw_lifetime_factor="guo"), "Sound waves")
    assert old_guo < old_min


def test_lifetime_cap_for_long_shells():
    params = dict(PARAMS, RH=2.0)
    p_guo = _peak(_spectrum(params, sw_lifetime_factor="guo"), "Sound waves")
    p_cap = _peak(_spectrum(params, sw_lifetime_factor="guo_capped"), "Sound waves")
    p_min = _peak(_spectrum(params, sw_lifetime_factor="min"), "Sound waves")
    assert p_cap / p_min == pytest.approx(1 - 1 / np.sqrt(3), rel=1e-6)
    assert p_guo > p_cap


def test_expansion_rescales_the_lisa_template_and_adds_the_tail():
    """With the expansion the LISA template is multiplied by A(beta)/A(inf) of
    eq. (23) of 2511.15687, its frequencies scaled by f_p(beta)/f_p(inf) of
    eq. (22), and it falls as f^3 below the horizon frequency of eq. (21)."""
    flat = _spectrum()
    exp = _spectrum(expansion_suppression=True)
    f = np.logspace(-10, -5, 4000)
    y_flat, y_exp = flat.h2Omega_0_sum(f), exp.h2Omega_0_sum(f)
    bH = exp.betaH
    r_A = 1 + 0.8 * (1 - np.exp(np.sqrt(1 / bH)))
    r_f = 1 + 1.8 * bH ** (-1.2)
    # the peak ratios on the sound waves, whose peak lies far above the horizon frequency (the
    # collision peak sits close to it, where the smooth join of the causal tail acts)
    sw_flat, sw_exp = flat.h2Omega_0_sum(f, "Sound waves"), exp.h2Omega_0_sum(f, "Sound waves")
    assert sw_exp.max() / sw_flat.max() == pytest.approx(r_A, rel=2e-3)
    # the peak positions on a grid fine enough that its step (Delta ln f = 3e-3 above) does not
    # decide the comparison
    peaks = []
    for spec, y in ((flat, sw_flat), (exp, sw_exp)):
        fp = f[np.argmax(y)]
        ff = np.geomspace(fp / 1.05, fp * 1.05, 20001)
        peaks.append(ff[np.argmax(spec.h2Omega_0_sum(ff, "Sound waves"))])
    assert peaks[1] / peaks[0] == pytest.approx(r_f, rel=2e-3)
    f_H = H_P0 / (2 * np.pi) * (1 - 0.6 * bH ** (-0.67))
    low = f < f_H / 20
    slope = np.diff(np.log(y_exp[low])) / np.diff(np.log(f[low]))
    assert np.allclose(slope, 3.0, atol=0.05)
    # well above f_H the smooth join is inert
    high = f > 20 * f_H
    assert np.allclose(y_exp[high], r_A * flat.h2Omega_0_sum(f[high] / r_f), rtol=1e-6)
    # a sharp tail and no tail
    sharp = _spectrum(expansion_suppression=True, causality_tail="sharp").h2Omega_0_sum(f)
    off = _spectrum(expansion_suppression=True, causality_tail="off").h2Omega_0_sum(f)
    slope = np.diff(np.log(sharp[low])) / np.diff(np.log(f[low]))
    assert np.allclose(slope, 3.0, atol=1e-6)
    assert np.allclose(off[high], y_exp[high], rtol=1e-6)


def test_lisa_update2019_runs_and_uses_alpha_ds_when_given():
    spec = _spectrum(source_model="LISA-update2019")
    y = spec.h2Omega_0_sum(FREQ)
    assert np.all(np.isfinite(y)) and y.max() > 0
    assert spec.sources == ["Bubble collisions", "Sound waves", "Turbulence"]
    with_ds = _spectrum(dict(PARAMS, alpha_DS=30.0), source_model="LISA-update2019")
    assert _peak(with_ds, "Sound waves") != pytest.approx(_peak(spec, "Sound waves"))


def test_lewicki_vaskonen_flat_space_fit_and_the_split():
    """The flat-space template is the table-1 fit of 2208.11697, one broken
    power law per shell type with its own energy fraction: kappa_phi for the
    scalar-field shells, kappa_sw for the fluid shells."""
    spec = _spectrum(source_model="LewickiVaskonen22")
    assert spec.sources == ["Field shells", "Fluid shells"]
    out = spec.h2Omega_0_sum(FREQ)
    bH = spec.betaH
    K = PARAMS["kappa_sw"] * PARAMS["alpha"] / (1 + PARAMS["alpha"])
    # kappa_phi = 0: only the fluid column (fluid_R3_v1: A = 0.0514, 2 pi f_p/beta = 0.66)
    assert np.all(spec.h2Omega_0_sum(FREQ, "Field shells") == 0)
    i = int(np.argmax(out))
    assert FREQ[i] == pytest.approx(H_P0 / (2 * np.pi) * 0.66 * bH, rel=0.03)
    assert out[i] == pytest.approx(RH2 * 0.0514 * K**2 / bH**2, rel=0.02)
    # all energy in the field: only the field column (bubbles_R3: A = 0.0513, 0.77)
    field = _spectrum(dict(PARAMS, kappa_phi=1.0, kappa_sw=0.0), source_model="LewickiVaskonen22")
    y = field.h2Omega_0_sum(FREQ)
    assert np.all(field.h2Omega_0_sum(FREQ, "Fluid shells") == 0)
    Kf = 1.0 * PARAMS["alpha"] / (1 + PARAMS["alpha"])
    assert FREQ[np.argmax(y)] == pytest.approx(H_P0 / (2 * np.pi) * 0.77 * bH, rel=0.03)
    assert y.max() == pytest.approx(RH2 * 0.0513 * Kf**2 / bH**2, rel=0.02)
    # a split: the two parts add
    half = _spectrum(dict(PARAMS, kappa_phi=0.5, kappa_sw=0.4), source_model="LewickiVaskonen22")
    tot = half.h2Omega_0_sum(FREQ)
    assert np.allclose(tot, half.h2Omega_0_sum(FREQ, "Field shells") + half.h2Omega_0_sum(FREQ, "Fluid shells"))
    # without the causal tail the low-frequency slope is the fit's a
    plain = _spectrum(source_model="LewickiVaskonen22", causality_tail="off").h2Omega_0_sum(FREQ)
    low = FREQ < 1e-10
    slope = np.diff(np.log(plain[low])) / np.diff(np.log(FREQ[low]))
    assert np.allclose(slope, 2.36, atol=0.05)


def test_lewicki_vaskonen_expansion_reduces_to_flat_space_for_fast_transitions():
    """For beta/H -> infinity the 2511.15687 amplitude tends to 0.06 and the
    peak to 0.7 beta/H, close to the flat-space fluid_R3_v1 column (0.0514,
    0.66); at beta/H = 15 the expansion lowers the amplitude and the peak sits
    at 0.7 beta/H (1 + 1.8 (beta/H)^-1.2)."""
    fast = dict(PARAMS, RH=1e-3)
    f = np.logspace(-9, -3, 3000)
    flat = _spectrum(fast, source_model="LewickiVaskonen22").h2Omega_0_sum(f)
    exp = _spectrum(fast, source_model="LewickiVaskonen22", expansion_suppression=True, causality_tail="off").h2Omega_0_sum(f)
    assert exp.max() / flat.max() == pytest.approx(0.06 / 0.0514, rel=0.05)
    assert f[np.argmax(exp)] / f[np.argmax(flat)] == pytest.approx(0.7 / 0.66, rel=0.05)
    slow = dict(PARAMS, RH=0.3)
    exp_s = _spectrum(slow, source_model="LewickiVaskonen22", expansion_suppression=True)
    o_exp = exp_s.h2Omega_0_sum(f)
    bH = exp_s.betaH
    f_p_expected = H_P0 / (2 * np.pi) * 0.7 * bH * (1 + 1.8 * bH ** (-1.2))
    assert f[np.argmax(o_exp)] == pytest.approx(f_p_expected, rel=0.03)
    K = PARAMS["kappa_sw"] * PARAMS["alpha"] / (1 + PARAMS["alpha"])
    assert o_exp.max() == pytest.approx(RH2 * 0.06 * (1 + 0.8 * (1 - np.exp(np.sqrt(1 / bH)))) * K**2 / bH**2, rel=0.03)
    # the causal tail below the horizon frequency
    f_H = H_P0 / (2 * np.pi) * (1 - 0.6 * bH ** (-0.67))
    low = f < f_H / 20
    slope = np.diff(np.log(o_exp[low])) / np.diff(np.log(f[low]))
    assert np.allclose(slope, 3.0, atol=0.05)


def test_flat_space_fits_get_the_causal_tail_unless_switched_off():
    """The flat-space fits of 2208.11697 rise as f^a down to f -> 0. By default they are continued as
    f^3 below the horizon frequency f_H = a_p H_p/(2 pi) of 2511.15687 eq. (20), without the
    expansion factor of its eq. (21); causality_tail = "off" keeps the published fit, which the tail
    leaves unchanged far above f_H."""
    lv = dict(source_model="LewickiVaskonen22", lv22_shape_fluid="fluid_R2_v1")   # a = 1.03
    f = np.logspace(-14, -6, 3000)
    tail = _spectrum(**lv)
    off = _spectrum(**lv, causality_tail="off")
    o_t, o_o = tail.h2Omega_0_sum(f), off.h2Omega_0_sum(f)
    f_H = H_P0 / (2 * np.pi)
    deep = f < f_H / 100
    assert np.allclose(np.diff(np.log(o_t[deep])) / np.diff(np.log(f[deep])), 3.0, atol=0.02)
    assert np.allclose(np.diff(np.log(o_o[deep])) / np.diff(np.log(f[deep])), 1.03, atol=0.02)
    high = f > 100 * f_H
    assert np.allclose(o_t[high] / o_o[high], 1.0, rtol=1e-4, atol=0)
    # at f_H the smooth join suppresses by 2^(-(3-a)/3), a the slope of the fit there
    k = int(np.argmin(np.abs(np.log(f / f_H))))
    a = float(np.log(o_o[k + 1] / o_o[k - 1]) / np.log(f[k + 1] / f[k - 1]))
    assert o_t[k] / o_o[k] == pytest.approx(2 ** (-(3 - a) / 3), rel=0.01)
    # the LISA templates in flat space are left alone
    lisa_t, lisa_o = _spectrum(), _spectrum(causality_tail="off")
    assert np.array_equal(lisa_t.h2Omega_0_sum(f), lisa_o.h2Omega_0_sum(f))


def test_invalid_choices_are_refused():
    conf = Configuration()
    conf.gwConf.source_model = "LISA-pipeline24 "
    with pytest.raises(errors.GWParamsError):
        FOPTspectrum(PARAMS, config=conf)
    conf = Configuration()
    conf.gwConf.kinetic_energy_efficiency = 0.0
    with pytest.raises(errors.GWParamsError):
        FOPTspectrum(PARAMS, config=conf)
    conf = Configuration()
    conf.percolationConf.f_perc = 1.0
    with pytest.raises(errors.GWParamsError):
        FOPTspectrum(PARAMS, config=conf)


def test_input_dictionary_overrides_reach_the_config():
    """The ``gw_<name>`` and ``percolation_f_perc`` keys of a scan's input
    dictionary set the switches, with validation at construction time."""
    conf = Configuration()
    runtime_options.apply_gw_overrides(conf.gwConf, {
        "gw_source_model": "LewickiVaskonen22",
        "gw_kinetic_energy_efficiency": 0.8,
        "gw_expansion_suppression": "true",
        "gw_sw_lifetime_factor": "guo",
        "gw_lv22_shape_field": "bubbles_R2",
        "gw_lv22_shape_fluid": "fluid_R3_vcs",
        "gw_lv22_expanding_shape": "non_dissipative",
        "gw_causality_tail": "sharp",
    })
    assert conf.gwConf.source_model == "LewickiVaskonen22"
    assert conf.gwConf.kinetic_energy_efficiency == pytest.approx(0.8)
    assert conf.gwConf.expansion_suppression is True
    assert conf.gwConf.sw_lifetime_factor == "guo"
    assert conf.gwConf.lv22_shape_field == "bubbles_R2"
    assert conf.gwConf.lv22_shape_fluid == "fluid_R3_vcs"
    assert conf.gwConf.lv22_expanding_shape == "non_dissipative"
    assert conf.gwConf.causality_tail == "sharp"
    runtime_options.apply_percolation_overrides(conf.percolationConf, {"percolation_f_perc": 0.4})
    assert conf.percolationConf.f_perc == pytest.approx(0.4)
    with pytest.raises(ValueError):
        runtime_options.apply_gw_overrides(Configuration().gwConf, {"gw_source_model": "bulk_flow"})
    with pytest.raises(ValueError):
        runtime_options.apply_gw_overrides(Configuration().gwConf, {"gw_kinetic_energy_efficiency": 1.5})
    with pytest.raises(ValueError):
        runtime_options.apply_percolation_overrides(Configuration().percolationConf, {"percolation_f_perc": 0.0})
    fresh = Configuration()
    assert fresh.gwConf.source_model == GWConf.source_model
    assert fresh.percolationConf.f_perc == PercolationConf.f_perc


def test_modelling_choices_report():
    spec = _spectrum(source_model="LewickiVaskonen22", lv22_shape_fluid="fluid_R2_v1")
    report = spec.modelling_choices()
    assert report["source_model"] == "LewickiVaskonen22"
    assert report["lv22_shape_fluid"] == "fluid_R2_v1"
    assert "kinetic_energy_efficiency" not in report and report["causality_tail"] == "smooth"
    assert "causality_tail" not in _spectrum().modelling_choices()
    spec = _spectrum(expansion_suppression=True)
    assert spec.modelling_choices()["causality_tail"] == "smooth"
