"""The sound-wave and turbulence templates are evaluated in their own separation convention.

arXiv:2403.03723 writes its sound-wave and turbulence spectra in terms of H_* R_* with
H_* R_* = (8 pi)^{1/3} max(v_w, c_s) H_*/beta (their eq. 2.4), a convention of the simulations
behind the fits (an estimate from beta at percolation, arXiv:1910.13125), not a measured separation.
TransitionListener's RH is the separation it measures at percolation, n_b(Tperc)^{-1/3}, from which
(beta/H)_RH = (8 pi/f_perc)^{1/3} max(v_w, c_s)/RH; for an exponential rate the two differ by
f_perc^{1/3}. These tests check that the templates receive their own convention and that the
collision part, written in beta/H, is unaffected.
"""
import unittest

import numpy as np

from transitionlistener import constants as c
from transitionlistener.gwfopt import FOPTspectrum

F_PERC = 0.28957
GW = dict(alpha=0.5, RH=0.3, Treh_SM_GeV=0.1, g_eff_tot_reh=20.0, h_eff_tot_reh=20.0,
          kappa_phi=0.0, kappa_sw=0.4, kappa_turb=0.0, v_wall=0.95)


def reference_sound_waves(p, f_Hz):
    """Eqs. (2.4), (2.8)-(2.10) of arXiv:2403.03723 written out with H_* R_* from beta/H."""
    v_w, c_s, D = p["v_wall"], 1.0 / np.sqrt(3.0), 1.0
    beta_H = (8.0 * np.pi / F_PERC) ** (1.0 / 3.0) * max(v_w, c_s) / p["RH"]
    RsH = (8.0 * np.pi) ** (1.0 / 3.0) * max(v_w, c_s) / beta_H
    H0 = 1.65e-5 * (p["Treh_SM_GeV"] / 100.0) * (p["g_eff_tot_reh"] / 100.0) ** 0.5 \
        * (100.0 / p["h_eff_tot_reh"]) ** (1.0 / 3.0)
    Rh2 = c.Omega_gamma_h2 * (c.h_eff_today / p["h_eff_tot_reh"]) ** (4.0 / 3.0) * p["g_eff_tot_reh"] / 2.0
    K = 0.6 * p["kappa_sw"] * p["alpha"] / (1.0 + p["alpha"])
    Delta_w = abs(v_w - c_s) / max(v_w, c_s)
    f1, f2 = 0.2 * H0 / RsH, 0.5 * H0 / RsH / Delta_w
    Htau = min(1.0, RsH / np.sqrt(K / (4.0 / 3.0)))
    Omega_int = Rh2 * 0.11 * K**2 * Htau * RsH
    Omega2 = (np.sqrt(2.0) + 2.0 * (f2 / f1) / (1.0 + f2**2 / f1**2)) * Omega_int / np.pi
    return f1, f2, Omega2, D


class TemplateSeparationTests(unittest.TestCase):
    def setUp(self):
        self.spec = FOPTspectrum(dict(GW), {}, verbose=False)
        self.f = np.logspace(-10, -6, 400)

    def test_the_end_of_transition_separation_is_fperc_to_the_third_of_the_percolation_one(self):
        self.spec.h2Omega_0_sum(1e-8)
        self.assertAlmostEqual(self.spec.RH_end / GW["RH"], F_PERC ** (1.0 / 3.0), places=12)

    def test_the_sound_wave_spectrum_follows_the_template_definition_of_the_separation(self):
        f1, f2, Omega2, _ = reference_sound_waves(GW, self.f)
        expected = self.spec.DBPL(self.f, f1, f2, Omega2, 3.0, 1.0, -3.0, 2, 4)
        got = np.array([self.spec.h2Omega_0_sum(x, "Sound waves") for x in self.f])
        np.testing.assert_allclose(got, expected, rtol=1e-12)

    def test_the_collision_part_depends_on_beta_over_h_only(self):
        p = dict(GW, kappa_phi=0.5, kappa_sw=0.0)
        spec = FOPTspectrum(p, {}, verbose=False)
        got = np.array([spec.h2Omega_0_sum(x, "Wall collisions") for x in self.f])
        beta_H = (8.0 * np.pi / F_PERC) ** (1.0 / 3.0) * max(p["v_wall"], 1.0 / np.sqrt(3.0)) / p["RH"]
        H0 = 1.65e-5 * (p["Treh_SM_GeV"] / 100.0) * (p["g_eff_tot_reh"] / 100.0) ** 0.5 \
            * (100.0 / p["h_eff_tot_reh"]) ** (1.0 / 3.0)
        peak = np.argmax(got)
        self.assertTrue(0 < peak < self.f.size - 1)
        # the collision peak sits at 0.11 beta, whatever separation the other sources use
        f_peak = self.f[peak]
        self.assertAlmostEqual(np.log10(f_peak / (0.11 * beta_H * H0)), 0.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()


class LisaUpdate2019SeparationTests(unittest.TestCase):
    """The LISA-update-2019 template (arXiv:1910.13125, 1903.09642) is written in the same
    separation convention, R_* H = (8 pi)^{1/3} v_w H/beta."""

    def _spec(self, RH):
        from transitionlistener.config import Configuration
        conf = Configuration()
        conf.gwConf.source_model = "LISA-update2019"
        conf.gwConf.expansion_suppression = False
        return FOPTspectrum(dict(GW, RH=RH, kappa_sw=0.4), {}, verbose=False, config=conf)

    def test_it_receives_the_end_of_transition_separation(self):
        spec = self._spec(GW["RH"])
        spec.h2Omega_0_sum(1e-8)
        self.assertAlmostEqual(spec.RH_end / GW["RH"], F_PERC ** (1.0 / 3.0), places=12)

    def test_the_sound_wave_peak_follows_beta_over_h(self):
        # f_sw ~ 8.9 muHz (beta/H) (T/100 GeV) ...: doubling beta/H (halving RH) doubles it
        f = np.logspace(-11, -5, 4000)
        peaks = []
        for RH in (GW["RH"], GW["RH"] / 2):
            spec = self._spec(RH)
            peaks.append(f[np.argmax([spec.h2Omega_0_sum(x, "Sound waves") for x in f])])
        self.assertAlmostEqual(np.log(peaks[0] / peaks[1]), np.log(0.5), delta=0.01)
        beta_H = (8.0 * np.pi / F_PERC) ** (1.0 / 3.0) * GW["v_wall"] / GW["RH"]
        spec = self._spec(GW["RH"])
        H0_Hz = 8.87619e-6 * (GW["Treh_SM_GeV"] / 100) * (GW["g_eff_tot_reh"] / 100) ** 0.5 \
            * (100 / GW["h_eff_tot_reh"]) ** (1 / 3)
        # the (3, 4, 2) shape of 1910.13125 peaks at f0sw
        self.assertAlmostEqual(np.log(peaks[0] / (H0_Hz * beta_H)), 0.0, delta=0.01)
