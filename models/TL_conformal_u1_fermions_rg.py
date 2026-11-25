"""Conformal U(1) with fermions: the four-dimensional daisy-resummed potential with running couplings.

The model of ``TL_conformal_u1_fermions`` (a dark Higgs, a dark photon and two Weyl fermions of charges -1/2 and
+1/2 with the Yukawa coupling y), with the MS-bar one-loop potential in Landau gauge and the couplings of the input
scale mu0 run at one loop to the temperature-dependent scale mu = mu_over_piT * pi T, as
``TL_conformal_u1_no_fermions_rg`` does for the model without fermions:

    V = lambda phi^4/4 + sum_i n_i m_i^4/(64 pi^2) [ln(m_i^2/mu^2) - c_i] + T^4/(2 pi^2) sum_i n_i J_i(m_i^2/T^2)
        - T/(12 pi) sum_bosons n_i [(m_i^2 + Pi_i)^{3/2} - (m_i^2)^{3/2}],

c_i = 3/2 for the dark Higgs, the Goldstone mode and the fermions, 5/6 for the three polarisations of the dark
photon, with m_phi^2 = 3 lambda phi^2, m_chi^2 = lambda phi^2, m_A'^2 = g^2 phi^2, m_psi^2 = y^2 phi^2/2 for each
Weyl fermion (n = -2 each, J_F), Pi_phi = Pi_chi = (lambda/3 + g^2/4 + y^2/12) T^2 and Pi_A'L = (5/12) g^2 T^2, and
g = g(mu), y = y(mu), lambda = lambda(mu). The input is that of the three-dimensional rate: g and y at mu0 and
lambda(mu0) = 0, with mu0 such that the one-loop minimum lies at v. At y = 0 the fermions still carry charge: they
enter the Debye mass and the running of g, so the potential differs from ``TL_conformal_u1_no_fermions_rg``.

The scale follows the temperature only, so the potential is meant for the barrier, at field values of a few T. At
the broken minimum, phi = v >> T, the logarithms ln(g v/pi T) are large. The phases, the vacuum energy and the
thermodynamics are therefore those of TL_conformal_u1_fermions (``Vtot``), and only the bounce is solved on the
potential above (``V_rg``): the false vacuum at the origin, the action through ``bounceFreeAction``, with the 4D rate
T^4 (S3/2 pi T)^{3/2} exp(-S3/T).
"""
import math

import numpy as np

import TL_conformal_u1_fermions as base4d
import dr_conformal_u1_fermions_matching as matching
from transitionlistener.eft_rate.running_coupling_bounce import RunningCouplingBounce
from transitionlistener.finiteT import Jb_spline, Jf_spline

class RunningCouplingV:
    """``V_rg`` of the models with running couplings: the potential below, per temperature."""

    def V_rg(self, X, T, include_radiation=True, include_decoupled=True):
        """The potential with running couplings, per temperature; the field-independent terms of the parent."""
        X = np.asanyarray(X, dtype=float)
        T = np.asanyarray(T, dtype=float)
        phi, Tb = np.broadcast_arrays(X[..., 0], T)
        V = np.empty(phi.shape)
        for t in np.unique(Tb):
            sel = Tb == t
            V[sel] = self._V_rg(phi[sel], float(t))
        if include_radiation:
            V = V + self.constantTerms(T, include_decoupled=include_decoupled)
        return V


class specific_potential(RunningCouplingV, RunningCouplingBounce, base4d.specific_potential):
    """Conformal U(1) with fermions: running couplings at mu = mu_over_piT * pi T."""

    #: the shipped table of S_3/T, tab_data/rg_action_tables/<name>.npz (eft_rate.running_coupling_bounce)
    rg_table_name = "conformal_u1_fermions_rg"
    model_parameters = dict(base4d.specific_potential.model_parameters)
    model_parameters["mu_over_piT"] = {"default": 1.0, "plotname": r"$\mu/\pi T$", "min": 0, "max": np.inf}

    def init(self, inputparam_dict, verbose=True):
        """The fixed-coupling model of the parent, plus the input scale and the scale factor."""
        super().init({k: v for k, v in inputparam_dict.items() if k != "mu_over_piT"}, verbose)
        self.mp["mu_over_piT"] = {"value": float(inputparam_dict.get("mu_over_piT", 1.0))}
        self.mu_over_piT = self.mp["mu_over_piT"]["value"]
        # internal units, lambda(mu0) = 0 with the one-loop minimum at v
        self.mu0 = self.v / matching.vev_from_zero_quartic(self.g, self.y, 1.0)
        self._couplings = {}

    def rg_table_point(self):
        """The couplings of the table."""
        return {"g": self.g, "y": self.y}

    def rg_reference_action(self, ln_x):
        """The reference of the table: S_3/T of the leading-order 3D rate of the same couplings at T/mu0 = exp(ln_x)."""
        if getattr(self, "_reference_3d", None) is None:
            import TL_conformal_u1_fermions_3d as m3d
            ref = m3d.specific_potential({"g": self.g, "y": self.y, "v_GeV": self.v_GeV}, verbose=False)
            ref.config.nucleationConf.include_determinants = ref.config.nucleationConf.include_dim6 = False
            self._reference_3d = ref
        return self._reference_3d.lo_action_over_T(self._reference_3d.input_scale_GeV() * math.exp(ln_x))

    def couplings(self, T):
        """(g^2, y^2, lambda, mu) at mu = mu_over_piT * pi T, T in internal units."""
        T = float(T)
        if T not in self._couplings:
            mu = self.mu_over_piT * math.pi * T
            g, y, lam, _ = matching.rg_run(self.g, self.y, 0.0, self.mu0, mu)
            self._couplings[T] = (g * g, y * y, lam, mu)
        return self._couplings[T]

    def _V_rg(self, phi, T):
        g2, y2, lam, mu = self.couplings(T)
        p2 = phi * phi
        Pi_s = (lam / 3.0 + g2 / 4.0 + y2 / 12.0) * T * T
        bosons = ((1.0, 3.0 * lam * p2, 1.5, Pi_s),
                  (1.0, lam * p2, 1.5, Pi_s),
                  (2.0, g2 * p2, 5.0 / 6.0, 0.0),
                  (1.0, g2 * p2, 5.0 / 6.0, 5.0 / 12.0 * g2 * T * T))
        V = lam / 4.0 * p2 * p2
        for n, m2, c, Pi in bosons:
            m2c = m2.astype(complex)
            with np.errstate(divide="ignore", invalid="ignore"):
                cw = np.where(m2 != 0.0, m2c * m2c * (np.log(m2c / mu ** 2) - c), 0.0)
            V = V + n * np.real(cw) / (64.0 * math.pi ** 2)
            if T > 0.0:
                V = V + n * T ** 4 / (2.0 * math.pi ** 2) * Jb_spline(m2 / (T * T))
                V = V - n * T / (12.0 * math.pi) * np.real((m2c + Pi) ** 1.5 - m2c ** 1.5)
        if y2 > 0.0:
            mf2 = y2 * p2 / 2.0                    # each Weyl fermion, two of them with two degrees of freedom
            with np.errstate(divide="ignore", invalid="ignore"):
                cw = np.where(mf2 > 0.0, mf2 * mf2 * (np.log(np.where(mf2 > 0.0, mf2, 1.0) / mu ** 2) - 1.5), 0.0)
            V = V - 4.0 * cw / (64.0 * math.pi ** 2)
            if T > 0.0:
                V = V + 4.0 * T ** 4 / (2.0 * math.pi ** 2) * Jf_spline(mf2 / (T * T))
        return V
