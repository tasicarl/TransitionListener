"""Conformal U(1) without fermions, with the nucleation rate from the 3D theory.

``TL_conformal_u1_fermions_3d`` on the fermion-free model, with the matching of
``dr_conformal_u1_no_fermions_matching``, and the same defaults: the 3D rate with the NLO
correction and the dimension-six shift of the shipped grid. Two run-configuration
switches, read from ``nucleationConf``:

  input_scheme  "zero_quartic"  lambda(mu0) = 0 at mu0 = g v e^{-1/6}, the input
                                of arXiv:2511.02910 and 2609.12098 (the two agree
                                to O(g^6), tests/test_matching_nofermion.py) (default);
                "CW"            the parent's quartic converted to MS-bar at
                                mu0 = g v.
  mu3_choice    "gT"            mu3 = g T with the input coupling g, not run to the
                                matching scale (default);
                "g3sq"          mu3 = g3^2.
"""
import math

import TL_conformal_u1_fermions as fermions4d
import TL_conformal_u1_fermions_3d as fermions3d
import TL_conformal_u1_no_fermions as base4d
import dr_conformal_u1_fermions_matching as fermions_matching
import dr_conformal_u1_no_fermions_matching as matching
from transitionlistener.eft_rate.one_field_rate_3d import ThreeDimensionalRate


class specific_potential(fermions3d.specific_potential, base4d.specific_potential):
    """Conformal U(1) without fermions whose rate comes from the 3D effective theory."""

    nlo_grid_name = "conformal_u1_no_fermions"
    #: Above these couplings the NLO expansion, not the grid, limits the accuracy of the rate (a warning).
    nlo_expansion_limits = {"g": 0.85}

    def nlo_grid_point(self) -> dict:
        """The coupling the NLO grid of this model is a function of."""
        return {"g": self.g}

    def two_loop_potential(self, p: dict):
        """As the parent, from the fermion-free matching module (the same soft-theory expression)."""
        return lambda phi3: matching.V3_soft_two_loop(phi3, p["g3sq"], p["mD2"], p["lambda_vl"], p["lambda_vll"],
                                                      p["mu3"])

    def two_loop_kinetic(self, p: dict):
        """As the parent, from the fermion-free matching module."""
        return lambda phi3: matching.Z3_soft_NLO(phi3, p["g3sq"], p["mD2"], p["lambda_vl"])

    def input_scale_GeV(self) -> float:
        """The scale mu0 of the couplings: at fixed coupling, everything in the 3D theory is a function of T/mu0."""
        if self.config.nucleationConf.input_scheme == "zero_quartic":
            return self.g * self.v_GeV * math.exp(-1.0 / 6.0)
        return fermions_matching.input_scale(self.g, self.v_GeV)

    def vacuum_energy_GeV4(self) -> float:
        """The energy difference of the two vacua at T = 0, zero-quartic expression at y = 0."""
        return fermions_matching.vacuum_energy_zero_quartic(self.g, 0.0, self.g * self.v_GeV * math.exp(-1.0 / 6.0))

    def soft_parameters(self, T_GeV: float) -> dict:
        """Every matched 3D parameter at ``T_GeV`` from the fermion-free matching."""
        conf = self.config.nucleationConf
        scheme = conf.input_scheme
        if scheme == "CW":
            mu0 = fermions_matching.input_scale(self.g, self.v_GeV)
            lam0 = fermions_matching.msbar_quartic_from_cw(self.g, 0.0, self.l, self.v_GeV, mu0)
        elif scheme == "zero_quartic":
            mu0, lam0 = self.g * self.v_GeV * math.exp(-1.0 / 6.0), 0.0
        else:
            raise ValueError(f"input_scheme must be 'CW' or 'zero_quartic', not {scheme!r}")
        mu3 = None
        if conf.mu3_choice == "gT":
            mu3 = self.g * T_GeV
        return matching.matched_soft_parameters(
            self.g, lam0, mu0, T_GeV, mu4_over_T=conf.mu4_over_T,
            matching_order=getattr(conf, "matching_order", "NLO"), mu3=mu3)


assert specific_potential.__mro__[1:5] == (fermions3d.specific_potential, ThreeDimensionalRate, base4d.specific_potential,
                                           fermions4d.specific_potential)
