"""Conformal U(1) with fermions with the nucleation rate from the 3D effective theory.

The four-dimensional model of ``TL_conformal_u1_fermions`` with ``one_field_rate_3d.ThreeDimensionalRate``, which replaces the
rate where the run configuration sets ``rate_mode: "model"``. This file adds its
matching (``dr_conformal_u1_fermions_matching``), the input scale of its couplings and its shipped NLO grid. By default
the rate is the 3D one (``rate_mode = "model"``) with the NLO correction and the dimension-six shift of the shipped
grid (``include_determinants``, ``include_dim6``). Two run-configuration switches, read from ``nucleationConf``:

  input_scheme  "zero_quartic"  lambda(mu0) = 0, with mu0 such that the minimum of the potential is at v (default);
                "CW"            the quartic of the 4D model file converted to MS-bar at mu0 = g v.
  mu3_choice    "gT"            mu3 = g T with the input coupling g, not run to the matching scale (default);
                "g3sq"          mu3 = g3^2.

The shipped grid is for zero_quartic, gT; with the other choices, attach a table (``nlo_correction``).
"""
import TL_conformal_u1_fermions as base
import dr_conformal_u1_fermions_matching as matching
from transitionlistener.eft_rate.one_field_rate_3d import ThreeDimensionalRate


class specific_potential(ThreeDimensionalRate, base.specific_potential):
    """Conformal U(1) with fermions whose rate comes from the 3D effective theory."""

    #: Name of the shipped grid: tab_data/nlo_grids/<name>.npz.
    nlo_grid_name = "conformal_u1_fermions"
    #: Above these couplings the NLO expansion, not the grid, limits the accuracy of the rate (a warning).
    nlo_expansion_limits = {"g": 0.85}

    def setConfigParameters(self) -> None:
        """The 3D defaults, in the conventions of the shipped grid."""
        super().setConfigParameters()
        self.config.nucleationConf.input_scheme = "zero_quartic"
        self.config.nucleationConf.mu3_choice = "gT"

    def field_dependent_mass_over_field(self) -> float:
        """The gauge boson, the heaviest mode that depends on the field."""
        return self.g

    def nlo_grid_point(self) -> dict:
        """The couplings the NLO grid of this model is a function of."""
        return {"g": self.g, "y": self.y}

    def two_loop_potential(self, p: dict):
        """The two-loop soft potential as a function of the 3D field, for ``p = soft_parameters(T)``."""
        return lambda phi3: matching.V3_soft_two_loop(phi3, p["g3sq"], p["mD2"], p["lambda_vl"], p["lambda_vll"],
                                                      p["mu3"])

    def two_loop_kinetic(self, p: dict):
        """Z_3 at NLO as a function of the 3D field (a diagnostic of the table, not part of the exponent)."""
        return lambda phi3: matching.Z3_soft_NLO(phi3, p["g3sq"], p["mD2"], p["lambda_vl"])

    def input_scale_GeV(self) -> float:
        """The scale mu0 of the couplings: at fixed couplings, everything in the 3D theory is a function of T/mu0."""
        if self.config.nucleationConf.input_scheme == "zero_quartic":
            return self.v_GeV / matching.vev_from_zero_quartic(self.g, self.y, 1.0)
        return matching.input_scale(self.g, self.v_GeV)

    def vacuum_energy_GeV4(self) -> float:
        """The energy difference of the two vacua at T = 0 (the zero-quartic expression)."""
        return matching.vacuum_energy_zero_quartic(
            self.g, self.y, self.v_GeV / matching.vev_from_zero_quartic(self.g, self.y, 1.0))

    def soft_parameters(self, T_GeV: float) -> dict:
        """Every matched 3D parameter at ``T_GeV`` as the configuration asks for: the dict of the matching module."""
        conf = self.config.nucleationConf
        scheme = conf.input_scheme
        if scheme == "zero_quartic":
            mu0 = self.v_GeV / matching.vev_from_zero_quartic(self.g, self.y, 1.0)
            mu3 = self.g * T_GeV if conf.mu3_choice == "gT" else None
            p = matching.matched_soft_parameters_zero_quartic(
                self.g, self.y, mu0, T_GeV, mu4_over_T=conf.mu4_over_T,
                matching_order=getattr(conf, "matching_order", "NLO"), mu3=mu3)
            return p
        return matching.matched_soft_parameters(
            self.g, self.y, self.l, self.v_GeV, T_GeV,
            mu4_over_T=conf.mu4_over_T,
            matching_order=getattr(conf, "matching_order", "NLO"),
            input_scheme=scheme)
