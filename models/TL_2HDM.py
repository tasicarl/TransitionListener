"""
Two-Higgs-doublet model (2HDM) with internal counterterm calculation.

The implementation mirrors the renormalisation prescription used in BSMPT for
the CP-conserving 2HDM (ClassPotentialR2HDM).

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import numpy as np

from transitionlistener import config, errors, generic_potential
from transitionlistener.constants import SM_FERMION_MASSES_GEV, SM_GAUGE_MASSES_GEV
from transitionlistener.generic_potential import console as tl_console
from transitionlistener.helper_functions import eigenvalues_2x2
from transitionlistener.particles import (
    GaugeBoson,
    MassSpectrum,
    Scalar,
    Fermion,
    Goldstone,
)
from transitionlistener.counterterms import CounterTerms
from transitionlistener.counterterms.twohdm_solver import compute_counterterms


class R2HDM(generic_potential.generic_potential):
    """CP-conserving 2HDM with internally computed counterterms."""

    _YUKAWA_TYPE_MAP = {
        1: "TypeI",
        2: "TypeII",
        3: "TypeLeptonSpecific",
        4: "TypeFlipped",
    }

    model_parameters = {
        "lambda1": {
            "default": 0.6,
            "plotname": r"$\lambda_1$",
            "min": -4 * np.pi,
            "max": 4 * np.pi,
        },
        "lambda2": {
            "default": 0.6,
            "plotname": r"$\lambda_2$",
            "min": -4 * np.pi,
            "max": 4 * np.pi,
        },
        "lambda3": {
            "default": 0.1,
            "plotname": r"$\lambda_3$",
            "min": -4 * np.pi,
            "max": 4 * np.pi,
        },
        "lambda4": {
            "default": -0.2,
            "plotname": r"$\lambda_4$",
            "min": -4 * np.pi,
            "max": 4 * np.pi,
        },
        "lambda5": {
            "default": 0.0,
            "plotname": r"$\lambda_5$",
            "min": -4 * np.pi,
            "max": 4 * np.pi,
        },
        "m12_sq_GeV2": {
            "default": 200.0**2,
            "plotname": r"$m_{12}^2$ [GeV$^2$]",
            "min": -np.inf,
            "max": np.inf,
        },
        "tan_beta": {
            "default": 2.0,
            "plotname": r"$\tan\beta$",
            "min": 0.05,
            "max": 50.0,
        },
        "yukawa_type": {
            "default": 2,
            "plotname": r"Type",
            "min": 1,
            "max": 4,
        },
        "v_GeV": {
            "default": 246.21965079413735,
            "plotname": r"$v$ [GeV]",
            "min": 0,
            "max": np.inf,
        },
    }

    def init(self, inputparam_dict, verbose: bool = True):  # noqa: D401
        self.mp = self.set_modelparams(inputparam_dict)

        self.lambda1 = float(self.mp["lambda1"]["value"])
        self.lambda2 = float(self.mp["lambda2"]["value"])
        self.lambda3 = float(self.mp["lambda3"]["value"])
        self.lambda4 = float(self.mp["lambda4"]["value"])
        self.lambda5 = float(self.mp["lambda5"]["value"])
        self.m12_sq_GeV2 = float(self.mp["m12_sq_GeV2"]["value"])
        self.tan_beta = float(self.mp["tan_beta"]["value"])
        self.yukawa_type = int(round(float(self.mp["yukawa_type"]["value"])))
        self.v_GeV = float(self.mp["v_GeV"]["value"])

        self.retrieved_input = np.array(
            [
                self.lambda1,
                self.lambda2,
                self.lambda3,
                self.lambda4,
                self.lambda5,
                self.m12_sq_GeV2,
                self.tan_beta,
                self.yukawa_type,
                self.v_GeV,
            ],
            dtype=float,
        )

        self.Ndim = 2
        self.verbose = verbose

        self.daisy = "ArnoldEspinosa"
        # The 2HDM is the only model in v2.0 where the natural internal scale
        # coincides with the SM electroweak vev, so we keep v_stable and the
        # conversion factor in their physical units rather than rescaling.
        self.v_stable = 246.21965079413735
        self.v = self.v_stable
        self.conversionFactor = 1.0
        self.deriv_order = max(self.deriv_order, 4)

        if not (1 <= self.yukawa_type <= 4):
            raise errors.InitPotentialError(
                f"Invalid Yukawa type {self.yukawa_type}. Expected integer in [1, 4]."
            )
        if np.isclose(self.tan_beta, 0.0):
            raise errors.InitPotentialError("tan_beta may not vanish in the 2HDM.")

        cos_beta = 1.0 / np.sqrt(1.0 + self.tan_beta**2)
        sin_beta = self.tan_beta * cos_beta
        if self.tan_beta < 0.0:
            sin_beta = -abs(sin_beta)
        self.cos_beta = cos_beta
        self.sin_beta = sin_beta
        self.beta = np.arctan(self.tan_beta)

        self.v1 = self.v * self.cos_beta
        self.v2 = self.v * self.sin_beta
        self.v1_GeV = self.v1 * self.conversionFactor
        self.v2_GeV = self.v2 * self.conversionFactor

        self.m12_sq = self.m12_sq_GeV2 / self.conversionFactor**2

        C_CosBetaSquared = 1.0 / (1 + self.tan_beta * self.tan_beta)
        C_CosBeta        = np.sqrt(C_CosBetaSquared)
        C_SinBetaSquared = self.tan_beta * self.tan_beta * C_CosBetaSquared
        C_SinBeta        = np.sqrt(C_SinBetaSquared)

        if (self.tan_beta < 0):
            C_SinBeta *= -1;

        print("RealMMix = ", self.m12_sq)
        print("TanBeta = ", self.tan_beta)
        print("scale = ", self.v_stable)
        print("L1 = ", self.lambda1)
        print("L2 = ", self.lambda2)
        print("L3 = ", self.lambda3)
        print("L4 = ", self.lambda4)
        print("L5 = ", self.lambda5)

        self.m11_sq = self.m12_sq * self.tan_beta - self.v_stable * self.v_stable * \
            C_SinBetaSquared * (self.lambda4 + self.lambda5 + self.lambda3) / 0.2e1 - \
            self.v_stable * self.v_stable * C_CosBetaSquared * self.lambda1 / 0.2e1
        self.m22_sq = self.m12_sq * 1.0 / self.tan_beta - self.v_stable * self.v_stable * \
            C_CosBetaSquared * (self.lambda4 + self.lambda5 + self.lambda3) / 0.2e1 - \
            self.v_stable * self.v_stable * C_SinBetaSquared * self.lambda2 / 0.2e1

        print("m11 = ", self.m11_sq)
        print("m22 = ", self.m22_sq)

        self.lambda345 = self.lambda3 + self.lambda4 + self.lambda5
        self.lambda34m5 = self.lambda3 + self.lambda4 - self.lambda5
        self.lambda45 = self.lambda4 + self.lambda5

        # self.m11_sq = (
        #     self.m12_sq * self.tan_beta
        #     - 0.5 * self.lambda1 * self.v1 * self.v1
        #     - 0.5 * self.lambda345 * self.v2 * self.v2
        # )
        # self.m22_sq = (
        #     self.m12_sq / self.tan_beta
        #     - 0.5 * self.lambda2 * self.v2 * self.v2
        #     - 0.5 * self.lambda345 * self.v1 * self.v1
        # )

        # SM inputs (GeV)
        self.mW_GeV = SM_GAUGE_MASSES_GEV["mW"]
        self.mZ_GeV = SM_GAUGE_MASSES_GEV["mZ"]
        self.mt_GeV = SM_FERMION_MASSES_GEV["m_t"]
        self.mb_GeV = SM_FERMION_MASSES_GEV["m_b"]
        self.mtau_GeV = SM_FERMION_MASSES_GEV["m_tau"]

        self.g2 = 2.0 * self.mW_GeV / self.v_GeV
        self.g1 = np.sqrt(4.0 * self.mZ_GeV**2 / self.v_GeV**2 - self.g2**2)

        self.yt = self.mt_GeV / self.v2_GeV
        self._bottom_uses_phi2 = self.yukawa_type in (1, 3)
        self._lepton_uses_phi2 = self.yukawa_type in (1, 4)
        v_bottom = self.v2_GeV if self._bottom_uses_phi2 else self.v1_GeV
        v_lepton = self.v2_GeV if self._lepton_uses_phi2 else self.v1_GeV
        self.yb = self.mb_GeV / v_bottom
        self.ytau = self.mtau_GeV / v_lepton

        coeff_gauge = 3.0 * (3.0 * self.g2**2 + self.g1**2)
        ct = np.sqrt(2.0) * self.yt
        cb = np.sqrt(2.0) * self.yb
        self.CTempC1 = (
            12.0 * self.lambda1 + 8.0 * self.lambda3 + 4.0 * self.lambda4 + coeff_gauge
        ) / 48.0
        self.CTempC2 = (
            12.0 * self.lambda2
            + 8.0 * self.lambda3
            + 4.0 * self.lambda4
            + coeff_gauge
            + 12.0 * ct * ct
        ) / 48.0
        if self._bottom_uses_phi2:
            self.CTempC2 += 12.0 * cb * cb / 48.0
        else:
            self.CTempC1 += 12.0 * cb * cb / 48.0

        self.X0 = np.array([self.v1, self.v2])
        self.renormScaleSq = self.v1 * self.v1 + self.v2 * self.v2
        self.mass_spectrum = self._build_mass_spectrum()

        counterterms = self._initialize_counterterms()
        self.dlambda1 = counterterms.dlambda1
        self.dlambda2 = counterterms.dlambda2
        self.dlambda3 = counterterms.dlambda3
        self.dlambda4 = counterterms.dlambda4
        self.dlambda5 = counterterms.dlambda5
        self.dm11_sq = counterterms.dm11_sq
        self.dm22_sq = counterterms.dm22_sq
        self.dm12_sq = counterterms.dm12_sq

        inv_cf_sq = 1.0 / (self.conversionFactor * self.conversionFactor)
        self.dm11_sq_GeV2 = self.dm11_sq * inv_cf_sq
        self.dm22_sq_GeV2 = self.dm22_sq * inv_cf_sq
        self.dm12_sq_GeV2 = self.dm12_sq * inv_cf_sq

        mh_sq, mH_sq = self._cp_even_eigenvalues(self.v1, self.v2, 0.0)
        _, mA_sq = self._cp_odd_eigenvalues(self.v1, self.v2, 0.0)
        _, mHpm_sq = self._charged_eigenvalues(self.v1, self.v2, 0.0)

        self.derived_parameters = {
            "v1_GeV": self.v1_GeV,
            "v2_GeV": self.v2_GeV,
            "m11_sq_GeV2": self.m11_sq * self.conversionFactor**2,
            "m22_sq_GeV2": self.m22_sq * self.conversionFactor**2,
            "m12_sq_GeV2": self.m12_sq_GeV2,
            "mh_GeV": np.sqrt(max(mh_sq, 0.0)) * self.conversionFactor,
            "mH_GeV": np.sqrt(max(mH_sq, 0.0)) * self.conversionFactor,
            "mA_GeV": np.sqrt(max(mA_sq, 0.0)) * self.conversionFactor,
            "mHp_GeV": np.sqrt(max(mHpm_sq, 0.0)) * self.conversionFactor,
            "delta_m11_sq_GeV2": self.dm11_sq_GeV2,
            "delta_m22_sq_GeV2": self.dm22_sq_GeV2,
            "delta_m12_sq_GeV2": self.dm12_sq_GeV2,
            "delta_lambda1": self.dlambda1,
            "delta_lambda2": self.dlambda2,
            "delta_lambda3": self.dlambda3,
            "delta_lambda4": self.dlambda4,
            "delta_lambda5": self.dlambda5,
        }

    # ------------------------------------------------------------------
    # Counterterm determination
    # ------------------------------------------------------------------
    def _initialize_counterterms(self) -> CounterTerms:
        (
            counterterms,
            grad_internal,
            H_internal,
            residual_internal,
        ) = compute_counterterms(self)
        residual_norm = float(np.max(np.abs(residual_internal)))
        if residual_norm > 1e-6:
            tl_console.print(
                "[yellow]Warning: Renormalised Coleman–Weinberg conditions "
                f"deviate by {residual_norm:.3e} in internal units "
                "(expected to be near zero).[/yellow]"
            )
        self._cw_gradient = grad_internal
        self._cw_hessian = H_internal
        self._cw_residual = residual_internal
        return counterterms

    # ------------------------------------------------------------------
    # Thermal mass shifts and eigenvalues in the 2D field space
    # ------------------------------------------------------------------
    def _scalar_daisy_shifts(self, T):
        if self.daisy not in ("ArnoldEspinosa", "Parwani"):
            return 0.0, 0.0
        T_arr = np.asarray(T, dtype=float)
        T2 = T_arr * T_arr
        Pi1 = self.CTempC1 * T2
        Pi2 = self.CTempC2 * T2
        return Pi1, Pi2

    def _scalar_mass_entries(self, phi1, phi2, sector):
        phi1 = np.asarray(phi1, dtype=float)
        phi2 = np.asarray(phi2, dtype=float)
        if sector == "cp_even":
            a = (
                self.m11_sq
                + 1.5 * self.lambda1 * phi1**2
                + 0.5 * self.lambda345 * phi2**2
            )
            c = (
                self.m22_sq
                + 1.5 * self.lambda2 * phi2**2
                + 0.5 * self.lambda345 * phi1**2
            )
            b = -self.m12_sq + self.lambda345 * phi1 * phi2
        elif sector == "cp_odd":
            a = (
                self.m11_sq
                + 0.5 * self.lambda1 * phi1**2
                + 0.5 * self.lambda34m5 * phi2**2
            )
            c = (
                self.m22_sq
                + 0.5 * self.lambda2 * phi2**2
                + 0.5 * self.lambda34m5 * phi1**2
            )
            b = -self.m12_sq + self.lambda5 * phi1 * phi2
        elif sector == "charged":
            a = (
                self.m11_sq
                + 0.5 * self.lambda1 * phi1**2
                + 0.5 * self.lambda3 * phi2**2
            )
            c = (
                self.m22_sq
                + 0.5 * self.lambda2 * phi2**2
                + 0.5 * self.lambda3 * phi1**2
            )
            b = -self.m12_sq + 0.5 * self.lambda45 * phi1 * phi2
        else:
            raise ValueError(f"Unknown scalar sector '{sector}'.")
        return a, b, c

    def _cp_even_eigenvalues(self, phi1, phi2, T):
        a, b, c = self._scalar_mass_entries(phi1, phi2, "cp_even")
        Pi1, Pi2 = self._scalar_daisy_shifts(T)
        return eigenvalues_2x2(a + Pi1, b, c + Pi2)

    def _cp_odd_eigenvalues(self, phi1, phi2, T):
        a, b, c = self._scalar_mass_entries(phi1, phi2, "cp_odd")
        Pi1, Pi2 = self._scalar_daisy_shifts(T)
        return eigenvalues_2x2(a + Pi1, b, c + Pi2)

    def _charged_eigenvalues(self, phi1, phi2, T):
        a, b, c = self._scalar_mass_entries(phi1, phi2, "charged")
        Pi1, Pi2 = self._scalar_daisy_shifts(T)
        return eigenvalues_2x2(a + Pi1, b, c + Pi2)

    # ------------------------------------------------------------------
    # Potentials used by TransitionListener
    # ------------------------------------------------------------------
    def V0(self, X):
        X = np.asarray(X, dtype=float)
        phi1 = X[..., 0]
        phi2 = X[..., 1]
        tree = 0.5 * self.m11_sq * phi1**2 + 0.5 * self.m22_sq * phi2**2
        tree -= self.m12_sq * phi1 * phi2
        tree += 0.125 * self.lambda1 * phi1**4
        tree += 0.125 * self.lambda2 * phi2**4
        tree += 0.25 * self.lambda345 * phi1**2 * phi2**2
        return tree

    def Vct(self, X):
        X = np.asarray(X, dtype=float)
        phi1 = X[..., 0]
        phi2 = X[..., 1]
        counter = 0.5 * self.dm11_sq * phi1**2 + 0.5 * self.dm22_sq * phi2**2
        counter -= self.dm12_sq * phi1 * phi2
        counter += 0.125 * self.dlambda1 * phi1**4
        counter += 0.125 * self.dlambda2 * phi2**4
        counter += (
            0.25 * (self.dlambda3 + self.dlambda4 + self.dlambda5) * phi1**2 * phi2**2
        )
        return counter

    # ------------------------------------------------------------------
    # Spectrum construction (2D field space)
    # ------------------------------------------------------------------
    def _build_mass_spectrum(self) -> MassSpectrum:

        def boson_mass_function(X, T):
            X = np.asarray(X)
            T2 = T * T
            phi1 = X[..., 0]
            phi2 = X[..., 1]
            phi_sq = phi1**2 + phi2**2

            Nbosons = 12  # How many entries?
            M2 = np.empty(phi1.shape + (Nbosons,))

            # Scalars and Goldstones
            cp_even_light, cp_even_heavy = self._cp_even_eigenvalues(phi1, phi2, T)
            cp_odd_small, cp_odd_large = self._cp_odd_eigenvalues(phi1, phi2, T)
            charged_small, charged_large = self._charged_eigenvalues(phi1, phi2, T)

            M2[..., 0] = cp_even_light
            M2[..., 1] = cp_even_heavy
            M2[..., 2] = cp_odd_large
            M2[..., 3] = cp_odd_small
            M2[..., 4] = charged_large
            M2[..., 5] = charged_small

            # Gauge bosons
            Pi_WL = 2 * self.g2**2 * T2
            Pi_BL = 2 * self.g1**2 * T2
            a = 0.25 * self.g2**2 * phi_sq + Pi_WL
            b = -0.25 * self.g1 * self.g2 * phi_sq
            c = 0.25 * self.g1**2 * phi_sq + Pi_BL
            discr = np.sqrt(np.maximum(a**2 + 4.0 * b**2 - 2.0 * a * c + c**2, 0.0))
            mZ2 = 0.25 * (self.g1**2 + self.g2**2) * phi_sq
            mZ2L = 0.5 * (a + c + discr)
            mA2L = 0.5 * (a + c - discr)
            mA2 = 0.5 * (a + c - discr)
            mW2 = 0.25 * self.g2**2 * phi_sq
            mW2L = mW2 + Pi_WL

            M2[..., 6] = mZ2
            M2[..., 7] = mZ2L
            M2[..., 8] = mA2
            M2[..., 9] = mA2L
            M2[..., 10] = mW2
            M2[..., 11] = mW2L

            return M2

        scalars = [
            Scalar(name="m_h", latex_name=r"$m_h$", dof=1, is_SM=True),
            Scalar(name="m_H", latex_name=r"$m_H$", dof=1, is_SM=False),
            Scalar(name="m_A", latex_name=r"$m_A$", dof=1, is_SM=False),
            Goldstone(name="m_G0", latex_name=r"$m_{G^0}$", dof=1, is_SM=True),
            Scalar(name="m_Hpm", latex_name=r"$m_{H^\pm}$", dof=2, is_SM=False),
            Goldstone(name="m_Gpm", latex_name=r"$m_{G^\pm}$", dof=2, is_SM=True),
        ]

        sqrt_g = np.sqrt(self.g1**2 + self.g2**2) / 2.0
        gauge_bosons = [
            GaugeBoson(name="m_ZT", latex_name=r"$m_{Z, T}$", dof=2, gauge_coupling=sqrt_g, is_SM=True,),
            GaugeBoson(name="m_ZL", latex_name=r"$m_{Z, L}$", dof=1, gauge_coupling=sqrt_g, is_SM=True,),
            GaugeBoson(name="m_AT", latex_name=r"$m_{A, T}$", dof=2, gauge_coupling=sqrt_g, is_SM=True,),
            GaugeBoson(name="m_AL", latex_name=r"$m_{A, L}$", dof=1, gauge_coupling=sqrt_g, is_SM=True,),
            GaugeBoson(name="m_WT", latex_name=r"$m_{W, T}$", dof=4, gauge_coupling=self.g2 / 2.0, is_SM=True,),
            GaugeBoson(name="m_WL", latex_name=r"$m_{W, L}$", dof=2, gauge_coupling=self.g2 / 2.0, is_SM=True,),
        ]

        fermions = [
            Fermion(name="m_t", latex_name=r"$m_t$", dof=12, is_SM=True),
            Fermion(name="m_b", latex_name=r"$m_b$", dof=12, is_SM=True),
            Fermion(name="m_tau", latex_name=r"$m_{\\tau}$", dof=4, is_SM=True),
        ]

        def fermion_mass_function(X):
            X = np.asarray(X, dtype=float)
            phi1 = X[..., 0]
            phi2 = X[..., 1]

            Nfermions = 3
            M2 = np.empty(phi1.shape + (Nfermions,))

            mt_sq = (self.yt * phi2) ** 2
            bottom_field = phi2 if self._bottom_uses_phi2 else phi1
            lepton_field = phi2 if self._lepton_uses_phi2 else phi1
            mb_sq = (self.yb * bottom_field) ** 2
            mtau_sq = (self.ytau * lepton_field) ** 2

            M2[..., 0] = mt_sq
            M2[..., 1] = mb_sq
            M2[..., 2] = mtau_sq

            return M2


        spectrum = MassSpectrum(scalars=scalars, gaugeBosons=gauge_bosons,
                                boson_massSq_fn=boson_mass_function,
                                fermions=fermions, fermion_massSq_fn=fermion_mass_function)

        return spectrum

    # ------------------------------------------------------------------
    # Miscellaneous
    # ------------------------------------------------------------------
    def forbidPhaseCrit(self, X):
        return False

    def approxZeroTMin(self):
        return [np.array([self.v1, self.v2])]
