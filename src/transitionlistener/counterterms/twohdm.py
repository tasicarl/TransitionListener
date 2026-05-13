"""
SymPy curvature tensor generator for the CP-conserving 2HDM.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from itertools import product
from typing import Dict, Tuple

import sympy as sp

from . import CountertermGenerator, register_generator


class TwoHDMCountertermGenerator(CountertermGenerator):
    r"""Generate symbolic curvature tensors for the CP-conserving 2HDM.

    The generator rewrites the two Higgs doublets in an eight-component real
    basis and differentiates the tree-level potential

    .. math::
       V = m_{11}^2\,\Phi_1^\dagger\Phi_1
         + m_{22}^2\,\Phi_2^\dagger\Phi_2
         - m_{12}^2\,(\Phi_1^\dagger\Phi_2 + \Phi_2^\dagger\Phi_1)
         + \frac{\lambda_1}{2}(\Phi_1^\dagger\Phi_1)^2
         + \frac{\lambda_2}{2}(\Phi_2^\dagger\Phi_2)^2
         + \lambda_3(\Phi_1^\dagger\Phi_1)(\Phi_2^\dagger\Phi_2)
         + \lambda_4(\Phi_1^\dagger\Phi_2)(\Phi_2^\dagger\Phi_1)
         + \frac{\lambda_5}{2}\!\left[(\Phi_1^\dagger\Phi_2)^2 + (\Phi_2^\dagger\Phi_1)^2\right]

    to obtain the Higgs, gauge, and Yukawa tensors consumed by the runtime
    Coleman-Weinberg solver.
    """

    model_key = "models.TL_2HDM"
    alias = "TL_2HDM"
    output_filename = "2hdm_curvature_tensors.json"

    def _field_basis(self) -> Tuple[list[sp.Symbol], Dict[str, sp.Symbol]]:
        """Return the eight real scalar fields spanning the 2HDM gauge basis."""
        phi1p_R, phi1p_I, phi2p_R, phi2p_I, phi1_R, phi1_I, phi2_R, phi2_I = sp.symbols(
            "phi1p_R phi1p_I phi2p_R phi2p_I phi1_R phi1_I phi2_R phi2_I", real=True
        )
        fields = [
            phi1p_R,
            phi1p_I,
            phi2p_R,
            phi2p_I,
            phi1_R,
            phi1_I,
            phi2_R,
            phi2_I,
        ]
        return fields, {str(sym): sym for sym in fields}

    def _parameters(self) -> Dict[str, sp.Symbol]:
        """Create the symbolic model, CKM, and gauge parameters appearing in the tensors."""
        lambda1, lambda2, lambda3, lambda4, lambda5 = sp.symbols(
            "lambda1 lambda2 lambda3 lambda4 lambda5", real=True
        )
        m11sq, m22sq, m12sq = sp.symbols("m11sq m22sq m12sq", real=True)
        tan_beta = sp.symbols("tan_beta", real=True)
        v1, v2 = sp.symbols("v1 v2", real=True)
        yuk_type = sp.symbols("yukawa_type", integer=True)
        params: Dict[str, sp.Symbol] = {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda4": lambda4,
            "lambda5": lambda5,
            "m11sq": m11sq,
            "m22sq": m22sq,
            "m12sq": m12sq,
            "tan_beta": tan_beta,
            "v1": v1,
            "v2": v2,
            "yukawa_type": yuk_type,
        }
        Cg, Cgs = sp.symbols("C_g C_gs", real=True)
        params["Cg"] = Cg
        params["Cgs"] = Cgs

        params.update(
            {
                "m_e": sp.symbols("C_MassElectron", real=True),
                "m_mu": sp.symbols("C_MassMu", real=True),
                "m_tau": sp.symbols("C_MassTau", real=True),
                "m_u": sp.symbols("C_MassUp", real=True),
                "m_c": sp.symbols("C_MassCharm", real=True),
                "m_t": sp.symbols("C_MassTop", real=True),
                "m_d": sp.symbols("C_MassDown", real=True),
                "m_s": sp.symbols("C_MassStrange", real=True),
                "m_b": sp.symbols("C_MassBottom", real=True),
            }
        )

        Vud, Vus, Vub, Vcd, Vcs, Vcb, Vtd, Vts, Vtb = sp.symbols("Vud Vus Vub Vcd Vcs Vcb Vtd Vts Vtb")
        params.update(
            {
                "Vud": Vud,
                "Vus": Vus,
                "Vub": Vub,
                "Vcd": Vcd,
                "Vcs": Vcs,
                "Vcb": Vcb,
                "Vtd": Vtd,
                "Vts": Vts,
                "Vtb": Vtb,
            }
        )
        return params

    def _higgs_potential(self, fields, params):
        r"""Construct the CP-conserving 2HDM tree-level scalar potential.

        The returned SymPy expression implements

        .. math::
           V(\Phi_1,\Phi_2)
           = m_{11}^2\,\Phi_1^\dagger\Phi_1
             + m_{22}^2\,\Phi_2^\dagger\Phi_2
             - m_{12}^2(\Phi_1^\dagger\Phi_2 + \Phi_2^\dagger\Phi_1)
             + \frac{\lambda_1}{2}(\Phi_1^\dagger\Phi_1)^2
             + \frac{\lambda_2}{2}(\Phi_2^\dagger\Phi_2)^2
             + \lambda_3(\Phi_1^\dagger\Phi_1)(\Phi_2^\dagger\Phi_2)
             + \lambda_4(\Phi_1^\dagger\Phi_2)(\Phi_2^\dagger\Phi_1)
             + \frac{\lambda_5}{2}\!\left[(\Phi_1^\dagger\Phi_2)^2 + (\Phi_2^\dagger\Phi_1)^2\right].
        """
        (
            phi1p_R,
            phi1p_I,
            phi2p_R,
            phi2p_I,
            phi1_R,
            phi1_I,
            phi2_R,
            phi2_I,
        ) = fields

        phi1 = sp.Matrix(
            [
                (phi1p_R + sp.I * phi1p_I) / sp.sqrt(2),
                (phi1_R + sp.I * phi1_I) / sp.sqrt(2),
            ]
        )
        phi2 = sp.Matrix(
            [
                (phi2p_R + sp.I * phi2p_I) / sp.sqrt(2),
                (phi2_R + sp.I * phi2_I) / sp.sqrt(2),
            ]
        )
        phi1_sq = sp.simplify((phi1.conjugate().T * phi1)[0])
        phi2_sq = sp.simplify((phi2.conjugate().T * phi2)[0])
        phi12 = sp.simplify((phi1.conjugate().T * phi2)[0])
        phi21 = sp.simplify((phi2.conjugate().T * phi1)[0])

        lam1 = params["lambda1"]
        lam2 = params["lambda2"]
        lam3 = params["lambda3"]
        lam4 = params["lambda4"]
        lam5 = params["lambda5"]
        m11sq = params["m11sq"]
        m22sq = params["m22sq"]
        m12sq = params["m12sq"]

        return sp.simplify(
            m11sq * phi1_sq
            + m22sq * phi2_sq
            - m12sq * (phi12 + phi21)
            + (lam1 / 2) * phi1_sq**2
            + (lam2 / 2) * phi2_sq**2
            + lam3 * phi1_sq * phi2_sq
            + lam4 * phi12 * phi21
            + (lam5 / 2) * (phi12**2 + phi21**2)
        )

    def _build_gauge_curvature(self, params):
        r"""Build the gauge tensor :math:`G_{abij}` in the real Higgs basis.

        The tensor is defined so that the gauge-boson mass matrix can be written
        as

        .. math::
           (M_G^2)_{ab} = \frac{1}{2} G_{abij}\, \phi_i \phi_j.
        """
        Cg = params["Cg"]
        Cgs = params["Cgs"]
        half_cg_sq = sp.Rational(1, 2) * Cg * Cg
        half_cgs_sq = sp.Rational(1, 2) * Cgs * Cgs
        half_cg_cgs = sp.Rational(1, 2) * Cg * Cgs

        tensor = sp.MutableDenseNDimArray.zeros(4, 4, 8, 8)

        for a in range(3):
            for i in range(8):
                tensor[a, a, i, i] = half_cg_sq

        for i in range(8):
            tensor[3, 3, i, i] = half_cgs_sq

        def set_pair(a: int, b: int, i: int, j: int, value: sp.Expr) -> None:
            tensor[a, b, i, j] = value
            tensor[b, a, j, i] = value

        positive_pairs = [
            (0, 3, [(0, 4), (1, 5), (2, 6), (3, 7), (4, 0), (5, 1), (6, 2), (7, 3)]),
            (2, 3, [(0, 0), (1, 1), (2, 2), (3, 3)]),
        ]
        for a, b, entries in positive_pairs:
            for i, j in entries:
                set_pair(a, b, i, j, half_cg_cgs)

        negative_pairs = [
            (2, 3, [(4, 4), (5, 5), (6, 6), (7, 7)]),
        ]
        for a, b, entries in negative_pairs:
            for i, j in entries:
                set_pair(a, b, i, j, -half_cg_cgs)

        entries_13 = [
            ((0, 5), 1),
            ((1, 4), -1),
            ((2, 7), 1),
            ((3, 6), -1),
            ((4, 1), -1),
            ((5, 0), 1),
            ((6, 3), -1),
            ((7, 2), 1),
        ]
        for (i, j), sign in entries_13:
            set_pair(1, 3, i, j, sign * half_cg_cgs)

        entries_30 = [
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
            (4, 0),
            (5, 1),
            (6, 2),
            (7, 3),
        ]
        for i, j in entries_30:
            set_pair(3, 0, i, j, half_cg_cgs)

        entries_31 = [
            ((0, 5), 1),
            ((1, 4), -1),
            ((2, 7), 1),
            ((3, 6), -1),
            ((4, 1), -1),
            ((5, 0), 1),
            ((6, 3), -1),
            ((7, 2), 1),
        ]
        for (i, j), sign in entries_31:
            set_pair(3, 1, i, j, sign * half_cg_cgs)

        result = {}
        for a in range(4):
            for b in range(4):
                for i in range(8):
                    for j in range(8):
                        value = sp.simplify(tensor[a, b, i, j])
                        if value != 0:
                            result[(a, b, i, j)] = value
        return result

    def _lepton_curvature(self, params, yukawa_type: str):
        r"""Build the lepton Yukawa tensor :math:`Y^{(\ell)}_{ab i}` for a given 2HDM type."""
        NH = 8
        NL = 9

        v1 = params["v1"]
        v2 = params["v2"]
        m_e = params["m_e"]
        m_mu = params["m_mu"]
        m_tau = params["m_tau"]

        if yukawa_type in ("TypeI", "TypeFlipped"):
            vL = v2
        elif yukawa_type in ("TypeII", "TypeLeptonSpecific"):
            vL = v1
        else:
            raise ValueError(f"Unknown Yukawa type {yukawa_type}")

        YIJRL = sp.Matrix.zeros(NL, NL)
        YIJSL = sp.Matrix.zeros(NL, NL)
        YIJEL = sp.Matrix.zeros(NL, NL)
        YIJPL = sp.Matrix.zeros(NL, NL)

        masses = [m_e, m_mu, m_tau]
        lepton_pairs = [(1, 6), (3, 7), (5, 8)]
        for (i, j), mass in zip(lepton_pairs, masses):
            YIJRL[i, j] = mass / vL
        for i in range(NL):
            for j in range(i):
                YIJRL[i, j] = YIJRL[j, i]

        YIJSL[0, 1] = m_e / vL
        YIJSL[2, 3] = m_mu / vL
        YIJSL[4, 5] = m_tau / vL
        for i in range(NL):
            for j in range(i):
                YIJSL[i, j] = YIJSL[j, i]

        YIJPL = sp.I * YIJSL
        YIJEL = sp.I * YIJRL

        tensor = sp.MutableDenseNDimArray.zeros(NL, NL, NH)

        if yukawa_type in ("TypeI", "TypeFlipped"):
            for i in range(NL):
                for j in range(NL):
                    if YIJRL[i, j] != 0:
                        tensor[i, j, 2] = YIJRL[i, j]
                        tensor[i, j, 3] = YIJEL[i, j]
                    if YIJSL[i, j] != 0:
                        tensor[i, j, 6] = YIJSL[i, j]
                        tensor[i, j, 7] = YIJPL[i, j]
        else:
            for i in range(NL):
                for j in range(NL):
                    if YIJRL[i, j] != 0:
                        tensor[i, j, 0] = YIJRL[i, j]
                        tensor[i, j, 1] = YIJEL[i, j]
                    if YIJSL[i, j] != 0:
                        tensor[i, j, 4] = YIJSL[i, j]
                        tensor[i, j, 5] = YIJPL[i, j]

        result = {}
        for a in range(NL):
            for b in range(NL):
                for i in range(NH):
                    val = sp.simplify(tensor[a, b, i])
                    if val != 0:
                        result[(a, b, i)] = val
        return result

    def _quark_curvature(self, params, type_label: str):
        r"""Build the quark Yukawa tensor :math:`Y^{(q)}_{ab i}` for a given 2HDM type."""
        NH = 8
        NQ = 12

        v1 = params["v1"]
        v2 = params["v2"]
        VCKM = sp.Matrix(
            [
                [params["Vud"], params["Vus"], params["Vub"]],
                [params["Vcd"], params["Vcs"], params["Vcb"]],
                [params["Vtd"], params["Vts"], params["Vtb"]],
            ]
        )

        up_masses = [params["m_u"], params["m_c"], params["m_t"]]
        down_masses = [params["m_d"], params["m_s"], params["m_b"]]

        if type_label in ("TypeI", "TypeLeptonSpecific"):
            vD = v2
        elif type_label in ("TypeII", "TypeFlipped"):
            vD = v1
        else:
            raise ValueError(f"Unknown Yukawa type {type_label}")

        YIJR2 = sp.Matrix.zeros(NQ, NQ)
        YIJE2 = sp.Matrix.zeros(NQ, NQ)
        YIJS2 = sp.Matrix.zeros(NQ, NQ)
        YIJP2 = sp.Matrix.zeros(NQ, NQ)
        YIJRD = sp.Matrix.zeros(NQ, NQ)
        YIJED = sp.Matrix.zeros(NQ, NQ)
        YIJSD = sp.Matrix.zeros(NQ, NQ)
        YIJPD = sp.Matrix.zeros(NQ, NQ)

        for up_idx, mass in enumerate(up_masses):
            for down_idx in range(3):
                V_elem = VCKM[up_idx, down_idx]
                YIJR2[up_idx, 9 + down_idx] = -sp.conjugate(V_elem) * mass / v2
            YIJS2[up_idx, 6 + up_idx] = mass / v2

        for down_idx, mass in enumerate(down_masses):
            YIJSD[3 + down_idx, 9 + down_idx] = mass / vD

        for down_idx, mass in enumerate(down_masses):
            for up_idx in range(3):
                V_elem = VCKM[up_idx, down_idx]
                YIJRD[3 + down_idx, 6 + up_idx] = V_elem * mass / vD

        for i in range(NQ):
            for j in range(i):
                YIJR2[i, j] = YIJR2[j, i]
                YIJS2[i, j] = YIJS2[j, i]
                YIJRD[i, j] = YIJRD[j, i]
                YIJSD[i, j] = YIJSD[j, i]

        YIJP2 = -sp.I * YIJS2
        YIJE2 = -sp.I * YIJR2
        YIJPD = sp.I * YIJSD
        YIJED = sp.I * YIJRD

        tensor = sp.MutableDenseNDimArray.zeros(NQ, NQ, NH)
        for i in range(NQ):
            for j in range(NQ):
                tensor[i, j, 2] = YIJR2[i, j]
                tensor[i, j, 3] = YIJE2[i, j]
                tensor[i, j, 6] = YIJS2[i, j]
                tensor[i, j, 7] = YIJP2[i, j]
                if type_label in ("TypeI", "TypeLeptonSpecific"):
                    tensor[i, j, 2] += YIJRD[i, j]
                    tensor[i, j, 3] += YIJED[i, j]
                    tensor[i, j, 6] += YIJSD[i, j]
                    tensor[i, j, 7] += YIJPD[i, j]
                else:
                    tensor[i, j, 0] += YIJRD[i, j]
                    tensor[i, j, 1] += YIJED[i, j]
                    tensor[i, j, 4] += YIJSD[i, j]
                    tensor[i, j, 5] += YIJPD[i, j]

        result = {}
        for a in range(NQ):
            for b in range(NQ):
                for i in range(NH):
                    val = sp.simplify(tensor[a, b, i])
                    if val != 0:
                        result[(a, b, i)] = val
        return result

    def _curvature_tensor(self, expr: sp.Expr, fields, order: int):
        r"""Differentiate ``expr`` and store the non-vanishing Taylor coefficients.

        For derivative order :math:`n`, the returned dictionary contains the
        tensors

        .. math::
           \Lambda_{i_1\cdots i_n}
           = \left.\frac{\partial^n V}{\partial \phi_{i_1}\cdots\partial \phi_{i_n}}\right|_{\phi=0}.
        """
        zero_subs = {f: 0 for f in fields}

        def simplify(val: sp.Expr) -> sp.Expr:
            if isinstance(val, sp.Basic):
                return sp.simplify(val)
            return sp.simplify(sp.sympify(val))

        if order == 1:
            data = {}
            for i, fi in enumerate(fields):
                val = simplify(sp.diff(expr, fi).subs(zero_subs))
                if val != 0:
                    data[(i,)] = val
            return data

        if order == 2:
            data = {}
            for i, fi in enumerate(fields):
                for j, fj in enumerate(fields[i:], start=i):
                    val = simplify(sp.diff(expr, fi, fj).subs(zero_subs))
                    if val != 0:
                        data[(i, j)] = val
            return data

        if order == 3:
            data = {}
            for i, fi in enumerate(fields):
                for j, fj in enumerate(fields[i:], start=i):
                    for k, fk in enumerate(fields[j:], start=j):
                        val = simplify(sp.diff(expr, fi, fj, fk).subs(zero_subs))
                        if val != 0:
                            data[(i, j, k)] = val
            return data

        if order == 4:
            data = {}
            for i, fi in enumerate(fields):
                for j, fj in enumerate(fields[i:], start=i):
                    for k, fk in enumerate(fields[j:], start=j):
                        for l, fl in enumerate(fields[k:], start=k):
                            val = simplify(sp.diff(expr, fi, fj, fk, fl).subs(zero_subs))
                            if val != 0:
                                data[(i, j, k, l)] = val
            return data

        raise ValueError("Supported derivative order is 1..4")

    def build_dataset(self):
        """Assemble the full 2HDM tensor payload and accompanying metadata."""
        fields, field_map = self._field_basis()
        params = self._parameters()
        potential = self._higgs_potential(fields, params)

        curvature = {
            "Curvature_Higgs_L1": self._curvature_tensor(potential, fields, order=1),
            "Curvature_Higgs_L2": self._curvature_tensor(potential, fields, order=2),
            "Curvature_Higgs_L3": self._curvature_tensor(potential, fields, order=3),
            "Curvature_Higgs_L4": self._curvature_tensor(potential, fields, order=4),
            "Curvature_Gauge_G2H2": self._build_gauge_curvature(params),
        }

        type_labels = ("TypeI", "TypeII", "TypeLeptonSpecific", "TypeFlipped")
        curvature["Curvature_Lepton_F2H1"] = {label: self._lepton_curvature(params, label) for label in type_labels}
        curvature["Curvature_Quark_F2H1"] = {label: self._quark_curvature(params, label) for label in type_labels}

        metadata = {
            "generated_by": f"{__name__}.{self.__class__.__name__}",
            "sympy_version": sp.__version__,
            "field_basis": list(field_map.keys()),
            "parameters": list(params.keys()),
            "yukawa_types": list(type_labels),
            "parameter_symbol_names": {name: str(sym) for name, sym in params.items()},
        }
        return metadata, curvature


register_generator(TwoHDMCountertermGenerator)
