"""
Coleman–Weinberg gradient and Hessian utilities.

The functions defined here implement the analytic derivative formulas used in
BSMPT and ported into TransitionListener.  They operate on the curvature
tensors evaluated at the vacuum configuration and provide gradient / Hessian
matrices expressed in the physical field basis (i.e. after rotating into the
mass eigenstates).

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from transitionlistener.constants import CW_CONSTANTS
from .tensors import ensure_symmetric

EPS_LOOP = 1.0 / (16.0 * math.pi * math.pi)


def _log_term(m2: float, scale_sq: float, counterterm: float) -> float:
    r"""Return the Coleman-Weinberg logarithmic kernel
    :math:`\log(m^2/\mu^2) - c + 1/2`, regularised at ``m^2 = 0``."""
    if m2 == 0.0:
        return -counterterm + 0.5
    return math.log(m2 / scale_sq) - counterterm + 0.5


def _fbase(m2a: float, m2b: float, scale: float) -> float:
    """
    Helper matching ``Class_Potential_Origin::fbase`` in BSMPT.
    """

    if m2a == 0.0 and m2b == 0.0:
        return 1.0

    tol = 1e-5
    log_a = math.log(m2a) - 2.0 * math.log(scale) if m2a != 0.0 else 0.0
    if abs(m2a - m2b) > tol:
        log_b = math.log(m2b) - 2.0 * math.log(scale) if m2b != 0.0 else 0.0
        if m2a == 0.0:
            return log_b
        if m2b == 0.0:
            return log_a
        return (log_a * m2a - log_b * m2b) / (m2a - m2b)
    return 1.0 + log_a


def coleman_weinberg_derivatives(
    h2: np.ndarray,
    h3: np.ndarray,
    h4: np.ndarray,
    gauge_curv: np.ndarray,
    quark_curv: np.ndarray,
    lepton_curv: np.ndarray,
    vev: np.ndarray,
    *,
    scale: float,
    scalar_threshold: float = 1e-5,
    fermion_threshold: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Coleman–Weinberg gradient and Hessian in the physical basis.

    Parameters
    ----------
    h2, h3, h4:
        Higgs curvature tensors (quadratic, cubic, quartic) in the gauge basis.
    gauge_curv:
        Gauge curvature tensor :math:`G_{abij}`.
    quark_curv, lepton_curv:
        Fermionic Yukawa curvature tensors :math:`Y_{ij\\alpha}`.
    vev:
        Vacuum expectation value vector in the Higgs field basis (GeV).
    scale:
        Renormalisation scale :math:`\\mu` used in the Coleman–Weinberg logs.
    scalar_threshold, fermion_threshold:
        Numerical cut-offs below which eigenvalues are treated as zero.

    Returns
    -------
    (gradient, hessian):
        Coleman–Weinberg first and second derivatives in the physical basis,
        expressed in the same units as the input tensors.  The overall
        Coleman–Weinberg loop factor :math:`1/(16 \\pi^2)` is included.
    """

    scale_sq = scale * scale

    # ------------------------------------------------------------------
    # Mass matrices and rotations
    # ------------------------------------------------------------------
    mass_h = h2.copy()
    for k in range(vev.shape[0]):
        mass_h += h3[:, :, k] * vev[k]
    for k in range(vev.shape[0]):
        for l in range(vev.shape[0]):
            mass_h += 0.5 * h4[:, :, k, l] * vev[k] * vev[l]

    mass_g = np.zeros((gauge_curv.shape[0], gauge_curv.shape[1]), dtype=float)
    for a in range(gauge_curv.shape[0]):
        for b in range(gauge_curv.shape[1]):
            total = 0.0
            for i in range(vev.shape[0]):
                for j in range(vev.shape[0]):
                    total += gauge_curv[a, b, i, j] * vev[i] * vev[j]
            mass_g[a, b] = 0.5 * total

    MIJ_quark = np.tensordot(quark_curv, vev, axes=(2, 0))
    MIJ_lepton = np.tensordot(lepton_curv, vev, axes=(2, 0))

    mass_quark = MIJ_quark.conj().T @ MIJ_quark
    mass_lepton = MIJ_lepton.conj().T @ MIJ_lepton

    eigvals_h, eigvecs_h = np.linalg.eigh(mass_h)
    HiggsRot = np.real_if_close(eigvecs_h.T, tol=1e8)
    mass_h_sq = eigvals_h.copy()
    mass_h_sq[np.abs(mass_h_sq) < scalar_threshold] = 0.0

    eigvals_g, eigvecs_g = np.linalg.eigh(mass_g)
    GaugeRot = np.real_if_close(eigvecs_g.T, tol=1e8)
    mass_g_sq = eigvals_g.copy()
    mass_g_sq[np.abs(mass_g_sq) < scalar_threshold] = 0.0

    eigvals_q, eigvecs_q = np.linalg.eigh(mass_quark)
    QuarkRot = np.real_if_close(eigvecs_q.T, tol=1e8)
    mass_quark_sq = eigvals_q.real
    mass_quark_sq[np.abs(mass_quark_sq) < fermion_threshold] = 0.0

    eigvals_l, eigvecs_l = np.linalg.eigh(mass_lepton)
    LepRot = np.real_if_close(eigvecs_l.T, tol=1e8)
    mass_lepton_sq = eigvals_l.real
    mass_lepton_sq[np.abs(mass_lepton_sq) < fermion_threshold] = 0.0

    # ------------------------------------------------------------------
    # Physical couplings
    # ------------------------------------------------------------------
    LambdaGauge3 = np.einsum("abij,j->abi", gauge_curv, vev, optimize=True)
    LambdaHiggs3 = h3 + np.einsum("abij,j->abi", h4, vev, optimize=True)

    LambdaQuark3 = (
        np.einsum("ilk,lj->ijk", np.conjugate(quark_curv), MIJ_quark, optimize=True)
        + np.einsum("il,ljk->ijk", np.conjugate(MIJ_quark), quark_curv, optimize=True)
    )
    LambdaQuark4 = (
        np.einsum("ilk,ljm->ijkm", np.conjugate(quark_curv), quark_curv, optimize=True)
        + np.einsum("ilm,ljk->ijkm", np.conjugate(quark_curv), quark_curv, optimize=True)
    )

    LambdaLepton3 = (
        np.einsum("ilk,lj->ijk", np.conjugate(lepton_curv), MIJ_lepton, optimize=True)
        + np.einsum("il,ljk->ijk", np.conjugate(MIJ_lepton), lepton_curv, optimize=True)
    )
    LambdaLepton4 = (
        np.einsum("ilk,ljm->ijkm", np.conjugate(lepton_curv), lepton_curv, optimize=True)
        + np.einsum("ilm,ljk->ijkm", np.conjugate(lepton_curv), lepton_curv, optimize=True)
    )

    Couplings_Higgs_Triple = np.einsum(
        "pi,qj,rk,ijk->pqr", HiggsRot, HiggsRot, HiggsRot, LambdaHiggs3, optimize=True
    )
    Couplings_Higgs_Quartic = np.einsum(
        "pi,qj,rk,sl,ijkl->pqrs", HiggsRot, HiggsRot, HiggsRot, HiggsRot, h4, optimize=True
    )
    Couplings_Gauge_Higgs_21 = np.einsum(
        "pa,qb,ri,abi->pqr", GaugeRot, GaugeRot, HiggsRot, LambdaGauge3, optimize=True
    )
    Couplings_Gauge_Higgs_22 = np.einsum(
        "pa,qb,ri,sj,abij->pqrs", GaugeRot, GaugeRot, HiggsRot, HiggsRot, gauge_curv, optimize=True
    )
    Couplings_Quark_Higgs_21 = np.einsum(
        "pa,qb,ri,abi->pqr", QuarkRot, QuarkRot, HiggsRot, LambdaQuark3, optimize=True
    )
    Couplings_Quark_Higgs_22 = np.einsum(
        "pa,qb,ri,sj,abij->pqrs", QuarkRot, QuarkRot, HiggsRot, HiggsRot, LambdaQuark4, optimize=True
    )
    Couplings_Lepton_Higgs_21 = np.einsum(
        "pa,qb,ri,abi->pqr", LepRot, LepRot, HiggsRot, LambdaLepton3, optimize=True
    )
    Couplings_Lepton_Higgs_22 = np.einsum(
        "pa,qb,ri,sj,abij->pqrs", LepRot, LepRot, HiggsRot, HiggsRot, LambdaLepton4, optimize=True
    )

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------
    grad_mass_basis = np.zeros(vev.shape[0], dtype=float)
    cw_gauge = CW_CONSTANTS["gauge"]
    cw_scalar = CW_CONSTANTS["scalar"]
    cw_fermion = CW_CONSTANTS["fermion"]

    for i in range(vev.shape[0]):
        tmp = 0.0
        for a in range(mass_g_sq.shape[0]):
            m2 = mass_g_sq[a]
            if m2 != 0.0:
                coup = Couplings_Gauge_Higgs_21[a, a, i]
                tmp += 1.5 * float(coup) * m2 * _log_term(m2, scale_sq, cw_gauge)
        for a in range(mass_h_sq.shape[0]):
            m2 = mass_h_sq[a]
            if m2 != 0.0:
                coup = Couplings_Higgs_Triple[a, a, i]
                tmp += 0.5 * float(coup) * m2 * _log_term(m2, scale_sq, cw_scalar)
        for a in range(mass_quark_sq.shape[0]):
            m2 = mass_quark_sq[a]
            if m2 != 0.0:
                coup = Couplings_Quark_Higgs_21[a, a, i].real
                tmp -= 3.0 * coup * m2 * _log_term(m2, scale_sq, cw_fermion)
        for a in range(mass_lepton_sq.shape[0]):
            m2 = mass_lepton_sq[a]
            if m2 != 0.0:
                coup = Couplings_Lepton_Higgs_21[a, a, i].real
                tmp -= 1.0 * coup * m2 * _log_term(m2, scale_sq, cw_fermion)
        grad_mass_basis[i] = tmp

    grad_phys = HiggsRot.T @ grad_mass_basis
    grad_phys = np.real_if_close(grad_phys * EPS_LOOP, tol=1e8)

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------
    storage = np.zeros((vev.shape[0], vev.shape[0]), dtype=float)
    for i in range(vev.shape[0]):
        for j in range(vev.shape[0]):
            total = 0.0
            # Gauge sector
            block = 0.0
            for a in range(mass_g_sq.shape[0]):
                for b in range(mass_g_sq.shape[0]):
                    coup1 = Couplings_Gauge_Higgs_21[a, b, i]
                    coup2 = Couplings_Gauge_Higgs_21[b, a, j]
                    br = _fbase(mass_g_sq[a], mass_g_sq[b], scale) - cw_gauge + 0.5
                    block += float(coup1) * float(coup2) * br
                if mass_g_sq[a] != 0.0:
                    coup = Couplings_Gauge_Higgs_22[a, a, i, j]
                    block += float(coup) * mass_g_sq[a] * _log_term(mass_g_sq[a], scale_sq, cw_gauge)
            total += 1.5 * block

            # Scalar sector
            block = 0.0
            for a in range(mass_h_sq.shape[0]):
                for b in range(mass_h_sq.shape[0]):
                    coup1 = Couplings_Higgs_Triple[a, b, i]
                    coup2 = Couplings_Higgs_Triple[b, a, j]
                    br = _fbase(mass_h_sq[a], mass_h_sq[b], scale) - cw_scalar + 0.5
                    block += float(coup1) * float(coup2) * br
                if mass_h_sq[a] != 0.0:
                    coup = Couplings_Higgs_Quartic[a, a, i, j]
                    block += float(coup) * mass_h_sq[a] * _log_term(mass_h_sq[a], scale_sq, cw_scalar)
            total += 0.5 * block

            # Quark sector
            block = 0.0
            for a in range(mass_quark_sq.shape[0]):
                for b in range(mass_quark_sq.shape[0]):
                    coup = (
                        Couplings_Quark_Higgs_21[a, b, i]
                        * Couplings_Quark_Higgs_21[b, a, j]
                    ).real
                    br = _fbase(mass_quark_sq[a], mass_quark_sq[b], scale) - cw_fermion + 0.5
                    block += coup * br
                if mass_quark_sq[a] != 0.0:
                    coup = Couplings_Quark_Higgs_22[a, a, i, j].real
                    block += coup * mass_quark_sq[a] * _log_term(mass_quark_sq[a], scale_sq, cw_fermion)
            total -= 3.0 * block

            # Lepton sector
            block = 0.0
            for a in range(mass_lepton_sq.shape[0]):
                for b in range(mass_lepton_sq.shape[0]):
                    coup = (
                        Couplings_Lepton_Higgs_21[a, b, i]
                        * Couplings_Lepton_Higgs_21[b, a, j]
                    ).real
                    br = _fbase(mass_lepton_sq[a], mass_lepton_sq[b], scale) - cw_fermion + 0.5
                    block += coup * br
                if mass_lepton_sq[a] != 0.0:
                    coup = Couplings_Lepton_Higgs_22[a, a, i, j].real
                    block += coup * mass_lepton_sq[a] * _log_term(mass_lepton_sq[a], scale_sq, cw_fermion)
            total -= block

            storage[i, j] = total

    storage = ensure_symmetric(storage)
    hess_phys = HiggsRot.T @ storage @ HiggsRot
    hess_phys = np.real_if_close(hess_phys * EPS_LOOP, tol=1e8)

    return grad_phys, ensure_symmetric(hess_phys)
