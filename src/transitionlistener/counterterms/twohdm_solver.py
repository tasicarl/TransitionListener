"""
Coleman–Weinberg counterterm solver for the CP-conserving 2HDM.

This module evaluates the analytic curvature tensors at the vacuum and matches
the BSMPT counterterm prescriptions.  It is invoked directly from the 2HDM
model so that the model file itself remains agnostic of the 8-dimensional
tensor algebra.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import numpy as np
import sympy as sp
from rich.progress import Progress, SpinnerColumn, TextColumn

from transitionlistener import errors
from transitionlistener.ckm import wolfenstein_ckm
from transitionlistener.constants import SM_FERMION_MASSES_GEV
from transitionlistener.generic_potential import console as tl_console
from . import CounterTerms, generate_counterterms, tensors as tensor_utils

CKM_DEFAULT = wolfenstein_ckm()
GENERATED_TENSOR_FILE = (
    Path(__file__).resolve().parents[3]
    / "models"
    / "generated"
    / "2hdm_curvature_tensors.json"
)

CONDITION_SPECS = (
    ("grad", 4),
    ("grad", 6),
    ("hess", (4, 4)),
    ("hess", (6, 6)),
    ("hess", (4, 6)),
    ("hess", (5, 5)),
    ("hess", (0, 0)),
)

PARAMETER_SYMBOLS = ("dm11", "dm22", "dm12", "dl1", "dl2", "dl3", "dl5")


def _load_dataset() -> Tuple[dict, dict]:
    """
    Load (or build) the cached curvature tensor dataset for the 2HDM.

    The JSON file is generated on demand using the SymPy-based generator when it
    is missing, ensuring that production runs never fail due to a stale cache.
    """

    if GENERATED_TENSOR_FILE.exists():
        tl_console.print(
            "[green]Loaded cached 2HDM curvature tensors.[/green]"
        )
        return tensor_utils.load_tensor_dataset(GENERATED_TENSOR_FILE)

    with Progress(
        SpinnerColumn("bouncingBall"),
        TextColumn("Generating 2HDM curvature tensors…"),
        transient=True,
        console=tl_console,
    ) as progress:
        task = progress.add_task("generate", total=None)
        try:
            generate_counterterms("models.TL_2HDM")
        finally:
            progress.update(task, advance=1.0)

    try:
        dataset = tensor_utils.load_tensor_dataset(GENERATED_TENSOR_FILE)
    except errors.InitPotentialError as retry_exc:
        raise errors.InitPotentialError(
            "Failed to load generated curvature tensors for the 2HDM after "
            "regeneration. Inspect SymPy generation logs for details."
        ) from retry_exc

    tl_console.print("[green]Curvature tensors generated successfully.[/green]")
    return dataset


def _tensor_parameter_values(model) -> dict[str, complex]:
    """
    Assemble the parameter substitution dictionary used by the tensor evaluator.

    Values are expressed in physical units (GeV or dimensionless) because the
    generated SymPy expressions follow the BSMPT normalisation.
    """

    cf_sq = model.conversionFactor**2
    values: dict[str, complex] = {
        "lambda1": float(model.lambda1),
        "lambda2": float(model.lambda2),
        "lambda3": float(model.lambda3),
        "lambda4": float(model.lambda4),
        "lambda5": float(model.lambda5),
        "m11sq": float(model.m11_sq * cf_sq),
        "m22sq": float(model.m22_sq * cf_sq),
        "m12sq": float(model.m12_sq * cf_sq),
        "tan_beta": float(model.tan_beta),
        "v1": float(model.v1_GeV),
        "v2": float(model.v2_GeV),
        "yukawa_type": float(model.yukawa_type),
        "Cg": float(model.g2),
        "Cgs": float(model.g1),
    }
    values.update({key: float(mass) for key, mass in SM_FERMION_MASSES_GEV.items()})
    values.update({key: complex(val) for key, val in CKM_DEFAULT.items()})
    return values


@lru_cache(maxsize=1)
def _counterterm_matrix_factory():
    """
    Build the SymPy representation of the on-shell renormalisation conditions.

    Following Eqs. (3.59)–(3.63) of arXiv:1803.02846, we impose two tadpole
    conditions and five mass-matrix conditions (CP-even, CP-odd, and charged
    sectors).  The result is a 7×7 matrix ``M(v_1, v_2)`` such that
    ``M · δ = -Δ`` where ``δ`` collects the counterterms and ``Δ`` the Coleman–
    Weinberg corrections.
    """

    dm11, dm22, dm12, dl1, dl2, dl3, dl5 = sp.symbols(
        "dm11 dm22 dm12 dl1 dl2 dl3 dl5",
        real=True,
    )
    delta_params = (dm11, dm22, dm12, dl1, dl2, dl3, dl5)

    (
        phi1p_R,
        phi1p_I,
        phi2p_R,
        phi2p_I,
        phi1_R,
        phi1_I,
        phi2_R,
        phi2_I,
    ) = sp.symbols(
        "phi1p_R phi1p_I phi2p_R phi2p_I phi1_R phi1_I phi2_R phi2_I",
        real=True,
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

    Vct = (
        dm11 * phi1_sq
        + dm22 * phi2_sq
        - dm12 * (phi12 + phi21)
        + (dl1 / 2) * phi1_sq**2
        + (dl2 / 2) * phi2_sq**2
        + dl3 * phi1_sq * phi2_sq
        + (dl5 / 2) * (phi12**2 + phi21**2)
    )

    grad_exprs = [sp.diff(Vct, field) for field in fields]
    hess_exprs = [
        [sp.diff(grad_exprs[i], field_j) for field_j in fields]
        for i in range(len(fields))
    ]

    v1_sym, v2_sym = sp.symbols("v1 v2", real=True)
    vacuum_subs = {
        phi1p_R: 0,
        phi1p_I: 0,
        phi2p_R: 0,
        phi2p_I: 0,
        phi1_I: 0,
        phi2_I: 0,
        phi1_R: v1_sym,
        phi2_R: v2_sym,
    }

    condition_rows = []
    for kind, index in CONDITION_SPECS:
        if kind == "grad":
            expr = grad_exprs[index]
        else:
            i, j = index
            expr = hess_exprs[i][j]
        expr = sp.simplify(expr.subs(vacuum_subs))
        row = [sp.simplify(sp.diff(expr, param)) for param in delta_params]
        condition_rows.append(row)

    matrix = sp.Matrix(condition_rows)
    return sp.lambdify((v1_sym, v2_sym), matrix, "numpy")


def _counterterm_matrix(v1: float, v2: float) -> np.ndarray:
    r"""Evaluate the linear counterterm system matrix :math:`M(v_1, v_2)`.

    The matrix is defined through the on-shell matching equations

    .. math::
       M(v_1, v_2)\,\delta = -\Delta,

    where :math:`\delta = (\delta m_{11}^2, \delta m_{22}^2, \delta m_{12}^2,
    \delta\lambda_1, \delta\lambda_2, \delta\lambda_3, \delta\lambda_5)`.
    """
    matrix_func = _counterterm_matrix_factory()
    return np.asarray(matrix_func(v1, v2), dtype=float)


def _condition_vector(grad_phys: np.ndarray, hess_phys: np.ndarray) -> np.ndarray:
    """Collect the tadpole and Hessian conditions into the solver right-hand side."""
    values = []
    for kind, index in CONDITION_SPECS:
        if kind == "grad":
            values.append(-grad_phys[index])
        else:
            i, j = index
            values.append(-hess_phys[i, j])
    return np.asarray(values, dtype=float)


def compute_counterterms(
    model,
) -> Tuple[CounterTerms, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the Coleman–Weinberg gradient/Hessian and solve for the 2HDM counterterms.

    Parameters
    ----------
    model:
        Initialised 2HDM model instance providing vevs, couplings, and conversion
        factors.

    Returns
    -------
    counterterms, gradient, hessian, residual:
        Counterterm container together with the Coleman–Weinberg derivatives in
        the internal TransitionListener unit system, and the residual of the
        renormalisation conditions (also expressed in internal units).
    """

    metadata, tensors = _load_dataset()
    arrays = tensor_utils.evaluate_curvature_arrays(
        tensors,
        copy.deepcopy(metadata),
        _tensor_parameter_values(model),
        yukawa_key=model._YUKAWA_TYPE_MAP[int(model.yukawa_type)],
    )

    vev = np.zeros(8, dtype=float)
    vev[4] = float(model.v1_GeV)
    vev[6] = float(model.v2_GeV)

    grad_phys, hess_phys = model.coleman_weinberg_from_curvatures(
        arrays,
        vev,
        scale=float(model.v_GeV),
    )
    grad_phys = np.real_if_close(grad_phys, tol=1e8)
    hess_phys = np.real_if_close(hess_phys, tol=1e8)
    if np.iscomplexobj(grad_phys):
        if not np.allclose(grad_phys.imag, 0.0, atol=1e-12):
            raise ValueError(
                "Coleman–Weinberg gradient carries sizeable imaginary parts."
            )
        grad_phys = grad_phys.real
    if np.iscomplexobj(hess_phys):
        if not np.allclose(hess_phys.imag, 0.0, atol=1e-12):
            raise ValueError(
                "Coleman–Weinberg Hessian carries sizeable imaginary parts."
            )
        hess_phys = hess_phys.real
    hess_phys = tensor_utils.ensure_symmetric(np.asarray(hess_phys, dtype=float))

    v1 = float(model.v1_GeV)
    v2 = float(model.v2_GeV)
    A = _counterterm_matrix(v1, v2)
    b = _condition_vector(grad_phys, hess_phys)
    try:
        (
            dm11_sq_GeV2,
            dm22_sq_GeV2,
            dm12_sq_GeV2,
            dlambda1,
            dlambda2,
            dlambda3,
            dlambda5,
        ) = np.linalg.solve(A, b)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Failed to solve the counterterm system; vacuum configuration is "
            "singular."
        ) from exc
    dlambda4 = 0.0

    cf = model.conversionFactor
    grad_internal = np.asarray(
        np.where(np.abs(grad_phys) < 1e-10, 0.0, grad_phys * cf),
        dtype=float,
    )
    hess_internal = np.asarray(
        np.where(
            np.abs(hess_phys) < 1e-10,
            0.0,
            hess_phys * (cf * cf),
        ),
        dtype=float,
    )
    hess_internal = tensor_utils.ensure_symmetric(hess_internal)

    dm11_sq_internal = dm11_sq_GeV2 * (cf * cf)
    dm22_sq_internal = dm22_sq_GeV2 * (cf * cf)
    dm12_sq_internal = dm12_sq_GeV2 * (cf * cf)

    counterterm_values = {
        "dlambda1": float(dlambda1),
        "dlambda2": float(dlambda2),
        "dlambda3": float(dlambda3),
        "dlambda4": float(dlambda4),
        "dlambda5": float(dlambda5),
        "dm11_sq": float(dm11_sq_internal),
        "dm22_sq": float(dm22_sq_internal),
        "dm12_sq": float(dm12_sq_internal),
    }
    counterterms = CounterTerms.from_dict(counterterm_values)

    counterterm_vector = np.array(
        [
            dm11_sq_GeV2,
            dm22_sq_GeV2,
            dm12_sq_GeV2,
            dlambda1,
            dlambda2,
            dlambda3,
            dlambda5,
        ],
        dtype=float,
    )
    residual_phys = A @ counterterm_vector - b
    residual_internal = residual_phys.copy()
    residual_internal[:2] *= cf
    residual_internal[2:] *= cf * cf

    return counterterms, grad_internal, hess_internal, residual_internal
