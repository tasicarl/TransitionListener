"""
Coleman–Weinberg counterterm solver for the CP-conserving 2HDM.

This module evaluates the analytic curvature tensors at the vacuum and matches
the BSMPT counterterm prescriptions.  It is invoked directly from the 2HDM
model so that the model file itself remains agnostic of the 8-dimensional
tensor algebra.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
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
    Assemble the numbers the tensor evaluator reads.

    Values are expressed in physical units (GeV or dimensionless) because the
    generated expressions follow the BSMPT normalisation.
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


def _counterterm_matrix(v1: float, v2: float) -> np.ndarray:
    r"""The linear counterterm system matrix :math:`M(v_1, v_2)`.

    Following Eqs. (3.59)–(3.63) of arXiv:1803.02846, the two tadpole conditions and five mass-matrix conditions
    (CP-even, CP-odd, and charged sectors) of the counterterm potential

    .. math::
       V_{ct} = \delta m_{11}^2 \Phi_1^\dagger\Phi_1 + \delta m_{22}^2 \Phi_2^\dagger\Phi_2
       - \delta m_{12}^2 (\Phi_1^\dagger\Phi_2 + h.c.) + \tfrac{\delta\lambda_1}{2} (\Phi_1^\dagger\Phi_1)^2
       + \tfrac{\delta\lambda_2}{2} (\Phi_2^\dagger\Phi_2)^2
       + \delta\lambda_3 \Phi_1^\dagger\Phi_1 \Phi_2^\dagger\Phi_2
       + \tfrac{\delta\lambda_5}{2} ((\Phi_1^\dagger\Phi_2)^2 + h.c.)

    are the rows of

    .. math::
       M(v_1, v_2)\,\delta = -\Delta,

    where :math:`\delta = (\delta m_{11}^2, \delta m_{22}^2, \delta m_{12}^2,
    \delta\lambda_1, \delta\lambda_2, \delta\lambda_3, \delta\lambda_5)` and the rows are in the
    order of ``CONDITION_SPECS``.
    """
    v1_sq, v2_sq, product = v1 * v1, v2 * v2, v1 * v2
    return np.array([
        [v1, 0.0, -v2, 0.5 * v1 * v1_sq, 0.0, 0.5 * v1 * v2_sq, 0.5 * v1 * v2_sq],
        [0.0, v2, -v1, 0.0, 0.5 * v2 * v2_sq, 0.5 * v1_sq * v2, 0.5 * v1_sq * v2],
        [1.0, 0.0, 0.0, 1.5 * v1_sq, 0.0, 0.5 * v2_sq, 0.5 * v2_sq],
        [0.0, 1.0, 0.0, 0.0, 1.5 * v2_sq, 0.5 * v1_sq, 0.5 * v1_sq],
        [0.0, 0.0, -1.0, 0.0, 0.0, product, product],
        [1.0, 0.0, 0.0, 0.5 * v1_sq, 0.0, 0.5 * v2_sq, -0.5 * v2_sq],
        [1.0, 0.0, 0.0, 0.5 * v1_sq, 0.0, 0.5 * v2_sq, 0.0],
    ])


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
        metadata,
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
