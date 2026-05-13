r"""
Cabibbo–Kobayashi–Maskawa matrix helpers.

The functions defined here provide ready-to-use parametrisations of the CKM
matrix that can be shared between models.  The default implementation follows
the Wolfenstein expansion,

.. math::
   V_{\mathrm{CKM}} \approx
   \begin{pmatrix}
     1 - \tfrac{1}{2}\lambda^2 & \lambda &
     A \lambda^3 (\rho - i \eta) \\
     -\lambda & 1 - \tfrac{1}{2}\lambda^2 & A \lambda^2 \\
     A \lambda^3 \left(1 - \rho - i \eta\right) &
     -A \lambda^2 & 1
   \end{pmatrix},

with higher-order terms resummed to guarantee unitarity.  The phases are
returned as complex numbers so that downstream code can keep track of CP
violating effects without additional bookkeeping.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import cmath
import math
from typing import Dict

CKMElementMap = Dict[str, complex]


def wolfenstein_ckm(
    lambda_w: float = 0.22537,
    A: float = 0.814,
    rho: float = 0.117,
    eta: float = 0.353,
) -> CKMElementMap:
    r"""
    Return the CKM matrix elements in the Wolfenstein parametrisation.

    Parameters
    ----------
    lambda_w:
        Cabibbo angle expansion parameter :math:`\\lambda`.
    A:
        Overall strength of the 2–3 mixing.
    rho:
        Real part of the complex phase.
    eta:
        Imaginary part of the complex phase.

    Returns
    -------
    dict[str, complex]
        Mapping ``Vxy -> V_{xy}`` covering all nine CKM entries.
    """

    theta12 = math.asin(lambda_w)
    theta23 = math.asin(A * lambda_w**2)
    complex_phase = A * (lambda_w**3) * (rho + 1j * eta)
    delta = cmath.phase(complex_phase)
    theta13 = math.asin(abs(complex_phase))

    Vud = math.cos(theta12) * math.cos(theta13)
    Vus = math.sin(theta12) * math.cos(theta13)
    Vub = math.sin(theta13) * cmath.exp(-1j * delta)
    Vcd = -math.sin(theta12) * math.cos(theta23) - math.cos(theta12) * math.sin(theta23) * math.sin(theta13) * cmath.exp(1j * delta)
    Vcs = math.cos(theta12) * math.cos(theta23) - math.sin(theta12) * math.sin(theta23) * math.sin(theta13) * cmath.exp(1j * delta)
    Vcb = math.sin(theta23) * math.cos(theta13)
    Vtd = math.sin(theta12) * math.sin(theta23) - math.cos(theta12) * math.cos(theta23) * math.sin(theta13) * cmath.exp(1j * delta)
    Vts = -math.cos(theta12) * math.sin(theta23) - math.sin(theta12) * math.cos(theta23) * math.sin(theta13) * cmath.exp(1j * delta)
    Vtb = math.cos(theta23) * math.cos(theta13)

    return {
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
