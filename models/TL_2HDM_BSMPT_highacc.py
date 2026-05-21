"""2HDM high-accuracy wrapper with tuned path-deformation convergence controls.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

_BASE_MODEL_PATH = Path(__file__).with_name("TL_2HDM_BSMPT.py")
_SPEC = importlib.util.spec_from_file_location("tl_2hdm_base_model_nact100_conv", _BASE_MODEL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load base model from {_BASE_MODEL_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
R2HDM = _MODULE.R2HDM


class R2HDM_HighAcc_NAct100_Conv1_FRatio005(R2HDM):
    """2HDM model with n_action=100 and adjusted deformation convergence settings."""

    def setConfigParameters(self) -> None:
        super().setConfigParameters()
        self.config.percolationConf.n_action = 100

        tunneling = copy.deepcopy(self.config.tracingConf.tunneling_params)
        deform = dict(tunneling.get("deformation_deform_params", {}))
        deform["converge_0"] = 1.0
        deform["fRatioConv"] = 5e-3
        tunneling["deformation_deform_params"] = deform
        self.config.tracingConf.tunneling_params = tunneling
