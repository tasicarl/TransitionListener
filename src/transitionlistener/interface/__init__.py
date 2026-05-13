"""Interface entry points for TransitionListener scans.

This module exposes the main functions and classes for performing
scans and single-point evaluations using TransitionListener. It also
manages global state variables such as debug mode, timeout settings,
and logging configurations.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

import logging
logging.getLogger("numexpr").setLevel(logging.WARNING)

from . import state as _state
from .logging_utils import Logger
from .config_loader import ScanConfig
from .pipeline import (
    run_TL, _compute_single_point,
    _build_result_from_context,
    _handle_single_point_error)
from .samplers import (
    call_by_sampler,
    sample_and_export_to_csv,
    scanfunc,
    extract_parameters,
    log_likelihood,
    prior_transform,
    ultranest,
)
from .scans import (
    create_grid,
    line_scan,
    grid_scan,
    table,
    random_scan,
    save_results_1d,
    save_results_2d,
    plot_results_1d,
    plot_results_2d,
)
from .single_point import single
from .cli import main

__all__ = [
    "Logger",
    "ScanConfig",
    "run_TL",
    "_compute_single_point",
    "_build_result_from_context",
    "_handle_single_point_error",
    "call_by_sampler",
    "sample_and_export_to_csv",
    "scanfunc",
    "extract_parameters",
    "log_likelihood",
    "prior_transform",
    "ultranest",
    "create_grid",
    "line_scan",
    "grid_scan",
    "table",
    "random_scan",
    "save_results_1d",
    "save_results_2d",
    "plot_results_1d",
    "plot_results_2d",
    "single",
    "main",
    "DEBUGMODE",
    "TIMEOUT",
    "model",
    "errorlogger",
    "resultlogger",
    "config",
]


def __getattr__(name):
    """Proxy selected legacy module-level state to :mod:`transitionlistener.interface.state`."""
    if name in {"DEBUGMODE", "TIMEOUT", "model", "errorlogger", "resultlogger", "config"}:
        return getattr(_state, name)
    raise AttributeError(f"module 'transitionlistener.interface' has no attribute '{name}'")


def __setattr__(name, value):
    """Mirror assignments to the shared interface state when applicable."""
    if name in {"DEBUGMODE", "TIMEOUT", "model", "errorlogger", "resultlogger", "config"}:
        setattr(_state, name, value)
    else:
        globals()[name] = value
