"""Shared global state and runtime configuration for the TL interface.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations
import signal
import warnings
import numpy as np
from transitionlistener.errors import Timeout

DEBUGMODE: bool = False
TIMEOUT: int = 300
model = None
potential = None
errorlogger = None
resultlogger = None
config = None

np.random.seed(19971104)

_SUPPRESSED_WARNINGS = [
    "invalid value encountered in subtract",
    "overflow encountered in add",
    "invalid value encountered in multiply",
    "overflow encountered in multiply",
    "divide by zero encountered in log",
    "invalid value encountered in scalar subtract",
    "overflow encountered in scalar multiply",
    "divide by zero encountered in scalar divide",
    "invalid value encountered in scalar divide",
    "invalid value encountered in scalar multiply",
    "invalid value encountered in divide",
    "divide by zero encountered in divide",
    "invalid value encountered in scalar divide",
]


def set_debugmode(enabled: bool) -> None:
    """Toggle debug mode and update warning filters accordingly."""
    global DEBUGMODE
    DEBUGMODE = bool(enabled)
    warnings.resetwarnings()
    if not DEBUGMODE:
        for warning in _SUPPRESSED_WARNINGS:
            warnings.filterwarnings("ignore", message=f".*{warning}.*", category=RuntimeWarning)


set_debugmode(DEBUGMODE)

def timeout_handler(signum, frame):  # pragma: no cover - deterministic signal hook
    """Raise a :class:`Timeout` when a POSIX alarm fires."""
    msg = f"The calculation timed out after {TIMEOUT} seconds."
    raise Timeout(msg)

signal.signal(signal.SIGALRM, timeout_handler)
