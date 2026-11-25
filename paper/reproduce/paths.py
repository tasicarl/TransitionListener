"""Locations shared by the scripts of paper/reproduce.

    PAPER   paper/
    DATA    paper/data: the results that stage A (figs/make_figures.py) reads and stage B writes
    WORK    intermediate results of stage B that stage A does not read (action caches of every coupling, line-scan
            points, TransitionListener nodes, partial tables); paper/reproduce/work unless PAPER_WORK is set
    REPO    the repository root, with src/ (TransitionListener) and models/

Importing this module puts src/ (or $TL_CODE_ROOT/src), models/ and every folder of paper/reproduce on sys.path, so
that the scripts of one folder can import those of another."""
import os
import sys

REPRODUCE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(REPRODUCE)
REPO = os.path.dirname(PAPER)
DATA = os.path.join(PAPER, "data")
WORK = os.path.abspath(os.environ.get("PAPER_WORK", os.path.join(REPRODUCE, "work")))
CODE = os.path.abspath(os.environ.get("TL_CODE_ROOT", REPO))
SRC = os.path.join(CODE, "src")
MODELS = os.path.join(CODE, "models")

#: bounce actions per coupling and kind (rate/compute_actions.py); stage A reads those of g = 0.5 in DATA
ACTION_CACHE = os.path.abspath(os.environ.get("ACTION_CACHE_DIR", os.path.join(DATA, "action_cache")))
#: TransitionListener's line scan, one folder per coupling (linescan/scan_point.py)
LINESCAN_POINTS = os.path.abspath(os.environ.get("LINESCAN_POINTS", os.path.join(DATA, "linescan_points")))
#: actions of the four-dimensional potential with the daisy term (daisy/compute_daisy_actions.py)
DAISY_CACHE = os.path.abspath(os.environ.get("DAISY_CACHE", os.path.join(DATA, "daisy_cache")))

for _p in [SRC, MODELS] + [os.path.join(REPRODUCE, d) for d in ("rate", "daisy", "plane", "binaries", "linescan")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def data(*parts):
    """A path under paper/data."""
    return os.path.join(DATA, *parts)


def work(*parts):
    """A path under the work folder of stage B (created on first use)."""
    os.makedirs(WORK, exist_ok=True)
    return os.path.join(WORK, *parts)
