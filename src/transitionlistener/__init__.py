"""TransitionListener package providing tools to analyze cosmological phase transitions.

Part of TransitionListener v2
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("TransitionListener")  # Resolve version when installed
except PackageNotFoundError:
    # Fallback for in-tree usage before installation. `tests/test_the_version_is_one_number.py`
    # keeps it equal to the version in pyproject.toml, which is what an installed copy reports.
    __version__ = "2.2.0"

from rich.console import Console
console = Console(highlight=False, highlighter=None)
print = console.print
