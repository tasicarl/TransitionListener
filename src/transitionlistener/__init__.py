"""TransitionListener package providing tools to analyze cosmological phase transitions.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("TransitionListener")  # Resolve version when installed
except PackageNotFoundError:
    # Fallback for in-tree usage before installation
    __version__ = "1.5"

from rich.console import Console
console = Console(highlight=False, highlighter=None)
print = console.print
