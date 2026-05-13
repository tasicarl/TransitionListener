"""Small helpers for logging scan output to disk.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

from pathlib import Path


class Logger:
    """Very small helper that mirrors stdout logging to a file."""

    def __init__(self, file_name: str | Path):
        """Open `file_name` in append mode so logs persist across scans."""
        self.file = open(file_name, "a")

    def log(self, message: str):
        """Write `message` to the backing file and flush immediately."""
        self.file.write(message)
        self.file.flush()

    def close(self):
        """Close the on-disk log file."""
        self.file.close()
