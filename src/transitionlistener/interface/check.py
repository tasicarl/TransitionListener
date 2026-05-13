"""Entry point for ``tl --check``.

All check logic lives in ``tests/check_installation.py`` in the repository.
For pip installs, hatchling bundles that file as
``transitionlistener/check_installation.py`` so it is always reachable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_and_run(path: Path) -> int:
    """Import the installation-check script at ``path`` and execute it."""
    spec = importlib.util.spec_from_file_location("_tl_check_installation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.run_checks()


def run_checks() -> int:
    """Locate and run the packaged installation checks.

    The helper first prefers the repository-local ``tests/check_installation.py``
    file and falls back to the copy bundled into wheel installs.
    """
    # Option 1: repo clone — tests/check_installation.py is at parents[3]/tests/
    # (this file: src/transitionlistener/interface/check.py → parents[3] = repo root)
    repo_path = Path(__file__).resolve().parents[3] / "tests" / "check_installation.py"
    if repo_path.exists():
        return _load_and_run(repo_path)

    # Option 2: pip install — hatchling bundled it as transitionlistener/check_installation.py
    try:
        import importlib.resources as ir
        import contextlib
        ref = ir.files("transitionlistener") / "check_installation.py"
        with contextlib.ExitStack() as stack:
            bundled = stack.enter_context(ir.as_file(ref))
            return _load_and_run(bundled)
    except Exception:
        pass

    from transitionlistener import console
    console.print(
        "[bold red]ERROR[/bold red]  check_installation.py not found.\n"
        "  Repo clone: expected at tests/check_installation.py\n"
        "  pip install: rebuild the wheel with hatchling."
    )
    return 1
