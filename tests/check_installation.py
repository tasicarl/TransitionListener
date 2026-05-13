"""Installation self-test for TransitionListener.

Called by ``tl --check`` (via src/transitionlistener/interface/check.py).
Can also be run directly: ``python tests/check_installation.py``.

The quick checks (~300 ms) use only code that ships inside the package.
They are followed automatically by an end-to-end physics smoke test
(~1-2 min) that runs ``example_point.yaml`` and validates key physical
outputs.
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _result(label: str, ok: bool, detail: str = "") -> None:
    from transitionlistener import console
    status = "[bold green]PASS[/bold green]" if ok else "[bold red]FAIL[/bold red]"
    line = f"  {status}  {label}"
    if detail:
        line += f"  ({detail})"
    console.print(line)


def _find_examples_root() -> Path | None:
    """Return the directory that contains both examples/ and models/.

    Works for repo clones (this file lives at tests/check_installation.py,
    so the repo root is one level up) and for pip installs where hatchling
    has bundled examples/ as transitionlistener/examples/.
    """
    # Repo clone: tests/check_installation.py → repo root is parents[1]
    candidate = Path(__file__).resolve().parents[1]
    if (candidate / "examples" / "example_point.yaml").exists():
        return candidate
    # pip install: hatchling bundled examples/ as transitionlistener/examples/
    try:
        import importlib.resources as ir
        pkg_root = Path(str(ir.files("transitionlistener")))
        if (pkg_root / "examples" / "example_point.yaml").exists():
            return pkg_root
    except Exception:
        pass
    return None


def _run_smoke(console, failures: int) -> int:
    """Run the end-to-end physics smoke test. Returns updated failure count."""
    from rich.panel import Panel

    console.print()
    console.print(Panel.fit("[bold]End-to-end physics check[/bold]", border_style="cyan"))
    console.print()

    repo_root = _find_examples_root()
    if repo_root is None:
        console.print("  [bold yellow]SKIP[/bold yellow]  smoke test  (examples/ directory not found)")
        return failures

    yaml_src = repo_root / "examples" / "example_point.yaml"
    model_src = repo_root / "models" / "TL_conformal_dark_u1.py"

    with tempfile.TemporaryDirectory() as _tmp:
        tmpdir = Path(_tmp)
        (tmpdir / "examples").mkdir()
        (tmpdir / "models").mkdir()
        shutil.copy(yaml_src, tmpdir / "examples" / "example_point.yaml")
        shutil.copy(model_src, tmpdir / "models" / "TL_conformal_dark_u1.py")

        env = os.environ.copy()
        env.update({"OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})

        console.print("  Running example_point.yaml  [dim](conformal U(1), SinglePoint — takes ~1-2 min)[/dim]")
        t0 = time.perf_counter()
        # Prefer the `tl` script from the same env as sys.executable.
        tl_script = Path(sys.executable).parent / "tl"
        if not tl_script.exists():
            tl_script = Path(sys.executable).parent / "tl.exe"  # Windows
        cmd = (
            [str(tl_script), "-c", "examples/example_point.yaml", "-j", "1"]
            if tl_script.exists()
            else [sys.executable, "-c",
                  "from transitionlistener.interface import main; main()",
                  "-c", "examples/example_point.yaml", "-j", "1"]
        )
        try:
            proc = subprocess.run(
                cmd,
                cwd=tmpdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            _result("smoke test", False, "timed out after 300 s")
            return failures + 1
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            _result("smoke test", False, f"exit code {proc.returncode}")
            console.print(f"  [dim]{proc.stderr[-500:]}[/dim]")
            return failures + 1

        out_file = tmpdir / "scans" / "example_point" / "1_All_params.txt"
        if not out_file.exists():
            _result("smoke test", False, "output file not created")
            return failures + 1

        # Parse key-value output table
        values: dict[str, float] = {}
        for line in out_file.read_text().splitlines():
            m = re.match(r"^\s*(\S+)\s+(\S.*?)\s*$", line)
            if m:
                try:
                    values[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass

        EXPECTED = {
            "Tperc_SM_GeV":    (1.0e-3, 5.0e-2),
            "alpha":           (1.0e+1, 1.0e+4),
            "RH":              (1.0e-6, 1.0e+0),
        }

        smoke_ok = True
        for key, (lo, hi) in EXPECTED.items():
            if key not in values:
                _result(f"  {key}", False, "missing from output")
                smoke_ok = False
                failures += 1
            elif not math.isfinite(values[key]) or not (lo <= values[key] <= hi):
                _result(f"  {key}", False, f"{values[key]:.3e} outside [{lo:.0e}, {hi:.0e}]")
                smoke_ok = False
                failures += 1
            else:
                _result(f"  {key}", True, f"{values[key]:.3e}")

        _result("smoke test (example_point.yaml)", smoke_ok, f"{elapsed:.0f} s")
        if not smoke_ok:
            failures += 1

    return failures


def run_checks() -> int:
    """Run all installation checks. Returns 0 on success, 1 on failure."""
    from transitionlistener import console
    from rich.panel import Panel

    console.print(Panel.fit("[bold]TransitionListener installation check[/bold]", border_style="cyan"))
    console.print()

    failures = 0

    # ------------------------------------------------------------------
    # 1. Core scientific imports
    # ------------------------------------------------------------------
    for name in ("numpy", "scipy", "matplotlib", "pandas", "rich", "tqdm", "yaml"):
        t0 = time.perf_counter()
        try:
            __import__(name)
            _result(f"import {name}", True, f"{(time.perf_counter()-t0)*1e3:.0f} ms")
        except ImportError as exc:
            _result(f"import {name}", False, str(exc))
            failures += 1

    # ------------------------------------------------------------------
    # 2. Thermodynamics (reads tab_data shipped with the package)
    # ------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        from transitionlistener.thermodynamics import e_geffSM, p_geffSM, s_geffSM
        import numpy as np
        T, CF = 10.0, 1.0
        e = (math.pi**2 / 30) * e_geffSM(T, CF) * T**4
        p = (math.pi**2 / 90) * p_geffSM(T, CF) * T**4
        s = (2 * math.pi**2 / 45) * s_geffSM(T, CF) * T**3
        residual = abs(s - (e + p) / T)
        ok = residual < 1e-4
        _result("thermodynamics (e+p=sT)", ok, f"residual={residual:.2e}, {(time.perf_counter()-t0)*1e3:.0f} ms")
        if not ok:
            failures += 1
    except Exception as exc:
        _result("thermodynamics", False, str(exc))
        failures += 1

    # ------------------------------------------------------------------
    # 3. Percolation integral (ODE solver, no model file needed)
    # ------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        import numpy as np
        from transitionlistener.bubbledynamics import percIntegralODE, percIntegral
        T = np.geomspace(50.0, 0.5, 80)
        s_over_t = 120.0 * (T / T[0]) ** 0.5 + 5.0
        S = s_over_t * T
        H = T**2 / 1e6
        I_ode = percIntegralODE(T, H, S, vw=1.0)
        ok = I_ode.shape == T.shape and float(I_ode[-1]) > 0.0
        _result("percolation ODE integral", ok, f"{(time.perf_counter()-t0)*1e3:.0f} ms")
        if not ok:
            failures += 1
    except Exception as exc:
        _result("percolation ODE integral", False, str(exc))
        failures += 1

    # ------------------------------------------------------------------
    # 4. Bounce action (asymmetric double-well: V = (x^2-1)^2/4 - eps*x)
    # Minima at x≈±1; tiny eps=0.01 makes x≈+1 the true vacuum.
    # Barrier at x≈0 sits well above both minima.
    # ------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        import numpy as np
        from transitionlistener import pathDeformation

        def V(x):
            phi = np.asarray(x, dtype=float).ravel()
            vals = 0.25 * (phi**2 - 1.0)**2 - 0.01 * phi
            return float(vals[0]) if vals.size == 1 else vals

        def dV(x):
            x = np.asarray(x, dtype=float)
            phi = x.ravel()
            return (phi**3 - phi - 0.01).reshape(x.shape)

        x_true = np.array([1.0])
        x_false = np.array([-1.0])
        tobj = pathDeformation.fullTunneling(
            [x_true, x_false], V, dV,
            callback_data=1.0,
        )
        finite = math.isfinite(float(tobj.action))
        _result("bounce action (synthetic 1D)", finite, f"S3={tobj.action:.4f}, {(time.perf_counter()-t0)*1e3:.0f} ms")
        if not finite:
            failures += 1
    except Exception as exc:
        _result("bounce action (synthetic 1D)", False, str(exc))
        failures += 1

    # ------------------------------------------------------------------
    # 5. Optional: MPI (informational — never fails)
    # ------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        from mpi4py import MPI  # noqa: F401
        size = MPI.COMM_WORLD.Get_size()
        console.print(f"  [bold cyan]INFO[/bold cyan]  mpi4py  [green]found[/green]  (MPI size={size}, {(time.perf_counter()-t0)*1e3:.0f} ms)")
    except Exception:
        console.print("  [bold cyan]INFO[/bold cyan]  mpi4py  [yellow]not installed[/yellow]  (MPI parallelism disabled)")

    # ------------------------------------------------------------------
    # 6. Optional: ptarcade / enterprise (informational — never fails)
    # ------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        import ptarcade  # noqa: F401
        from enterprise.signals.parameter import Uniform  # noqa: F401
        console.print(f"  [bold cyan]INFO[/bold cyan]  ptarcade + enterprise  [green]found[/green]  ({(time.perf_counter()-t0)*1e3:.0f} ms)")
    except Exception:
        console.print("  [bold cyan]INFO[/bold cyan]  ptarcade + enterprise  [yellow]not installed[/yellow]  (PTA likelihood disabled)")

    # ------------------------------------------------------------------
    # Quick-check summary
    # ------------------------------------------------------------------
    console.print()
    if failures == 0:
        console.print(Panel.fit(
            f"[bold green]Quick checks passed[/bold green]  "
            f"(Python {sys.version.split()[0]}, TL on {sys.platform})",
            border_style="green",
        ))
    else:
        console.print(Panel.fit(
            f"[bold red]{failures} quick check(s) failed[/bold red] — see above",
            border_style="red",
        ))

    # ------------------------------------------------------------------
    # 7. End-to-end physics smoke test (automatic follow-up)
    # ------------------------------------------------------------------
    failures = _run_smoke(console, failures)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    console.print()
    if failures == 0:
        console.print(Panel.fit("[bold green]All checks passed[/bold green]", border_style="green"))
    else:
        console.print(Panel.fit(
            f"[bold red]{failures} check(s) failed[/bold red] — see above",
            border_style="red",
        ))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run_checks())
