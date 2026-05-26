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
import warnings
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

    # Persistent output dir so the user can inspect the result of the
    # smoke run after ``tl --check`` exits.  Each invocation gets its
    # own timestamped subdirectory under the system temp dir; we print
    # a clickable file:// link to it at the end of this function.
    runs_root = Path(tempfile.gettempdir()) / "tl_check_runs"
    runs_root.mkdir(exist_ok=True)
    tmpdir = runs_root / f"example_point_{int(time.time())}"
    tmpdir.mkdir()
    try:
        (tmpdir / "examples").mkdir()
        (tmpdir / "models").mkdir()
        shutil.copy(yaml_src, tmpdir / "examples" / "example_point.yaml")
        shutil.copy(model_src, tmpdir / "models" / "TL_conformal_dark_u1.py")

        env = os.environ.copy()
        env.update({
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            # Force the child to flush stdout line by line so our streaming
            # progress indicator below sees output as it is produced.
            "PYTHONUNBUFFERED": "1",
        })

        console.print(
            "  Running example_point.yaml  "
            "[dim](conformal U(1), SinglePoint — ~30-60 s after warm-up; "
            "the first run can take 2-3 min while NumPy/SciPy/matplotlib "
            "warm their caches)[/dim]"
        )
        t0 = time.perf_counter()
        # Prefer the `tl` script from the same env as sys.executable.
        tl_script = Path(sys.executable).parent / "tl"
        if not tl_script.exists():
            tl_script = Path(sys.executable).parent / "tl.exe"  # Windows
        # ``-v`` switches on the per-stage ``print("Calculating ...")``
        # messages in transitionObservables.py.  They are what
        # STAGE_HINTS below converts into bouncing-ball status labels,
        # so without ``-v`` the second ball never appears.
        cmd = (
            [str(tl_script), "-c", "examples/example_point.yaml", "-j", "1", "-v"]
            if tl_script.exists()
            else [sys.executable, "-c",
                  "from transitionlistener.interface import main; main()",
                  "-c", "examples/example_point.yaml", "-j", "1", "-v"]
        )

        # Stream the subprocess output and surface progress through a
        # bouncing-ball status line. Without this, first-time users (when
        # nothing is cached yet) see a blank terminal for several minutes
        # and assume the run is frozen.
        STAGE_HINTS = (
            ("Tracing phase",                       "Tracing phases"),
            ("Calculating critical temperature",    "Locating critical temperature"),
            ("Calculating critical vev",            "Locating critical vev"),
            ("Calculating nucleation temperature",  "Locating nucleation temperature"),
            ("Calculating percolation splines",     "Building percolation splines"),
            ("Calculating percolation temperature", "Locating percolation temperature"),
            ("Calculating reheating temperature",   "Locating reheating temperature"),
            ("Calculating sound speed",             "Computing sound speed"),
            ("Calculating alpha parameters",        "Computing alpha (latent heat)"),
            ("Calculating beta/H",                  "Computing beta/H"),
            ("Calculating mean bubble separation",  "Computing mean bubble separation"),
            ("Calculating kappa parameters",        "Computing efficiency factors"),
            ("Calculating g_eff_tot_reh",           "Tabulating reheating dofs"),
            ("Calculating h_eff_tot_reh",           "Tabulating reheating dofs"),
            ("Calculating Tf_SM_GeV",               "Locating final temperature"),
            ("Computed thermodynamic quantities",   "Writing observables"),
        )

        captured: list[str] = []
        timed_out = False
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=tmpdir,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except OSError as exc:
            _result("smoke test", False, f"could not launch: {exc}")
            return failures + 1

        # 10 minute hard cap — first-time runs with cold caches on slower
        # CPUs have been observed to take up to ~5 min.
        deadline = t0 + 600
        # Two-stage status: a "Warming up" bouncing ball runs while the
        # subprocess imports NumPy / SciPy / matplotlib and builds its
        # caches; when the first STAGE_HINTS match shows up on stdout we
        # close that ball, print a one-line confirmation, and open a
        # fresh bouncing ball for the actual physics run. This way the
        # user can tell warm-up from the run, and the run's status label
        # tracks the current physics stage.
        status = console.status(
            "[bold cyan]Warming up TransitionListener…[/bold cyan]",
            spinner="bouncingBall",
        )
        status.start()
        warmed = False
        # Most of the per-stage ``print("Calculating ...")`` lines fire in a
        # tight burst near the end of the run (they label the assignment of
        # derived quantities, where the heavy work happened in earlier
        # steps).  Holding each label for at least MIN_LABEL_HOLD seconds
        # lets the user actually see them roll past instead of catching
        # only the last one before the run exits.
        MIN_LABEL_HOLD = 0.4
        last_label_t = 0.0
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                captured.append(line)
                s = line.strip()
                for needle, pretty in STAGE_HINTS:
                    if needle in s:
                        if not warmed:
                            status.stop()
                            console.print(
                                "  [green]✓[/green]  Warm-up complete — "
                                "starting the physics run"
                            )
                            status = console.status(
                                f"[bold cyan]{pretty}…[/bold cyan]",
                                spinner="bouncingBall",
                            )
                            status.start()
                            warmed = True
                            last_label_t = time.perf_counter()
                        else:
                            now = time.perf_counter()
                            held = now - last_label_t
                            if held < MIN_LABEL_HOLD:
                                time.sleep(MIN_LABEL_HOLD - held)
                            status.update(f"[bold cyan]{pretty}…[/bold cyan]")
                            last_label_t = time.perf_counter()
                        break
                if time.perf_counter() > deadline:
                    proc.kill()
                    timed_out = True
                    break
            proc.wait()
        finally:
            status.stop()
            if proc.stdout is not None:
                proc.stdout.close()
        elapsed = time.perf_counter() - t0

        if timed_out:
            _result("smoke test", False, "timed out after 600 s")
            return failures + 1

        if proc.returncode != 0:
            _result("smoke test", False, f"exit code {proc.returncode}")
            console.print(f"  [dim]{''.join(captured)[-500:]}[/dim]")
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
    finally:
        # Always print a clickable link to the run directory — even on
        # early returns / failures, so the user can inspect the output
        # (and the partial logs) of the just-finished smoke run.
        # ``soft_wrap=True`` keeps the path on a single line so a
        # terminal-width line break can't truncate copy-pastes or
        # clickable hyperlinks mid-path.
        console.print(
            f"  [dim]Output saved to[/dim] "
            f"[link=file://{tmpdir}]{tmpdir}[/link]",
            soft_wrap=True,
        )
        # Friendly "what's next" hint pointing at the same workspace —
        # the smoke-test tmpdir already contains a complete
        # ``examples/`` + ``models/`` layout, so the user can rerun
        # the conformal U(1) example interactively with one command
        # without hunting down the package install location.
        console.print(
            "  [dim]New to TransitionListener?  Try the same run "
            "interactively:[/dim]\n"
            f"    [bold]cd {tmpdir} && tl -c examples/example_point.yaml -v[/bold]",
            soft_wrap=True,
        )

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
        # SciPy's select_initial_step heuristic divides f0 by atol and squares
        # the result, which overflows float64 because percIntegralODE uses
        # atol=1e-300 (set deliberately to resolve the hot-tail exp(-S/T)
        # suppression). NumPy >= 2 surfaces this as a benign RuntimeWarning,
        # which is unrelated to the correctness of the result. Silence it
        # locally so the check output stays clean.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="overflow encountered in dot",
                category=RuntimeWarning,
            )
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
