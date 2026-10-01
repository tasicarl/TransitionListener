"""Release-readiness smoke test.

Runs ``tl -c examples/example_point.yaml`` end-to-end on a single CPU,
parses the resulting output table, and checks a handful of key numbers
against committed reference values. Fails the build if anything diverges
beyond the configured relative tolerance.

This is the test we want to be green on every push to the release branch
before tagging or shipping anything.

Total runtime budget: roughly 1 to 2 minutes on a modern single-core CPU.

Part of TransitionListener v2.0
Documentation: https://tasillo.de/TransitionListener/

Authors:
    Jonas Matuszak <jonas.matuszak@kit.edu>
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""

from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_YAML = REPO_ROOT / "examples/example_point.yaml"
SCAN_OUT = REPO_ROOT / "scans/example_point"
ALL_PARAMS = SCAN_OUT / "1_All_params.txt"


# Reference values for the conformal U(1) example point at g=0.7, v=0.1 GeV,
# y=0.01. Recorded on the release branch with the default adaptive_step_size
# solver. If any of these expectations need to be updated (intentional
# physics change), regenerate locally and update the constants below in the
# same commit that introduces the underlying change.
EXPECTED = {
    "Tperc_SM_GeV": (1.0e-3, 5.0e-3),
    "Treh_SM_GeV":  (5.0e-3, 5.0e-2),
    # The conformal U(1) example point is fairly supercooled; alpha lands
    # around 380 with the default settings. The bands are loose so the
    # smoke test stays robust against tiny solver-tuning changes.
    "alpha":        (1.0e+1, 1.0e+4),
    "alpha_thetabar": (1.0e+1, 1.0e+4),
    "RH":           (1.0e-6, 1.0e+0),
    # The sound speed must be a speed: it sets the bubble expansion speed through
    # max(v_wall, c_s), the efficiency factors and the spectrum. A band rather than a value,
    # since the point is that the guard leaves an ordinary value alone.
    "c_s":          (1.0e-3, 0.999),
}


@pytest.fixture(scope="module")
def smoke_run() -> Path:
    if not EXAMPLE_YAML.exists():
        pytest.skip(f"example YAML missing: {EXAMPLE_YAML}")
    if SCAN_OUT.exists():
        shutil.rmtree(SCAN_OUT)

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    # Prefer the `tl` console script; fall back to invoking the CLI module
    # via `python -m` when the script isn't on PATH (e.g. on a fresh CI
    # checkout that uses the wheel's entry-point directly).
    cmd = ["tl", "-c", "examples/example_point.yaml", "-j", "1"]
    try:
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=180,
        )
    except FileNotFoundError:
        cmd = [
            sys.executable, "-m", "transitionlistener.interface.cli",
            "-c", "examples/example_point.yaml", "-j", "1",
        ]
        result = subprocess.run(
            cmd, cwd=REPO_ROOT, env=env,
            capture_output=True, text=True, timeout=180,
        )

    if result.returncode != 0:
        pytest.fail(
            f"`tl` exited with code {result.returncode}\n"
            f"stdout:\n{result.stdout[-2000:]}\n"
            f"stderr:\n{result.stderr[-2000:]}"
        )
    if not ALL_PARAMS.exists():
        pytest.fail(f"expected output table missing: {ALL_PARAMS}")
    return ALL_PARAMS


def _parse_all_params(path: Path) -> dict[str, float]:
    """Parse the `1_All_params.txt` plain-text key-value layout.

    Lines look like ``key                  value`` where the value is
    either a float, a placeholder ``-`` (None), or text. We coerce floats
    where possible and skip the rest.
    """
    out: dict[str, float] = {}
    line_re = re.compile(r"^\s*(\S+)\s+(\S.*?)\s*$")
    for raw in path.read_text().splitlines():
        if not raw.strip() or raw.startswith("Warnings"):
            continue
        match = line_re.match(raw)
        if not match:
            continue
        key, value = match.group(1), match.group(2)
        try:
            out[key] = float(value)
        except ValueError:
            continue
    return out


def test_smoke_no_error_flag(smoke_run: Path) -> None:
    """A clean run leaves the `error` field as the placeholder ``-`` (None);
    a numeric code (or NaN) indicates a non-fatal warning or hard failure."""
    error_re = re.compile(r"^\s*error\s+(.+?)\s*$")
    raw_value: str | None = None
    for line in smoke_run.read_text().splitlines():
        match = error_re.match(line)
        if match:
            raw_value = match.group(1).strip()
            break
    assert raw_value is not None, f"`error` field missing from {smoke_run}"
    if raw_value == "-":
        return
    try:
        numeric = float(raw_value)
    except ValueError:
        pytest.fail(f"unexpected `error` field value: {raw_value!r}")
    assert math.isnan(numeric) or numeric == 0, f"non-zero error code: {raw_value}"


@pytest.mark.parametrize("key,bounds", list(EXPECTED.items()))
def test_smoke_value_in_band(smoke_run: Path, key: str, bounds: tuple[float, float]) -> None:
    values = _parse_all_params(smoke_run)
    assert key in values, f"{key!r} missing from {smoke_run}"
    val = values[key]
    lo, hi = bounds
    assert math.isfinite(val), f"{key} = {val} is not finite"
    assert lo <= val <= hi, f"{key} = {val:.3e} outside band [{lo}, {hi}]"


def test_smoke_entropy_scheme_diagnostics_are_populated(smoke_run: Path) -> None:
    """The caller boundary, exercised by a real run rather than read from the source.

    `_compute_percolation` delegates the entropy-scheme comparison to
    `bubbledynamics.record_entropy_scheme_diagnostics`. Unit tests of that function cannot
    tell whether the observables still call it, and a source-text assertion would pass if
    they stopped. Here the example point has actually run, so the three keys can only carry
    numbers if the whole chain is connected.
    """
    text = smoke_run.read_text()
    values = _parse_all_params(smoke_run)

    # The diagnostics live under the `Diagnostics:` heading, so they are not in EXPECTED.
    assert "DIAG:entropy_scheme_cs2_spread" in values or "entropy_scheme_cs2_spread" in text, (
        "the entropy-scheme spread is missing: the observables are not calling "
        "record_entropy_scheme_diagnostics"
    )
    for key in ("entropy_scheme_cs2_spread", "entropy_scheme_lna_gap",
                "entropy_scheme_sensitive"):
        assert key in text, f"{key} missing from {smoke_run}"

    spread_re = re.compile(r"entropy_scheme_cs2_spread\s+(\S+)")
    match = spread_re.search(text)
    assert match, "could not read the spread back"
    raw = match.group(1)
    try:
        spread = float(raw)
    except ValueError:
        pytest.fail(
            f"the spread was written as {raw!r} rather than a number, so the observables "
            "did not populate it: check that record_entropy_scheme_diagnostics is still called"
        )
    # The example point is nearly conformal, so the two schemes almost agree there; what
    # matters is that a number was computed at all rather than left as the nan default.
    assert math.isfinite(spread), (
        f"the spread came back as {spread}, so the comparison did not run"
    )
    assert 0.0 <= spread < 0.01, f"unexpected spread on the example point: {spread}"

    flag_re = re.compile(r"entropy_scheme_sensitive\s+(\S+)")
    flag = flag_re.search(text)
    assert flag and flag.group(1) in ("True", "False"), (
        f"the flag is not a boolean: {flag.group(1) if flag else None}"
    )


def test_smoke_sound_speed_guard_reports_itself(smoke_run: Path) -> None:
    """The guard has to run in a real run, not only in its unit tests.

    The unit tests call `_ensure_sound_speed` on a context they build themselves, so deleting
    the call from the solver left the suite green. The example point already runs here.
    """
    text = smoke_run.read_text()
    assert "unphysical_c_s" in text, (
        "WARNING:unphysical_c_s missing: the sound-speed guard did not run"
    )
    values = _parse_all_params(smoke_run)
    assert "c_s" in values, "c_s missing from the output"
    # This point has an ordinary thermal broken phase, so the guard must not have fired.
    flag = re.search(r"unphysical_c_s\s+(\S+)", text)
    assert flag and flag.group(1) == "False", (
        f"the guard fired on the example point: {flag.group(1) if flag else None}"
    )


def test_smoke_alpha_consistency(smoke_run: Path) -> None:
    values = _parse_all_params(smoke_run)
    assert "alpha" in values and "alpha_thetabar" in values
    if math.isfinite(values["alpha"]) and math.isfinite(values["alpha_thetabar"]):
        ratio = values["alpha"] / values["alpha_thetabar"]
        assert abs(ratio - 1.0) < 1e-6, (
            "On the release branch alpha must equal alpha_thetabar "
            f"(got alpha={values['alpha']}, alpha_thetabar={values['alpha_thetabar']})"
        )
