"""One point of the line scan, all treatments and the scale band. Resumable.

  stage 1   4D daisy (central); 3D leading-order action at each mu4 in the band
  anchor    the 3D leading-order percolation temperature at the central scale
  stage 2   the NLO correction table at each mu4 (nlo_table_one.py)
  stage 3   3D next-to-leading-order action at each mu4
  validity  the EFT validity ratios at the central NLO percolation temperature
  uncertainty  the error budget of T_perc there: the mu4 band, the next order
            (|Delta S| g3^2/(pi m3)) and the first dropped operator (Delta S_6),
            each as a fraction of T_perc through beta/H = T dS/dT

Every pipeline run is its own process (several treatments in one interpreter
share state), and runs within a stage go in parallel. Anything already on disk is skipped, so a
killed job resumes where it stopped. Matching is NLO throughout.
"""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.REPO

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--g", type=float, required=True)
ap.add_argument("--model", default="dark_u1", choices=("dark_u1", "u1"),
                help="u1: the fermion-free model of arXiv:2511.02910 and "
                     "2609.12098, with the zero-quartic input scheme")
ap.add_argument("--y", type=float, default=0.01, help="dark_u1 only")
ap.add_argument("--input-scheme", default="CW", choices=("CW", "zero_quartic"),
                help="dark_u1 only: CW (default) or zero_quartic, lambda(mu0) = 0 with the minimum at v")
ap.add_argument("--mu3-choice", default=None, choices=("g3sq", "gT"),
                help="u1 only: the 3D scale, g3sq (default) or gT (theirs)")
ap.add_argument("--stages", default="1,2,3",
                help="which stages to run; 1,2 stops after the NLO tables")
ap.add_argument("--v", type=float, default=0.173, help="GeV")
ap.add_argument("--band", default="0.5,1,2", help="mu4 multiples")
ap.add_argument("--l-max", type=int, default=70)
ap.add_argument("--root", default=paths.LINESCAN_POINTS)
ap.add_argument("--algorithm", default="adaptive_step_size",
                choices=("adaptive_step_size", "fixed_step_size"),
                help="percolation solver for every pipeline run of this point; see pipeline_one.py")
args = ap.parse_args()

# Preflight: every stage's imports, in this interpreter, before any work. The
# workers record per-row errors and carry on, so a missing BubbleDet would
# otherwise let the scan finish with every NLO table failed.
try:
    import BubbleDet  # noqa: F401  (stage 2)
    import transitionlistener  # noqa: F401
except ImportError as exc:
    raise SystemExit(f"preflight: {exc} under {sys.executable}; refusing to run "
                     "a scan that would fail stage 2 at every temperature")

BAND = [float(x) for x in args.band.split(",")]
if 1.0 not in BAND:
    raise SystemExit("the band must contain the central scale 1")
STAGES = {int(x) for x in args.stages.split(",")}
TAG = (f"y{args.y:g}" + ("_zq" if args.input_scheme == "zero_quartic" else "")) if args.model == "dark_u1" else "u1"
PT = os.path.join(args.root, f"g{args.g:.3f}_{TAG}_v{args.v:g}")
os.makedirs(PT, exist_ok=True)
PY = sys.executable


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] g={args.g:.3f}  {msg}", flush=True)


def common(for_pipeline=True):
    c = ["--g", str(args.g), "--v", str(args.v), "--model", args.model]
    if args.model == "dark_u1":
        c += ["--y", str(args.y)]
        if args.input_scheme == "zero_quartic":
            c += ["--input-scheme", "zero_quartic"]
            if args.mu3_choice:
                c += ["--mu3-choice", args.mu3_choice]
    else:
        # the fermion-free model is defined with lambda(mu0) = 0
        if for_pipeline:
            c += ["--input-scheme", "zero_quartic"]
        if args.mu3_choice:
            c += ["--mu3-choice", args.mu3_choice]
    return c


def run_parallel(jobs):
    """jobs: list of (label, argv, out_path). Skips outputs already present."""
    procs = []
    for label, argv, out in jobs:
        if os.path.exists(out):
            log(f"{label}: already done")
            continue
        logf = open(out.replace(".json", ".log"), "w")
        procs.append((label, subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT),
                      logf, time.time()))
        log(f"{label}: started")
    for label, pr, logf, t0 in procs:
        pr.wait()
        logf.close()
        log(f"{label}: finished in {time.time() - t0:.0f} s (exit {pr.returncode})")


def load(path):
    return json.load(open(path)) if os.path.exists(path) else None


def pipe(mode, mu, table=None):
    out = os.path.join(PT, f"{mode}_mu{mu:g}.json")
    argv = [PY, os.path.join(HERE, "pipeline_one.py"), *common(), "--mode", mode,
            "--mu4", str(mu), "--matching", "NLO", "--algorithm", args.algorithm,
            "--out", out]
    if table:
        argv += ["--table", table]
    return (f"{mode} mu4x{mu:g}", argv, out)


# ---------------- stage 1 ----------------
log("stage 1: 4D baseline and 3D leading-order action across the band")
run_parallel([pipe("4d", 1.0)] + [pipe("lo3d", mu) for mu in BAND])

central_lo = load(os.path.join(PT, "lo3d_mu1.json"))
anchor = (central_lo or {}).get("observables", {}).get("Tperc_SM_GeV")
summary = dict(g=args.g, y=args.y, v_GeV=args.v, model=args.model,
               mu3_choice=args.mu3_choice, band=BAND, anchor_GeV=anchor,
               algorithm=args.algorithm,
               **({"input_scheme": args.input_scheme} if args.model == "dark_u1" else {}))
if anchor is None:
    summary["stopped"] = ("no leading-order 3D percolation temperature at the "
                          "central scale, so there is nothing to anchor the NLO "
                          "table on; error: " + str((central_lo or {}).get("error")))
    log(summary["stopped"][:200])
else:
    log(f"anchor: leading-order 3D T_perc = {anchor:.5e} GeV")

    # ---------------- stage 2 ----------------
    log("stage 2: NLO correction tables")
    jobs = []
    for mu in BAND:
        out = os.path.join(PT, f"table_mu{mu:g}.json")
        done = (load(out) or {}).get("table") is not None or "table_error" in (load(out) or {})
        argv = [PY, os.path.join(HERE, "nlo_table_one.py"), *common(for_pipeline=False),
                "--mu4", str(mu),
                "--anchor", repr(anchor), "--l-max", str(args.l_max), "--out", out]
        jobs.append((f"table mu4x{mu:g}", argv, out if done else out + ".pending"))
    # nlo_table_one resumes by itself, so "pending" markers only gate relaunching
    procs = []
    for label, argv, marker in jobs:
        if not marker.endswith(".pending"):
            log(f"{label}: already done")
            continue
        logf = open(marker.replace(".json.pending", ".log"), "a")
        procs.append((label, subprocess.Popen(argv, stdout=logf, stderr=subprocess.STDOUT), logf, time.time()))
        log(f"{label}: started")
    for label, pr, logf, t0 in procs:
        pr.wait(); logf.close()
        log(f"{label}: finished in {time.time() - t0:.0f} s (exit {pr.returncode})")

    # ---------------- stage 3 ----------------
    if 3 in STAGES:
        log("stage 3: 3D next-to-leading-order action across the band")
        run_parallel([pipe("nlo3d", mu, os.path.join(PT, f"table_mu{mu:g}.json"))
                      for mu in BAND])
    else:
        log("stage 3: skipped (--stages)")

# ---------------- collect ----------------
for mode in ("4d", "lo3d", "nlo3d"):
    for mu in ([1.0] if mode == "4d" else BAND):
        rec = load(os.path.join(PT, f"{mode}_mu{mu:g}.json"))
        if rec is not None:
            summary.setdefault(mode, {})[f"{mu:g}"] = dict(
                observables=rec["observables"], error=rec["error"],
                extrapolating=rec.get("extrapolating"),
                table_range=rec.get("table_range"), seconds=rec.get("seconds"))
for mu in BAND:
    t = load(os.path.join(PT, f"table_mu{mu:g}.json"))
    if t:
        summary.setdefault("tables", {})[f"{mu:g}"] = dict(
            table=t.get("table"), table_error=t.get("table_error"),
            rows={k: {kk: vv for kk, vv in r.items() if kk != "mixing"}
                  | ({"mixing": r["mixing"]["value"], "mixing_spread": r["mixing"]["spread"]}
                     if "mixing" in r else {})
                  for k, r in t.get("rows", {}).items()})

# ---------------- validity and uncertainty at the central NLO (else LO) percolation temperature ----------------
try:
    import math

    from transitionlistener.eft_rate import abelian_higgs_3d

    from transitionlistener.eft_rate import validity
    import math as _math
    if args.model == "dark_u1":
        import dr_conformal_u1_fermions_matching as dr
        import TL_conformal_u1_fermions_3d as m3d
    else:
        import dr_conformal_u1_no_fermions_matching as dr
    nlo_c = (summary.get("nlo3d", {}).get("1", {}) or {}).get("observables", {})
    T_val = nlo_c.get("Tperc_SM_GeV") or anchor
    if T_val:
        if args.model == "dark_u1" and args.input_scheme == "zero_quartic":
            p = dr.matched_soft_parameters_zero_quartic(
                args.g, args.y, args.v / dr.vev_from_zero_quartic(args.g, args.y, 1.0), T_val,
                matching_order="NLO", mu3=(args.g * T_val if args.mu3_choice == "gT" else None))
        elif args.model == "dark_u1":
            lam0 = m3d.specific_potential({"g": args.g, "y": args.y, "v_GeV": args.v},
                                          verbose=False).l
            p = dr.matched_soft_parameters(args.g, args.y, lam0, args.v, T_val,
                                           matching_order="NLO")
        else:
            mu0 = args.g * args.v * _math.exp(-1.0 / 6.0)
            p = dr.matched_soft_parameters(
                args.g, 0.0, mu0, T_val, matching_order="NLO",
                mu3=(args.g * T_val if args.mu3_choice == "gT" else None))
        S3, r, phi3, _ = abelian_higgs_3d.solve_LO_bounce(p["m3sq"], p["lam3"], p["g3sq"], m_debye_sq=p["mD2"])
        ratios = abelian_higgs_3d.validity_ratios(T_val, p["m3sq"], p["lam3"], p["g3sq"], p["mD2"],
                                       r=r, phi3=phi3,
                                       fermion_masses_over_field=p["fermion_masses_over_field"],
                                       c6=p["c6"], S3_over_T=S3)
        # the soft theory throughout (A_0 kept); NLO where the NLO T_perc exists
        order = "NLO" if nlo_c.get("Tperc_SM_GeV") else "LO"
        report = validity.evaluate(ratios, branch="soft", order=order)
        summary["validity"] = dict(T_GeV=T_val, branch="soft", order=order, ratios=ratios,
                                   verdict=report.verdict, missing=sorted(report.missing),
                                   messages=report.messages)

        unc = dict(T_GeV=T_val, S3_LO=S3, delta_S6=abelian_higgs_3d.dim6_action_shift(r, phi3, p["c6"]))
        tab = load(os.path.join(PT, "table_mu1.json")) or {}
        if nlo_c.get("Tperc_SM_GeV") and tab.get("table"):
            t = tab["table"]
            corr = abelian_higgs_3d.NLOCorrection(t["T_GeV"], t["delta_S"], S3_LO=t.get("S3_LO"))
            unc["delta_S"] = float(corr(T_val, S3_LO=S3))
            unc["nlo_is_large"] = validity.nlo_is_large(unc["delta_S"], S3)
        u = validity.ExponentUncertainty.from_ratios(
            unc.get("delta_S", 0.0), ratios["expansion_3d"], unc["delta_S6"])
        unc["exponent"] = dict(nnlo=u.nnlo, dim6=u.dim6)
        bH = nlo_c.get("betaH_S3") or ((summary.get("lo3d", {}).get("1", {}) or {})
                                       .get("observables", {}).get("betaH_S3"))
        if bH:
            frac = u.on_tperc(bH)
            mode = "nlo3d" if nlo_c.get("Tperc_SM_GeV") else "lo3d"
            vals = [((summary.get(mode, {}).get(f"{mu:g}", {}) or {}).get("observables", {})
                     .get("Tperc_SM_GeV")) for mu in BAND]
            vals = [v for v in vals if v]
            if len(vals) == len(BAND):
                frac["mu4_half_band"] = 0.5 * (max(vals) - min(vals)) / T_val
            frac["total"] = math.sqrt(sum(v**2 for v in frac.values()))
            unc["betaH_S3"] = bH
            unc["Tperc_fraction"] = frac
        summary["uncertainty"] = unc
except Exception as exc:
    summary["validity_error"] = f"{type(exc).__name__}: {exc}"

failed_tables = [mu for mu, t in summary.get("tables", {}).items() if t.get("table") is None]
if anchor is not None and failed_tables:
    summary["stage2_failed"] = failed_tables
summary["python"] = sys.executable
with open(os.path.join(PT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
log("summary written")
if summary.get("stage2_failed"):
    log(f"STAGE 2 FAILED for mu4 x {failed_tables}: no NLO table, so no NLO result")
    sys.exit(3)
