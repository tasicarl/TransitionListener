"""The 4D daisy rate (variant N5) on the benchmark line v = e^{1/6} GeV/g, by exact percolation at every coupling
with a bounce-action cache in paths.DAISY_CACHE (build_daisy_table.percolate_at, kind "bounce", the action of the
paper's daisy table; couplings 0.44 to 0.8 that also have a direct-action cache), for the line curves of stage A:
the curves are drawn from these points instead of from the plane table interpolated along the line, whose nodes put
kinks into them, and they are resolved up to the end of percolation (steps of 0.0005 there). Writes
paper/data/daisy_line.json.
DAISY_LINE_KIND=bounce_rg: the same for the rate with the couplings run to mu = pi T (variant N4),
on the couplings of its own caches; writes paper/data/daisy_line_rg.json."""
import glob, json, math, os, sys
from multiprocessing import Pool
sys.argv = sys.argv[:1]
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402,F401  (the folders of paper/reproduce on sys.path)
import build_daisy_table as BD
import compute_daisy_actions as C
KIND = os.environ.get("DAISY_LINE_KIND", "bounce")
OUT = paths.data("daisy_line.json" if KIND == "bounce" else "daisy_line_rg.json")


def one(g):
    v = math.exp(1.0 / 6.0) / g
    r = BD.percolate_at(g, v, kind=KIND)
    keep = {k: (float(r[k]) if isinstance(r.get(k), (int, float)) else r.get(k))
            for k in ("ok", "T_p", "alpha", "betaH_R", "betaH_S", "beta_Gamma_H", "T_f", "P_t_end", "near_window")}
    return dict(g=g, v_GeV=v, **keep)


if __name__ == "__main__":
    gs = sorted({round(float(json.load(open(p))["g"]), 6) for p in glob.glob(os.path.join(C.CACHE, f"daisy_{KIND}_g*.json"))})
    if KIND == "bounce":
        gs = [g for g in gs if 0.44 <= g <= 0.8 + 1e-9 and os.path.exists(C.cache_path(g, "direct"))]
    with Pool(min(28, len(gs))) as pool:
        res = pool.map(one, gs)
    json.dump(res, open(OUT, "w"), indent=1)
    ok = [r for r in res if r["ok"]]
    print(f"{len(res)} couplings, {len(ok)} percolate; lowest that percolates g = {min(r['g'] for r in ok):.4f} "
          f"with (beta/H)_RH = {min(ok, key=lambda r: r['g'])['betaH_R']:.3f}; highest that does not: "
          f"{max([r['g'] for r in res if not r['ok']] or [float('nan')]):.4f}")
    print(f"wrote {OUT}")
