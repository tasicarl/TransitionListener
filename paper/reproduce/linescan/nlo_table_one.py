"""The tabulated NLO correction dS(T) for ONE parameter point at ONE scale.

At each temperature: the leading-order bounce, the four BubbleDet channels, the
decomposition offset, the double-counting subtraction and the two-loop soft
action (about twenty seconds), and at the table temperatures also the
gauge-Goldstone mixing (about seven minutes). The mixing enters halved and
linearly, so the exponent above the leading-order action is

    E(T) - S_3(T) = total_without_mixing(T) - S_3(T) + mixing(T) / 2,

and the tabulated correction is abelian_higgs_3d.action_seam_correction of that: TL's
rate formula supplies the zero-mode Jacobian itself, and the dynamical
prefactor replaces TL's A_dyn = T.

The work is done by transitionlistener.eft_rate.nlo_tables.build_table, resumable and
with the model's own matching; this script sets the input scheme, the scales
and the table temperatures, and writes the table to --out.

The table temperatures are the anchor -- the leading-order 3D percolation
temperature at the central scale -- times 0.4, 0.55, 0.75, 1, 1.5. Their
geometric centre is 0.77, close to where the next-to-leading-order
percolation temperature lies at both benchmarks (x0.79), so the table is centred on
where it will be used rather than on the leading-order point.
"""
import argparse
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
ROOT = paths.CODE

from transitionlistener.eft_rate import abelian_higgs_3d  # noqa: E402

from transitionlistener.eft_rate import nlo_tables  # noqa: E402
import dr_conformal_u1_fermions_matching as dr_dark  # noqa: E402
import TL_conformal_u1_fermions_3d as m3d_dark  # noqa: E402
import TL_conformal_u1_no_fermions_3d as m3d_nofermion  # noqa: E402

MU4_CENTRAL = float(os.environ.get("MU4_CENTRAL", "7.0553"))     # 2 pi for the settings of arXiv:2609.12098
TABLE_MULTS = (0.4, 0.55, 0.75, 1.0, 1.5)
EXTRA_MULTS = (0.25, 2.5)          # smoothness check; outside the anchor range of TABLE_MULTS
ASSEMBLY = nlo_tables.ASSEMBLY

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--g", type=float, required=True)
ap.add_argument("--model", default="dark_u1", choices=("dark_u1", "u1"),
                help="u1 is the fermion-free model of arXiv:2511.02910 and "
                     "2609.12098; it takes the zero-quartic input scheme only")
ap.add_argument("--y", type=float, default=None, help="dark_u1 only")
ap.add_argument("--input-scheme", default="CW", choices=("CW", "zero_quartic"),
                help="dark_u1 only: CW (the TL model's quartic at mu0 = g v, default) or zero_quartic "
                     "(lambda(mu0) = 0 with the minimum at v, as the u1 model and arXiv:2609.12098)")
ap.add_argument("--mu3-choice", default=None, choices=("g3sq", "gT"),
                help="u1, and dark_u1 with zero_quartic: the 3D scale, g3sq (default) or gT (theirs)")
ap.add_argument("--v", type=float, required=True)
ap.add_argument("--mu4", type=float, default=1.0)
ap.add_argument("--anchor", type=float, required=True, help="GeV")
ap.add_argument("--l-max", type=int, default=abelian_higgs_3d.MIXING_L_MAX)
ap.add_argument("--table-mults", default=",".join(f"{m:g}" for m in TABLE_MULTS),
                help="multiples of the anchor that get the mixing and enter the "
                     "table; add values to extend a table (existing rows are kept)")
ap.add_argument("--out", required=True)
args = ap.parse_args()
TABLE_MULTS = tuple(sorted({float(x) for x in args.table_mults.split(",")}))

if args.model == "dark_u1":
    if args.y is None:
        ap.error("--y is required for the dark_u1 model")
    pot = m3d_dark.specific_potential({"g": args.g, "y": args.y, "v_GeV": args.v}, verbose=False)
    if args.input_scheme == "zero_quartic":
        lam0, mu0 = 0.0, args.v / dr_dark.vev_from_zero_quartic(args.g, args.y, 1.0)
    else:
        lam0, mu0 = pot.l, None
else:
    # zero-quartic input: lambda(mu0) = 0 at mu0 = g v e^{-1/6} (2511.02910 eq. 2.2)
    pot = m3d_nofermion.specific_potential({"g": args.g, "v_GeV": args.v}, verbose=False)
    lam0, mu0 = 0.0, args.g * args.v * math.exp(-1.0 / 6.0)
conf = pot.config.nucleationConf
conf.mu4_over_T = MU4_CENTRAL * args.mu4
conf.matching_order = "NLO"
conf.input_scheme = args.input_scheme if args.model == "dark_u1" else "zero_quartic"
if args.mu3_choice is not None:
    conf.mu3_choice = args.mu3_choice

metadata = dict(g=args.g, y=args.y, v_GeV=args.v, lambda0=lam0, mu4=args.mu4, model=args.model, mu0_GeV=mu0,
                mu3_choice=args.mu3_choice, anchor_GeV=args.anchor)
if args.model == "dark_u1":
    metadata.update(input_scheme=args.input_scheme)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] g={args.g:.3f} mu4x{args.mu4:g}  {msg}", flush=True)


mults = sorted(set(TABLE_MULTS) | set(EXTRA_MULTS))
nlo_tables.build_table(pot, [(f"{m:g}", args.anchor * m, dict(mult=m)) for m in mults], args.out,
                       l_max=args.l_max, metadata=metadata, log=log)
