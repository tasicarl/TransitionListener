"""Command-line interface to the NLO tables of the 3D rate: ``tl-nlo-table``.

    tl-nlo-table build --model-file models/TL_conformal_u1_fermions_3d.py --param g=0.7 --param y=0.5 \
        --out-stem nodes/g0.7_y0.5 --mu4-factors 0.5,1,2
    tl-nlo-table assemble --nodes "nodes/*.json" --out grid.npz
    tl-nlo-table info grid.npz

The model is any model file whose potential mixes in ``eft_rate.one_field_rate_3d.ThreeDimensionalRate`` and supplies
what ``eft_rate.nlo_tables`` needs of it (``nlo_grid_name``, ``nlo_grid_point``, ``input_scale_GeV``,
``two_loop_potential``, ``two_loop_kinetic``). The models shipped with this support are the two conformal U(1) models,
``models/TL_conformal_u1_fermions_3d.py`` and ``models/TL_conformal_u1_no_fermions_3d.py``. A node table records the
names of the couplings it was built at (the keys of ``nlo_grid_point``), and ``assemble`` reads them from there.

``build`` computes one coupling point and is resumable. ``assemble`` collects node tables into a grid file and reports
the leave-one-out interpolation error. A run with ``include_determinants`` reads such a grid; see
transitionlistener/nlo_tables.py.

Documentation: https://tasillo.de/TransitionListener/

Authors:
    Carlo Tasillo <carlo.tasillo@ific.uv.es>
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time


#: The values of S_LO of a production table. At fixed couplings a table in S_LO holds for every v: the upper end is above
#: the action at which the rate reaches H^4 for any v below which a transition is possible, the lower end for v up to
#: order 1e12 GeV.
DEFAULT_S_NODES = (5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 105, 120, 135, 150, 165, 180, 200, 220, 245, 270, 300, 350)


#: The scale mu0 of a table, which does not depend on it: near y^4 = 3 g^4 a v of 1 GeV would put the temperatures at 1e100 GeV.
BUILD_MU0_GEV = 1.0


def _floats(text):
    return [float(q) for q in text.split(",") if q.strip()]


def _params(pairs):
    out = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not name or not value:
            raise SystemExit(f"--param takes name=value, not {pair!r}")
        out[name.strip()] = float(value)
    return out


def _build(args):
    from transitionlistener.eft_rate import nlo_tables

    model = nlo_tables.load_model(args.model_file, args.potential)
    params = _params(args.param)
    # the table in S_LO does not depend on the vacuum expectation value; without one given, a model whose vev is the
    # parameter v_GeV is run at the v of an input scale of BUILD_MU0_GEV
    rescale_v = "v_GeV" not in params and "v_GeV" in getattr(model, "model_parameters", {})
    if rescale_v:
        params["v_GeV"] = 1.0

    def configured():
        pot = model(params, verbose=False)
        conf = pot.config.nucleationConf
        conf.rate_mode = "model"
        conf.mu4_over_T = args.mu4_over_T
        conf.input_scheme = args.input_scheme
        conf.mu3_choice = args.mu3_choice
        conf.temporal_coupling = args.temporal_coupling
        conf.matching_order = "NLO"
        return pot

    pot = configured()
    if rescale_v:
        params["v_GeV"] = params["v_GeV"] * BUILD_MU0_GEV / pot.input_scale_GeV()
        pot = configured()

    def log(msg):
        print(f"[{time.strftime('%H:%M:%S')}] {params}  {msg}", flush=True)

    files = nlo_tables.build_node(pot, _floats(args.S_nodes), args.out_stem, mu4_factors=_floats(args.mu4_factors),
                                  l_max=args.l_max, metadata=nlo_tables.model_metadata(pot, pot.nlo_grid_name), log=log)
    failed = 0
    for path in files:
        with open(path) as f:
            rows = json.load(f)["rows"]
        # a row below the coldest bounce of a node fails by construction: the node's floor in the grid
        held = [r["S3_LO"] for r in rows.values() if "mixing" in r]
        floor = min(held, default=0.0)
        failed += sum(1 for r in rows.values() if ("mixing" not in r or "mixing_error" in r)
                      and r.get("target_S", float("inf")) > floor)
    if failed:
        print(f"{failed} rows failed; see the files {files}", file=sys.stderr)
    return 1 if failed else 0


def _assemble(args):
    from transitionlistener.eft_rate import nlo_tables

    grouped = {}
    for pattern in args.nodes:
        for path in sorted(glob.glob(pattern)):
            with open(path) as f:
                grouped.setdefault(float(json.load(f).get("mu4_factor", 1.0)), []).append(path)
    if not grouped:
        raise SystemExit(f"no node tables match {args.nodes}")
    couplings = [c for c in args.couplings.split(",") if c] if args.couplings else None
    head = nlo_tables.assemble_grid(grouped, _floats(args.S_nodes), args.out, purpose=args.purpose, couplings=couplings)
    print(json.dumps(head["leave_one_out"], indent=2))
    return 0


def _info(args):
    from transitionlistener.eft_rate import nlo_tables

    grid = nlo_tables.load_grid(args.grid)
    print(json.dumps(grid.header, indent=2))
    print(f"{len(grid.points)} nodes: {grid.coupling_range()}; S_LO in [{grid.S_nodes[0]:g}, {grid.S_nodes[-1]:g}], "
          f"{grid.S_nodes.size} values")
    return 0


def main(argv=None):
    """Console entry point."""
    parser = argparse.ArgumentParser(prog="tl-nlo-table", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build the table of one coupling point (resumable)")
    b.add_argument("--model-file", required=True, help="the model file, e.g. models/TL_conformal_u1_fermions_3d.py")
    b.add_argument("--potential", default="specific_potential", help="the potential class in it")
    b.add_argument("--param", action="append", default=[], metavar="NAME=VALUE",
                   help="a model parameter; repeat for each (the couplings of the table, e.g. g=0.7)")
    b.add_argument("--mu4-over-T", type=float, default=7.0553, help="central 4D matching scale (default 7.0553)")
    b.add_argument("--mu4-factors", default="1", help="comma-separated factors of mu4; one table for each (default 1)")
    b.add_argument("--mu3-choice", default="gT", choices=("gT", "g3sq"), help="3D scale (default gT, g(mu0) T)")
    b.add_argument("--input-scheme", default="zero_quartic", choices=("zero_quartic", "CW"))
    b.add_argument("--temporal-coupling", default="h3", choices=("h3", "2g3sq"),
                   help="A0^2 phi^2 coupling of the temporal mass: the NLO-matched h3 (default) or 2 g3^2")
    b.add_argument("--l-max", type=int, default=None, help="last partial wave of the mixing (default abelian_higgs_3d.MIXING_L_MAX)")
    b.add_argument("--S-nodes", default=",".join(f"{s:g}" for s in DEFAULT_S_NODES), help="values of S_LO")
    b.add_argument("--out-stem", required=True, help="files <stem>_mu<factor>.json")
    b.set_defaults(run=_build)

    a = sub.add_parser("assemble", help="collect node tables into a grid file")
    a.add_argument("--couplings", default=None,
                   help="comma-separated coupling names of the grid (default: those the node tables record)")
    a.add_argument("--nodes", required=True, nargs="+", help="node table files; patterns are expanded")
    a.add_argument("--S-nodes", default=",".join(f"{s:g}" for s in DEFAULT_S_NODES))
    a.add_argument("--purpose", default="production", choices=("production", "example"))
    a.add_argument("--out", required=True)
    a.set_defaults(run=_assemble)

    i = sub.add_parser("info", help="print the header of a grid file")
    i.add_argument("grid")
    i.set_defaults(run=_info)

    args = parser.parse_args(argv)
    if getattr(args, "l_max", 0) is None:
        from transitionlistener.eft_rate import abelian_higgs_3d
        args.l_max = abelian_higgs_3d.MIXING_L_MAX
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
