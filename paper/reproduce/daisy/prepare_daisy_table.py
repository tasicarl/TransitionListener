"""The tables of the four-dimensional daisy-resummed rate as plane_scan.py reads them.

Source: the tables of build_daisy_table.py in the work folder, where the action of TransitionListener's
4D potential (one-loop Coleman-Weinberg with log(m^2(phi)/m^2(v)), Arnold-Espinosa daisy), tabulated
once per coupling by compute_daisy_actions.py, enters the false-vacuum fraction integrated on the plane
with the machinery of the NLO table.
  bounce  daisy_bounce_table.npz: the same one-dimensional solver set up as for the 3D actions
          (abelian_higgs_3d.solve_bounce_1d: false vacuum at the origin, a wall at 30 x the escape point), so
          that its tolerances are fractions of the bounce rather than of v; down to T/v = 1e-13,
          joined to the direct action above T/v = 1e-3. The default; variant N5 of the paper.
  direct  daisy_direct_table.npz: the action from TransitionListener's one-dimensional instanton
          solver (tunneling1D.SingleFieldInstanton) on the unsplined potential, down to T/v = 1e-8.
          Its tolerances are fractions of v, which biases S3/T low by 0.19 at T/v = 1e-7 and by
          about 1 at 1e-8. A diagnostic.
  rg      daisy_bounce_rg_table.npz: the bounce action of the potential with the couplings run to mu = pi T
          (models/TL_conformal_u1_no_fermions_rg.py), variant N4 of the paper.
  tl      daisy_table.npz: the action from TransitionListener's default path solver, which returns no
          action below T/v ~ 2.7e-4 g because its path extension steps over the barrier;
          its edge of percolation is set by that failure. For comparison only.
Nodes where the window ends while the rate still rises (window_limited = 1) are written as not
percolating (P_t_end = 0). For the direct table these lie at g <= 0.47, where the speed at the
end of the window is about 5 and falls towards smaller g, i.e. below the range of the template;
without this, transition_table would fill whole rows of them from the neighbouring coupling.

  python prepare_daisy_table.py [bounce|direct|tl|rg]  ->  paper/data/daisy_table_plane.npz /
                         daisy_table_plane_direct.npz / daisy_table_plane_tl.npz / daisy_rg_table_plane.npz
The variant name in every output is "daisy"."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402
kind = sys.argv[1] if len(sys.argv) > 1 else "bounce"
src = paths.work({"bounce": "daisy_bounce_table.npz", "direct": "daisy_direct_table.npz", "tl": "daisy_table.npz",
                          "rg": "daisy_bounce_rg_table.npz"}[kind])
out = paths.data({"bounce": "daisy_table_plane.npz", "direct": "daisy_table_plane_direct.npz",
                          "tl": "daisy_table_plane_tl.npz", "rg": "daisy_rg_table_plane.npz"}[kind])
D = dict(np.load(src, allow_pickle=True))
wl = D["window_limited"] == 1
D["P_t_end"] = np.where(wl, 0.0, D["P_t_end"])
D["VARIANTS"] = np.array(["daisy"])
np.savez(out, **D)
print(f"wrote {out}: {int(wl.sum())} window-limited nodes of {wl.size} set to no percolation; G {D['G'].min()}-{D['G'].max()}")
