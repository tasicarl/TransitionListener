"""The no-signal values of the maps of tail_fog.npz, added to it as plateau_<key>: the likelihood with the transition
spectrum zero, which is what every cell where the amplitude bracket vanishes ((beta/H)_Gamma <= 1.52, no spectrum)
has. tail_fog.py leaves these cells empty; the figures fill them with this value inside the completion margin, as
make_figures._display_map does on the coarse grid. The draws are those of tail_fog.py (same generator and order).
    python tail_fog_plateau.py"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.argv = sys.argv[:1]
sys.path.insert(0, os.path.dirname(HERE))
import paths  # noqa: E402  (src/, models/ and the folders of paper/reproduce on sys.path)
import plane_scan as PS  # noqa: E402
import marginal_foreground as mf  # noqa: E402
import prior_study as pst  # noqa: E402

path = paths.data("tail_check", "tail_fog.npz")
out = dict(np.load(path))
zero = np.zeros((1, mf.NB))
lme = lambda l: float(l.max() + np.log(np.mean(np.exp(l - l.max()))))
rng = np.random.default_rng(20261108)                         # as tail_fog.py
th_h = mf.prior_draws_holodeck(rng, 32768)
out["plateau_lnL"] = float(mf.LIKE.ln_like(zero[0]))
out["plateau_lnL_holodeck"] = lme(np.array([mf.lnl_holodeck(zero, x)[0] for x in th_h]))
mu, cov = PS.signal_builder.bhb_priors["NG15"]
th_p = rng.multivariate_normal(np.array(mu, float), np.array(cov, float), size=16384)
out["plateau_lnL_pl"] = lme(np.array([mf.lnl_powerlaw(zero, x)[0] for x in th_p]))
l0 = np.array([mf.lnl_holodeck(zero, x)[0] for x in th_h[:4096]])
out["plateau_lnL_fog_prof"] = float(pst.profile_one(None, "nanograv", list(th_h[np.argsort(-l0)[:8]]))[0])
np.savez(path, **out)
print({k: round(v, 4) for k, v in out.items() if k.startswith("plateau_")})
