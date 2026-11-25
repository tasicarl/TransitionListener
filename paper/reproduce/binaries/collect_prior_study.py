"""Copies the parts of prior_study.py's output (prior_study_<prior>.npz in the work folder) that the figures and tables
read into paper/data:
    ng15_fog_priors.npz       lnL_marg_<prior>: NANOGrav with the binaryfog population marginalised over the
                              Tomczak-Kormendy ('literature') and survey-averaged ('recent') priors
    ng15_fog_prof_recent.npz  prior_study_recent.npz unchanged (the population profiled under the survey-averaged priors)
    python collect_prior_study.py"""
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paths  # noqa: E402

out = {}
for prior in ("literature", "recent"):
    q = np.load(paths.work(f"prior_study_{prior}.npz"))
    out["G"], out["LOGV"], out["lnL_marg_" + prior] = q["G"], q["LOGV"], q["lnL_marg"]
np.savez(paths.data("ng15_fog_priors.npz"), **out)
shutil.copyfile(paths.work("prior_study_recent.npz"), paths.data("ng15_fog_prof_recent.npz"))
print("wrote ng15_fog_priors.npz, ng15_fog_prof_recent.npz")
