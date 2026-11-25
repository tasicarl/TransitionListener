#!/bin/bash
# The binary foreground on the plane: B1 and B2 marginalised (binaries/marginal_foreground.py), IPTA with the population
# (binaries/ipta_fog.py, ipta_fog_priors.py), NANOGrav under the three prior sets (binaries/prior_study.py, copied to
# paper/data by collect_prior_study.py), the population priors against the default (binaries/bhb_population_dlnl.py),
# and the preference for the phase transition (binaries/preference.py at tau_f >= 0.1 and 4 Gyr; preference_map.py;
# preference_tau_profile.py for figure 5). Several hours on 56 cores.
set -euo pipefail
source "$(dirname "$0")/env.sh"
B=$REPRO/binaries
WORKERS=$NCPU "$PY" "$B/marginal_foreground.py" 32768 16384
WORKERS=$NCPU "$PY" "$B/ipta_fog.py" 32768
WORKERS=$NCPU PRIORS=literature,recent "$PY" "$B/ipta_fog_priors.py" 32768
WORKERS=$NCPU PRIORS=nanograv,literature,recent "$PY" "$B/prior_study.py"
"$PY" "$B/collect_prior_study.py"
"$PY" "$B/bhb_population_dlnl.py"
WORKERS=$NCPU TAU_MIN=0.1 OUT=preference_tau0.1.json "$PY" "$B/preference.py"
WORKERS=$NCPU TAU_MIN=4 OUT=preference_tau4.json "$PY" "$B/preference.py"
WORKERS=$NCPU "$PY" "$B/preference_map.py"
WORKERS=$NCPU "$PY" "$B/preference_tau_profile.py"
