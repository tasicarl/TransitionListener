#!/bin/bash
# The numbers quoted in the text (reproduce/numbers/*.py) from paper/data, each printout into
# reproduce/numbers/output/<script>.txt. Seconds to minutes; speed_public.py (the timings of appendix B) is not run
# here, since it needs a checkout of the public TransitionListener release (see its docstring).
set -euo pipefail
source "$(dirname "$0")/env.sh"
N=$REPRO/numbers; mkdir -p "$N/output"
for s in line_text_numbers line_daisy_numbers preference_profile_numbers text_numbers plane_numbers slim_region_numbers region_extra_numbers extra_rows_hpv \
         eft_validity_regions prior_range_numbers table_preference make_numbers determinant_reference_values; do
  "$PY" "$N/$s.py" > "$N/output/$s.txt"
  echo "$s done"
done
