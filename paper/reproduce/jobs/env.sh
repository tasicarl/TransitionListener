# Settings shared by the stage-B drivers; every machine-specific value comes from the environment.
#   PY          the Python interpreter of an environment with TransitionListener's dependencies, PTArcade/ceffyl and
#               binaryfog 1.0.0 (default: python)
#   NCPU        processes per step (default: the CPUs of the machine)
#   TL_CODE_ROOT  a checkout of this repository whose src/ and models/ are used (default: this checkout)
#   PAPER_WORK  the folder of the intermediate results (default: paper/reproduce/work)
#   LINESCAN_POINTS  the line-scan points of 01_linescan.sh, one folder per coupling (default: <PAPER_WORK>/linescan_points)
#   DAISY_CACHE  the actions of 05_daisy.sh, one file per coupling and kind (default: <PAPER_WORK>/daisy_cache)
# Under SLURM, submit a driver with sbatch --cpus-per-task=N and it takes NCPU from SLURM_CPUS_PER_TASK.
: "${PY:=python}"
: "${NCPU:=${SLURM_CPUS_PER_TASK:-$(nproc)}}"
REPRO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO=$(cd "$REPRO/../.." && pwd)
export PYTHONPATH="${TL_CODE_ROOT:-$REPO}/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
: "${LINESCAN_POINTS:=${PAPER_WORK:-$REPRO/work}/linescan_points}"
: "${DAISY_CACHE:=${PAPER_WORK:-$REPRO/work}/daisy_cache}"
export PY NCPU REPRO REPO LINESCAN_POINTS DAISY_CACHE
