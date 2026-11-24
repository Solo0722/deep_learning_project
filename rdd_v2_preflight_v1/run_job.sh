#!/bin/bash
set -eo pipefail
preflight_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
test -n "${PBS_JOBID:-}" || { printf 'Run inside a PBS allocation.\n' >&2; exit 2; }
cd "$preflight_project/rdd_v2_preflight_v1"
mkdir -p "$preflight_project/runs/preflight_job_logs"
preflight_log="$preflight_project/runs/preflight_job_logs/$PBS_JOBID.log"
test ! -e "$preflight_log" || { printf 'Existing job log; refusing replacement.\n' >&2; exit 2; }
exec > >(tee -a "$preflight_log") 2>&1
sha256sum -c SHA256SUMS
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
source "$preflight_project/.venvs/rdd_detection_v1/bin/activate"
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python - <<'PY'
import os, sys
from pathlib import Path
expected = Path('/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/.venvs/rdd_detection_v1')
if Path(sys.prefix).resolve() != expected.resolve():
    raise SystemExit('STOP: wrong Python environment')
print('PBS job:', os.environ['PBS_JOBID'])
print('Python:', sys.version)
print('CPU-only preflight: no detector inference or training.')
PY
python -m unittest discover -s tests -p test_preflight.py -v
python -u geometry_check.py \
  --root "$preflight_project/dataset/raw" \
  --splits "$preflight_project/dataset/detection_splits_v1" \
  --out "$preflight_project/runs/rdd_v2_geometry_check_v1" \
  --require-target-versions
printf 'PREFLIGHT JOB COMPLETE. Inspect the returned records and overlays before any training.\n'
