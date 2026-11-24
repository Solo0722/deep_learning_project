#!/bin/bash
set -eo pipefail
controller_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
test -n "${PBS_JOBID:-}" || { printf 'Run inside a PBS allocation.\n' >&2; exit 2; }
cd "$controller_project/rdd_v2_controller_check_v1"
mkdir -p "$controller_project/runs/controller_job_logs"
controller_log="$controller_project/runs/controller_job_logs/$PBS_JOBID.log"
test ! -e "$controller_log" || { printf 'Existing job log; refusing replacement.\n' >&2; exit 2; }
exec > >(tee -a "$controller_log") 2>&1
sha256sum -c SHA256SUMS
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
source "$controller_project/.venvs/rdd_detection_v1/bin/activate"
set -u
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python - <<'PY'
import os, sys
from pathlib import Path
import torch
expected = Path('/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/.venvs/rdd_detection_v1')
assert Path(sys.prefix).resolve() == expected.resolve(), 'Wrong environment'
assert os.environ.get('PBS_JOBID'), 'Missing PBS allocation'
assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'Expected one assigned CUDA GPU'
print('PBS job:', os.environ['PBS_JOBID'])
print('Assigned GPU:', torch.cuda.get_device_name(0))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
PY
python -m unittest discover -s tests -p test_diagnostic.py -v
python -u run_check.py \
  --root "$controller_project/dataset/raw" \
  --splits "$controller_project/dataset/detection_splits_v1" \
  --checkpoint "$controller_project/runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt" \
  --out "$controller_project/runs/rdd_v2_controller_check_v1"
printf 'CONTROLLER CHECK JOB COMPLETE. Review results before any training.\n'
