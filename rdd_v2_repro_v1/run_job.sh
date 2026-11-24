#!/bin/bash
set -eo pipefail
repro_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
test -n "${PBS_JOBID:-}" || { printf 'Run through the submitted PBS allocation.\n' >&2; exit 2; }
cd "$repro_project/rdd_v2_repro_v1"
mkdir -p "$repro_project/runs/repro_job_logs"
repro_log="$repro_project/runs/repro_job_logs/$PBS_JOBID.log"
test ! -e "$repro_log" || { printf 'Job log already exists.\n' >&2; exit 2; }
exec > >(tee -a "$repro_log") 2>&1
exec 8>"$repro_project/runs/.rdd_v2_repro_execution.lock"
flock -n 8 || { printf 'Another reproducibility diagnostic is active.\n' >&2; exit 2; }
sha256sum -c SHA256SUMS
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
source "$repro_project/.venvs/rdd_detection_v1/bin/activate"
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export RDD_REPRO_DEADLINE=$(( $(date +%s) + 3000 ))
repro_output="$repro_project/runs/rdd_v2_repro_check_v1"
test ! -e "$repro_output" || { printf 'Diagnostic output already exists; inspect it.\n' >&2; exit 2; }
test ! -e "${repro_output}_records.tar.gz" || { printf 'Records archive already exists.\n' >&2; exit 2; }
python - <<'PY'
import os
import torch
import repro
repro.verify_package()
repro.pilot.runtime(repro.read(repro.pilot.PACKAGE/'protocol.json'))
print('PBS job:', os.environ['PBS_JOBID'])
print('Assigned GPU:', torch.cuda.get_device_name(0))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('Scope: 4 conditions x 7 actual Faster R-CNN updates; no accuracy evaluation.')
PY
python -m unittest discover -s tests -p 'test_repro.py' -v
repro_checkpoint="$repro_project/runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt"
set +e
python -u repro.py --root "$repro_project/dataset/raw" \
  --splits "$repro_project/dataset/detection_splits_v1" \
  --checkpoint "$repro_checkpoint" --out "$repro_output"
repro_exit=$?
set -e
if [[ "$repro_exit" == 3 ]]; then
    printf 'Diagnostic reached its bound. Partial records are preserved; inspect before any further job.\n'
elif [[ "$repro_exit" != 0 ]]; then
    printf 'Diagnostic failed. Preserve this log and inspect any returned failure records.\n' >&2
    exit "$repro_exit"
else
    printf 'DIAGNOSTIC JOB COMPLETE. Download and audit the records; this does not approve longer training.\n'
fi
