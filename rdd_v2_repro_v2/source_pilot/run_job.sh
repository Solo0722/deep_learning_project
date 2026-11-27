#!/bin/bash
set -eo pipefail
pilot_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
test -n "${PBS_JOBID:-}" || { printf 'Run inside the submitted PBS allocation.\n' >&2; exit 2; }
cd "$pilot_project/rdd_v2_pilot_v1"
mkdir -p "$pilot_project/runs/pilot_job_logs"
pilot_log="$pilot_project/runs/pilot_job_logs/$PBS_JOBID.log"
test ! -e "$pilot_log" || { printf 'Job log already exists.\n' >&2; exit 2; }
exec > >(tee -a "$pilot_log") 2>&1
exec 8>"$pilot_project/runs/.rdd_v2_pilot_execution.lock"
flock -n 8 || { printf 'Another pilot execution is active.\n' >&2; exit 2; }
sha256sum -c SHA256SUMS
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
source "$pilot_project/.venvs/rdd_detection_v1/bin/activate"
set -u
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export RDD_PILOT_DEADLINE=$(( $(date +%s) + 39600 ))
pilot_action="${RDD_ACTION:-start}"
case "$pilot_action" in start|resume) ;; *) printf 'Invalid RDD_ACTION.\n' >&2; exit 2 ;; esac
pilot_output="$pilot_project/runs/rdd_v2_pilot_seed42_v1"
test ! -e "$pilot_output/COMPLETE.json" || { printf 'Pilot already complete.\n' >&2; exit 2; }
if [[ "$pilot_action" == start ]]; then
    test ! -e "$pilot_output" || { printf 'Existing run; inspect before an explicit resume.\n' >&2; exit 2; }
else
    test -f "$pilot_output/context.json" || { printf 'No run context to resume.\n' >&2; exit 2; }
fi
python - <<'PY'
import json, os
from pathlib import Path
import torch
import pilot
pilot.verify_package()
pilot.runtime(json.loads(Path('protocol.json').read_text()))
print('PBS job:', os.environ['PBS_JOBID'])
print('Assigned GPU:', torch.cuda.get_device_name(0))
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
PY
python -m unittest discover -s tests -p 'test_*.py' -v
pilot_checkpoint="$pilot_project/runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt"
pilot_smoke="$pilot_project/runs/rdd_v2_pilot_smoke_$PBS_JOBID"
python -u smoke.py --root "$pilot_project/dataset/raw" --splits "$pilot_project/dataset/detection_splits_v1" \
  --checkpoint "$pilot_checkpoint" --out "$pilot_smoke"
export RDD_PILOT_SMOKE_RECORD="$pilot_smoke/PASS.json"
pilot_resume=()
if [[ "$pilot_action" == resume ]]; then pilot_resume=(--resume); fi
set +e
python -u pilot.py --root "$pilot_project/dataset/raw" --splits "$pilot_project/dataset/detection_splits_v1" \
  --checkpoint "$pilot_checkpoint" --out "$pilot_output" "${pilot_resume[@]}"
pilot_exit=$?
set -e
if [[ "$pilot_exit" == 3 ]]; then
    printf 'Pilot stopped at its time cap with checkpoints. Inspect status before an explicit resume. No automatic resubmission.\n'
elif [[ "$pilot_exit" != 0 ]]; then
    printf 'Pilot failed. Preserve the log and inspect the last valid checkpoint.\n' >&2
    exit "$pilot_exit"
else
    printf 'SHORT FOUR-ARM PILOT COMPLETE. Download and audit the records before any longer experiment.\n'
fi
