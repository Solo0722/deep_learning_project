#!/bin/bash
set -eo pipefail
action="${1:-start}"
case "$action" in start|resume) ;; *) printf 'Use start or resume.\n' >&2; exit 2 ;; esac
project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
package="$project/fasterrcnn_shared_v1"
result="$project/runs/fasterrcnn_shared_seed42_v1"
smoke="$project/runs/fasterrcnn_shared_smoke_v1"
test -n "${PBS_JOBID:-}" || { printf 'Use qsub; do not run on a login node.\n' >&2; exit 2; }
cd "$package"
mkdir -p "$project/runs/baseline_job_logs"
log="$project/runs/baseline_job_logs/${PBS_JOBID}.log"
printf 'Live log: %s\n' "$log"
exec > >(tee "$log") 2>&1
sha256sum -c SHA256SUMS
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
source "$project/.venvs/rdd_detection_v1/bin/activate"
set -u
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
python -m unittest discover -s tests -p 'test_*.py' -v
common=(--root "$project/dataset/raw" --splits "$project/dataset/detection_splits_v1" --weights "$project/weights/detection_v1")
if [[ "$action" == start ]]; then
    test ! -e "$result" || { printf 'Full run already exists; inspect it before resuming.\n' >&2; exit 2; }
    test ! -e "$smoke" || { printf 'Smoke output already exists; inspect it before retrying.\n' >&2; exit 2; }
    python -u baseline.py "${common[@]}" --out "$smoke" --smoke --max-seconds 1800
    python - "$smoke/run.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1]))
assert record['status'] == 'complete' and record['profile'] == 'smoke'
assert record['checkpoint_prediction_replay'] == 'exact'
assert record['augmentation_operation_checks']['status'] == 'passed'
assert len(record['augmentation_operation_checks']['operations']) == 8
assert record['augmentation_applied_images'] > 0
print('Shared-augmentation GPU data/train/validation/checkpoint gate: PASS')
PY
    python -u baseline.py "${common[@]}" --out "$result" --max-seconds 81000
else
    python -u baseline.py "${common[@]}" --out "$result" --resume --max-seconds 82800
fi
