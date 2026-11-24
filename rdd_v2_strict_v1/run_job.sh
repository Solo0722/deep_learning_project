#!/bin/bash
set -eo pipefail
strict_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
test -n "${PBS_JOBID:-}" || { printf 'Run through the PBS allocation.\n' >&2; exit 2; }
cd "$strict_project/rdd_v2_strict_v1"
mkdir -p "$strict_project/runs/strict_job_logs"
strict_log="$strict_project/runs/strict_job_logs/$PBS_JOBID.log"
test ! -e "$strict_log" || { printf 'Job log already exists.\n' >&2; exit 2; }
exec > >(tee -a "$strict_log") 2>&1
exec 8>"$strict_project/runs/.rdd_v2_strict_execution.lock"
flock -n 8 || { printf 'Another strict check is active.\n' >&2; exit 2; }
sha256sum -c SHA256SUMS
source /mmfs1/apps/pyenvs/anaconda3-2022.05/etc/profile.d/conda.sh
conda activate pytorch
source "$strict_project/.venvs/rdd_detection_v1/bin/activate"
set -u
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export RDD_STRICT_DEADLINE=$(( $(date +%s) + 3000 ))
strict_output="$strict_project/runs/rdd_v2_strict_check_v1"
test ! -e "$strict_output" || { printf 'Output already exists; inspect it.\n' >&2; exit 2; }
test ! -e "${strict_output}_records.tar.gz" || { printf 'Records archive already exists.\n' >&2; exit 2; }
printf 'Running implementation tests. Diagnostic lines inside these tests use synthetic fixtures.\n'
python -m unittest discover -s source_repro/tests -p test_repro.py -v
python -m unittest discover -s tests -p test_strict.py -v
printf 'Implementation tests passed. Starting the real-image strict CUDA diagnostic.\n'
strict_checkpoint="$strict_project/runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt"
set +e
python -u strict_check.py --root "$strict_project/dataset/raw" \
  --splits "$strict_project/dataset/detection_splits_v1" \
  --checkpoint "$strict_checkpoint" --out "$strict_output"
strict_exit=$?
set -e
if [[ "$strict_exit" != 0 ]]; then
    printf 'Strict check stopped without a full pass. Download any records and inspect the status.\n' >&2
    exit "$strict_exit"
fi
printf 'STRICT CHECK COMPLETE. Review exactness, runtime and memory before longer training.\n'
