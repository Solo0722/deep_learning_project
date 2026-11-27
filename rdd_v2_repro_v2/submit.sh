#!/bin/bash
set -euo pipefail
case "$(hostname)" in login*) ;; *) printf 'Submit from a CCAST login node.\n' >&2; exit 2 ;; esac
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_repro_v2
sha256sum -c SHA256SUMS
command -v qsub >/dev/null
exec 9>.submission.lock
flock -n 9 || { printf 'Another submission is in progress.\n' >&2; exit 2; }
if [ -f submission_start.id ]; then
    printf 'Diagnostic already submitted: '
    cat submission_start.id
    printf 'Inspect status and records; do not submit a duplicate.\n' >&2
    exit 2
fi
test ! -e ../runs/rdd_v2_repro_check_v2 || { printf 'Existing diagnostic output.\n' >&2; exit 2; }
test ! -e ../runs/rdd_v2_repro_check_v2_records.tar.gz || { printf 'Existing records archive.\n' >&2; exit 2; }
repro_checkpoint=../runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt
printf '%s  %s\n' 'dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858' "$repro_checkpoint" | sha256sum -c -
repro_job=$(qsub job.pbs)
printf '%s\n' "$repro_job" > submission_start.id
printf 'Submitted reproducibility diagnostic PBS job: %s\n' "$repro_job"
printf 'Live log: ../runs/repro_job_logs/%s.log\n' "$repro_job"
