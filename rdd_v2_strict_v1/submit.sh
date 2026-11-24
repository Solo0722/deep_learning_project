#!/bin/bash
set -euo pipefail
case "$(hostname)" in login*) ;; *) printf 'Submit from a CCAST login node.\n' >&2; exit 2 ;; esac
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_strict_v1
sha256sum -c SHA256SUMS
command -v qsub >/dev/null
exec 9>.submission.lock
flock -n 9 || { printf 'Another submission is in progress.\n' >&2; exit 2; }
if [ -f submission_start.id ]; then
    printf 'Already submitted: '
    cat submission_start.id
    printf 'Inspect the existing job; do not submit a duplicate.\n' >&2
    exit 2
fi
test ! -e ../runs/rdd_v2_strict_check_v1 || { printf 'Existing strict output.\n' >&2; exit 2; }
test ! -e ../runs/rdd_v2_strict_check_v1_records.tar.gz || { printf 'Existing strict records.\n' >&2; exit 2; }
strict_checkpoint=../runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt
printf '%s  %s\n' 'dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858' "$strict_checkpoint" | sha256sum -c -
strict_job=$(qsub job.pbs)
printf '%s\n' "$strict_job" > submission_start.id
printf 'Submitted strict-determinism PBS job: %s\n' "$strict_job"
printf 'Live log: ../runs/strict_job_logs/%s.log\n' "$strict_job"
