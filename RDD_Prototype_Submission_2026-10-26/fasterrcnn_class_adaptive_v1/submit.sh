#!/bin/bash
set -euo pipefail
action="${1:-start}"
case "$action" in start|resume) ;; *) printf 'Use start or resume.\n' >&2; exit 2 ;; esac
case "$(hostname)" in login*) ;; *) printf 'Submit from a CCAST login node.\n' >&2; exit 2 ;; esac
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/fasterrcnn_class_adaptive_v1
sha256sum -c SHA256SUMS
command -v qsub >/dev/null
exec 9>.submission.lock
flock -n 9 || { printf 'Another submission is in progress.\n' >&2; exit 2; }
if [[ "$action" == start ]]; then
    test ! -e submission_start.id || { printf 'Already submitted: '; cat submission_start.id; exit 2; }
    test ! -e ../runs/fasterrcnn_class_adaptive_seed42_v1 || { printf 'Run already exists.\n' >&2; exit 2; }
    job=$(qsub -v RDD_ACTION=start job.pbs)
    printf '%s\n' "$job" > submission_start.id
else
    test -f ../runs/fasterrcnn_class_adaptive_seed42_v1/last.json || { printf 'No checkpoint to resume.\n' >&2; exit 2; }
    # Resumption is an explicit command, not an automatic retry or a longer epoch budget.
    job=$(qsub -v RDD_ACTION=resume job.pbs)
    printf '%s\n' "$job" > "submission_resume_${job}.id"
fi
printf 'Submitted PBS job: %s\n' "$job"
printf 'Live log: ../runs/baseline_job_logs/%s.log\n' "$job"
