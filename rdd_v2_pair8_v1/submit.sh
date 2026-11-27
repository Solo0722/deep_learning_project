#!/bin/bash
set -euo pipefail
pilot_action="${1:-start}"
case "$pilot_action" in start|resume) ;; *) printf 'Use start or resume.\n' >&2; exit 2 ;; esac
case "$(hostname)" in login*) ;; *) printf 'Submit from a CCAST login node.\n' >&2; exit 2 ;; esac
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_pair8_v1
sha256sum -c SHA256SUMS
command -v qsub >/dev/null
exec 9>.submission.lock
flock -n 9 || { printf 'Another submission is in progress.\n' >&2; exit 2; }
if [ -f submission_latest.id ]; then
    pilot_previous=$(cat submission_latest.id)
    if qstat "$pilot_previous" >/dev/null 2>&1; then
        printf 'Previous pilot job still exists: %s. Inspect it instead of duplicating it.\n' "$pilot_previous" >&2
        exit 2
    fi
fi
pilot_output=../runs/rdd_v2_pair8_seed42_v1
test ! -e "$pilot_output/COMPLETE.json" || { printf 'Pilot already complete.\n' >&2; exit 2; }
if [[ "$pilot_action" == start ]]; then
    test ! -e submission_start.id || { printf 'Already submitted: '; cat submission_start.id; exit 2; }
    test ! -e "$pilot_output" || { printf 'Run already exists.\n' >&2; exit 2; }
else
    test -f "$pilot_output/context.json" || { printf 'No run context to resume.\n' >&2; exit 2; }
fi
pilot_checkpoint=../runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt
printf '%s  %s\n' 'dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858' "$pilot_checkpoint" | sha256sum -c -
pilot_job=$(qsub -v "RDD_ACTION=$pilot_action" job.pbs)
if [[ "$pilot_action" == start ]]; then pilot_idfile=submission_start.id; else pilot_idfile="submission_resume_$pilot_job.id"; fi
printf '%s\n' "$pilot_job" > "$pilot_idfile"
printf '%s\n' "$pilot_job" > submission_latest.id
printf 'Submitted two-arm epoch-8 comparison PBS job: %s\n' "$pilot_job"
printf 'Live log: ../runs/pair8_job_logs/%s.log\n' "$pilot_job"
