#!/bin/bash
set -euo pipefail
case "$(hostname)" in login*) ;; *) printf 'Submit from a CCAST login node.\n' >&2; exit 2 ;; esac
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_controller_check_v1
sha256sum -c SHA256SUMS
command -v qsub >/dev/null
exec 9>.submission.lock
flock -n 9 || { printf 'Another submission is in progress.\n' >&2; exit 2; }
test ! -e submission_start.id || { printf 'Already submitted: '; cat submission_start.id; exit 2; }
test ! -e ../runs/rdd_v2_controller_check_v1 || { printf 'Results already exist; inspect them first.\n' >&2; exit 2; }
test ! -e ../runs/rdd_v2_controller_check_v1.tar.gz || { printf 'Archive already exists.\n' >&2; exit 2; }
controller_checkpoint=../runs/fasterrcnn_identity_seed42_v1/checkpoints/epoch_001_step_00006716_04dccd11bbd2.pt
test -f "$controller_checkpoint" || { printf 'The pinned epoch-1 checkpoint is missing. Stop; do not substitute a different checkpoint.\n' >&2; exit 2; }
printf '%s  %s\n' 'dbd619334a57ad4422e929b90b327f80d1ca3bfc5b86f6f37594714ec138e858' "$controller_checkpoint" | sha256sum -c -
controller_job=$(qsub job.pbs)
printf '%s\n' "$controller_job" > submission_start.id
printf 'Submitted GPU controller-check PBS job: %s\n' "$controller_job"
printf 'Live log: ../runs/controller_job_logs/%s.log\n' "$controller_job"
