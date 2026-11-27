#!/bin/bash
set -euo pipefail
case "$(hostname)" in login*) ;; *) printf 'Submit from a CCAST login node.\n' >&2; exit 2 ;; esac
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/rdd_v2_preflight_v1
sha256sum -c SHA256SUMS
command -v qsub >/dev/null
exec 9>.submission.lock
flock -n 9 || { printf 'Another submission is in progress.\n' >&2; exit 2; }
test ! -e submission_start.id || { printf 'Already submitted: '; cat submission_start.id; exit 2; }
test ! -e ../runs/rdd_v2_geometry_check_v1 || { printf 'Results already exist; inspect them before resubmitting.\n' >&2; exit 2; }
test ! -e ../runs/rdd_v2_geometry_check_v1.tar.gz || { printf 'Archive already exists.\n' >&2; exit 2; }
preflight_job=$(qsub job.pbs)
printf '%s\n' "$preflight_job" > submission_start.id
printf 'Submitted CPU preflight PBS job: %s\n' "$preflight_job"
printf 'Live log: ../runs/preflight_job_logs/%s.log\n' "$preflight_job"
