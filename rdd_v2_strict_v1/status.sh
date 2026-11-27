#!/bin/bash
set -euo pipefail
strict_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
strict_job=$(cat "$strict_project/rdd_v2_strict_v1/submission_start.id")
qstat "$strict_job" || true
strict_log="$strict_project/runs/strict_job_logs/$strict_job.log"
if [ -f "$strict_log" ]; then
    tail -n 80 "$strict_log"
else
    printf 'No live log yet; the job may still be queued.\n'
fi
