#!/bin/bash
set -euo pipefail
pilot_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
pilot_job=$(cat "$pilot_project/rdd_v2_pilot_v1/submission_latest.id")
qstat "$pilot_job" || true
pilot_log="$pilot_project/runs/pilot_job_logs/$pilot_job.log"
if [ -f "$pilot_log" ]; then
    tail -n 80 "$pilot_log"
else
    printf 'No live log yet; the job may still be queued.\n'
fi
