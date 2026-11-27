#!/bin/bash
set -euo pipefail
repro_project=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
repro_job=$(cat "$repro_project/rdd_v2_repro_v2/submission_start.id")
qstat "$repro_job" || true
repro_log="$repro_project/runs/repro_job_logs/$repro_job.log"
if [ -f "$repro_log" ]; then
    tail -n 80 "$repro_log"
else
    printf 'No live log yet; the job may still be queued.\n'
fi
