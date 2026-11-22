#!/bin/bash
# Submit a whole per-class PBT chain up front:
#   bash $P/cua_code/cua/ccast/submit_pbt.sh PBT-rdd-fasterrcnn-s0-pc [N_WINDOWS] [MEMBERS]
# Window k: MEMBERS member jobs (afterok on select k-1), then one select job (afterok on all of them).
# Steps already finished (ref_metrics.json / decisions.jsonl row present) are skipped, so the same
# command resumes a chain after a failure or a deadline stop.
set -euo pipefail
P=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
ID=$1; NW=${2:-10}; NM=${3:-4}
PBS=$P/cua_code/cua/ccast/pbt_step.pbs
LOG=$P/cua_runs/_logs/pbt; mkdir -p $LOG
ROOT=$P/cua_runs/$ID
prev=""
for k in $(seq 1 $NW); do
    if [ -f $ROOT/decisions.jsonl ] && grep -q "\"window\": $k," $ROOT/decisions.jsonl; then
        echo "window $k: already selected"; prev=""; continue
    fi
    dep=${prev:+-W depend=afterok:$prev}
    mem=""
    for j in $(seq 0 $((NM-1))); do
        if [ -f $ROOT/w$k-m$j/ref_metrics.json ]; then echo "w$k-m$j: done"; continue; fi
        jid=$(qsub $dep -N pbt_w${k}m$j -o $LOG/$ID-w$k-m$j.log -v STEP=member,PBT_ID=$ID,WINDOW=$k,MEMBER=$j $PBS)
        echo "w$k-m$j: $jid"; mem=${mem:+$mem:}$jid
    done
    sdep=${mem:+-W depend=afterok:$mem}
    [ -z "$sdep" ] && sdep=$dep
    prev=$(qsub $sdep -l walltime=00:30:00 -N pbt_sel$k -o $LOG/$ID-select$k.log -v STEP=select,PBT_ID=$ID,WINDOW=$k $PBS)
    echo "select $k: $prev"
done
