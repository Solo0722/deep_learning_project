# CCAST runbook (PBS Pro)

`make_jobs.py` writes one `<run_id>.pbs` per run spec plus a `submit.sh`. Layout on the cluster, with
`P=/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research`:

| Path | Purpose |
|---|---|
| `$P/cua_code` | the code (this repo); every job `cd`s here |
| `$P/cua_data/<dataset>` | prepared datasets (`rdd`, `voc`, `dacl10k`) |
| `$P/cua_runs/<run_id>/` | real runs; `COMPLETE.json` appears only when a run has finished |
| `$P/cua_runs/_logs/<run_id>.log` | PBS output of each job |
| `$P/cua_runs_smoke/` | smoke runs (never mixed with real runs) |

## One-time setup

1. Put the code at `$P/cua_code` (git clone of branch `study/class-utility-augmentation`, or rsync).
2. Prepare the data on the cluster (`--out` is the dataset directory under `cua_data`):
   ```
   python -m cua.prepare rdd     --splits <splits> --images <image dir> --out $P/cua_data/rdd
   python -m cua.prepare voc     --devkit <VOCdevkit dir> --out $P/cua_data/voc
   python -m cua.prepare dacl10k --root <dacl10k root> --out $P/cua_data/dacl10k [--seed 0]
   ```
3. Check the venv `$P/.venvs/rdd_detection_v1` has torch 2.3, torchvision 0.18, `pycocotools` and pytest
   (`pip install -r cua/requirements-test.txt`).

## Generate jobs

```
python -m cua.ccast.make_jobs --family part_a --dataset rdd --detector fasterrcnn --out jobs/A-rdd-fasterrcnn
python -m cua.ccast.make_jobs --family budget --dataset rdd --detector fasterrcnn --out jobs/budget-rdd-fasterrcnn
python -m cua.ccast.make_jobs --family proxy  --dataset rdd --detector fasterrcnn --epochs 20 --edits blur,jpeg --out jobs/proxy
```
Options: `--epochs N`, `--edits e1,e2`, `--seeds 0,1,2`, `--smoke`, `--project P`. One (dataset, detector) pair per call.

Walltime: 6 h for `A-*` (deadline 20400 s), 24 h for `budget-*` and `proxy-*` (deadline 85200 s); the deadline is
walltime minus 20 minutes.

## Smoke first

Add `--smoke` (64 training images, 1 h walltime, deadline 2400 s, runs under `cua_runs_smoke`) and submit:
```
python -m cua.ccast.make_jobs --family part_a --dataset rdd --detector fasterrcnn --seeds 0 --smoke --out jobs/smoke
bash jobs/smoke/submit.sh
```
Smoke applies to the warm run and all arms alike (the trainer requires them to match). Check the logs; smoke output stays in
`cua_runs_smoke` and never touches `cua_runs`. Use a separate `--out` directory for smoke jobs (make_jobs refuses to mix them).

## Submit order

`bash <out>/submit.sh` does `mkdir -p` on the log directory, submits the warm (or budget/proxy) jobs first, then
each arm with `qsub -W depend=afterok:$WARM_s<seed>`. Non-arm jobs run `pytest -q -x -m "not slow" tests/cua` before
training and exit non-zero if it fails; arm jobs skip pytest because their warm job already ran it.

## Resume after a deadline stop (exit 3)

A job that reaches its deadline exits with code 3 and prints the exact command. Resubmit that same job:
```
qsub -v CUA_RESUME=1 /abs/path/to/<run_id>.pbs
```
Exit 3 is non-zero, so `afterok` arms of a warm run that stopped early never start. Operator procedure: resubmit the
warm with `CUA_RESUME=1`; when it is COMPLETE, resubmit its arms (`qsub <arm>.pbs`, or `qsub -W depend=afterok:<new warm job id>`).
Arms that were already held with a failed dependency can be removed with `qdel`.

## Checking progress

```
find $P/cua_runs -name COMPLETE.json | wc -l
```
Expected COMPLETE.json counts:
- Part A: 129 per detector (3 seeds x (warm + base + valid edits): 42 rdd + 42 voc + 45 dacl10k), 258 for both.
- Budget: 3 per detector (one per dataset).
- Proxy: 1 + the number of chosen edits (the base run plus one per edit).

 Logs: `tail $P/cua_runs/_logs/<run_id>.log`.
