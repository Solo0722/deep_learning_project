# Operation and magnitude pilot: CCAST instructions

This is a diagnostic experiment for designing the adaptive controller. It tests **which operation, at which magnitude, for which class, at which training stage** helps subsequent learning. It does not yet implement an online policy or establish a novel algorithm. The manuscript and completed conventional baseline are unchanged.

## What the pilot measures

Start two training branches from the same saved CNN, AdamW moments, scheduler, and batch-normalization state. Give them the same training examples, conventional augmentation, and random stream. The reference branch uses conventional augmentation alone. The candidate branch adds one operation to examples of one target class. Evaluate both branches after 1, 10, and 50 optimizer updates. Restart from the checkpoint for every candidate; never train candidates sequentially into one model.

The pilot asks whether inexpensive feedback predicts the measured benefit of these interventions. It records correctness, conditional correctness retention, excess training loss, gradient alignment, an AdamW displacement approximation, and the measured one-step reference improvement. Longer-horizon measurements on separate development images are the outcomes. Model parameters, batch-normalization buffers, and optimizer moments evolve normally during branches; the checkpoint learning rate is held constant for this local experiment.

Positive `*_balanced_loss_gain` means lower class-balanced cross-entropy than the paired reference. Positive `*_macro_f1_gain` means higher macro-F1. Per-class outcomes reveal whether helping the target class harms another class. The AdamW linear score accounts for the actual first parameter displacement, but omits batch-normalization buffer effects and higher-order terms; its usefulness must be measured rather than assumed.

## Strength is part of this pilot

| Extra operation | Low magnitude | High magnitude | Meaning |
|---|---:|---:|---|
| Identity | 0 | — | No extra transformation; conventional augmentation remains |
| Brightness | 0.10 | 0.25 | Multiply brightness by 1 ± magnitude |
| Contrast | 0.10 | 0.25 | Multiply contrast by 1 ± magnitude |
| Gaussian blur | 0.5 | 1.0 | Radius in pixels of the resized content |
| Horizontal translation | 0.03 | 0.08 | Fraction of content width, rounded to pixels |
| Vertical translation | 0.03 | 0.08 | Fraction of content height, rounded to pixels |

This makes 11 candidates including identity. For signed operations, the sign is sampled with equal probability and shared across candidate comparisons. One extra operation is applied to every target-class example; operation count and application probability stay fixed. The shared conventional augmentation happens first. The letterbox padding stays unchanged. Translations use gray fill, clip at the content boundary, and never wrap. A small displacement can round to zero on a thin crop; the runner records how many target examples actually change.

These are provisional diagnostic ranges, not validated safe limits, equal-severity scales across operations, or the final search space. The two levels test whether strength matters; they cannot locate an optimum or identify a continuous response curve. More levels can follow after the initial measurements and image review. Do not add rotations that can change the longitudinal/transverse interpretation without resolving the label convention.

CUDA uses a class-level strength variable that controls both the number of operations and their magnitude, with correctness-based level updates; see [CUDA, Section 3.2](https://arxiv.org/html/2302.05499). Here, magnitude is varied separately for each operation so that its effect can be isolated. Our candidate direction is to select operation–magnitude pairs using estimated benefit and uncertainty. Neither that broad idea nor a gradient-based signal is a novelty claim. CUDA remains a relevant comparator; this runner is not a reproduction of CUDA.

## Data and experimental limits

- The existing split remains fixed: 38,513 training crops, 8,242 validation crops, and 8,247 test records. Test pixels are never opened. Manifest integrity checks still examine test metadata to detect cross-split relationships.
- All original validation data were already used in conventional-baseline development. Both new roles are **development data**, not an untouched final evaluation.
- `design/development_membership.csv` assigns every validation crop to a reference or audit role. Known source-image, group, and source-file-hash links stay within a role, including transitive links.
- Each selected panel contains 128 examples per class, 512 total. Selection prefers distinct groups and spreads examples across available sources. This is a diagnostic panel, not an estimate weighted to deployment prevalence. Source balancing and recorded grouping do not establish independence of unrecorded road sequences or locations.
- Only the reference panel supplies the feedback gradient and the one-step reference score. Audit outcomes measure feedback performance. If audit findings are used to redesign the method, they become development evidence; they cannot then serve as independent confirmation.
- Ordinary shuffled training produces checkpoints. Diagnostic branches use equal class counts per minibatch to give each target-class intervention comparable exposure. Effects therefore apply to that diagnostic sampling regime. They do not yet show an improvement under ordinary full-run sampling.
- Initial trajectory: 20 planned epochs, snapshots at 2, 10, and 18, seed 42. This retains a familiar baseline recipe and provides early/middle/late states; it is not an optimized final protocol. A new run is needed because the existing baseline lacks these early full-state snapshots. Existing baseline checkpoints are deliberately rejected by this runner.
- Two episodes in the full pilot vary batch/random streams at the same checkpoint. They are dependent development measurements, **not two independent trained models**. One trajectory cannot establish seed robustness, statistical significance, or a publishable improvement. Subsequent independent trajectories and an appropriately planned final comparison depend on the pilot's observed variability and effect sizes.

## Files prepared locally

- `pilot_runner.py`: prepare, preview, train, identity, and probe commands.
- `tests/test_pilot_runner.py`: synthetic scientific-invariant tests.
- `reports/pilot_design_v1/`: frozen design and complete development membership.
- `reports/pilot_previews_local_v1/`: 35 preview sheets across seven sources and five operations, using 56 distinct training examples.
- `reports/pilot_verification_v1/`: CPU identity report and test log.
- `pilot_ccast_v1.tar.gz`: upload package, including the unchanged baseline module and preview helper, frozen design, tests, this runbook, and file checksums. No datasets or checkpoints are bundled.

Local previews are explicitly marked **LOSSY JPEG CONTACT-SHEET PROXIES**. The original crop PNGs are on CCAST. The proxies are useful for visual screening but cannot establish exact PNG appearance or be used for training. The CCAST preview command below uses checksum-verified original training PNGs.

Visual inspection of five representative sheets found very thin content strips, faint cracks, and low-resolution defects. Blur can reduce visible crack detail; translation removes boundary content. Some low translations appear unchanged after rounding. These findings do not certify label preservation or justify excluding examples based on model performance. Record suspect operation/magnitude/image combinations during original-PNG review; any revised bank needs a new design and run directory.

## Step 1 — Upload from Windows PowerShell

The following commands use the project's established CCAST paths. They were prepared locally; no remote job has been submitted.

```powershell
Set-Location 'C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project'
scp .\pilot_ccast_v1.tar.gz solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
ssh solomon.owusu.1@prime.ccast.ndsu.edu
```

## Step 2 — Unpack once on the CCAST login node

```bash
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
mkdir pilot_ccast_v1 && tar -xzf pilot_ccast_v1.tar.gz -C pilot_ccast_v1
cd pilot_ccast_v1
sha256sum -c SHA256SUMS
```

Every listed file must report `OK`. If the directory already exists from this same upload, skip unpacking and verify its checksums. If checksums differ, investigate before running; do not overwrite an experiment's source files with a different version.

## Step 3 — Request a GPU and activate the existing environment

Request this only from the login node, not inside an existing allocation:

```bash
interact -c 4 -g 1 -m 16 -t 2
```

Once allocated:

```bash
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/pilot_ccast_v1
source /mmfs1/apps/pyenvs/anaconda3-2022.05/bin/activate pytorch
python -c "import torch, PIL, numpy; print('torch', torch.__version__, 'Pillow', PIL.__version__, 'numpy', numpy.__version__); assert torch.cuda.is_available() and torch.cuda.device_count() == 1; print(torch.cuda.get_device_name(0))"
python -m unittest discover -s tests -p test_pilot_runner.py -v
```

The tests run on CPU even in this allocation; all six must pass. They check software invariants with synthetic images. The real-data GPU identity check is a separate step below. The package needs Python 3.9+, PyTorch 2.3+, NumPy, and Pillow. Use the existing working environment; do not upgrade packages mid-experiment. The runner records library versions and refuses an incompatible resume configuration.

## Step 4 — Generate previews from original PNGs

```bash
python -u pilot_runner.py preview --data ../dataset/processed_v1 --design design --out runs/previews_png_v1
```

Expected: `PREVIEWS COMPLETE | ORIGINAL CHECKSUM-VERIFIED TRAINING PNGS | sheets: 35`.

To inspect them locally, open a **second Windows PowerShell terminal** and run:

```powershell
Set-Location 'C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project'
scp -r solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/pilot_ccast_v1/runs/previews_png_v1 .\reports\
```

Review whether the defect remains visible and its class meaning remains credible under both strengths and signs. The pilot bank can include harmful conditions as diagnostics, but those conditions must not silently become admissible controller actions. Neither high model confidence nor a favorable pilot score certifies label preservation.

## Step 5 — Train a fresh checkpoint trajectory

Back in the allocated CCAST terminal:

```bash
python -u pilot_runner.py train --data ../dataset/processed_v1 --design design --out runs/trajectory_seed42_v1 --device cuda --epochs 20 --snapshots 2,10,18 --batch-size 64 --eval-batch-size 64 --workers 2 --threads 2 --seed 42 --max-minutes 75
```

Expected final status: `TRAJECTORY COMPLETE`. The run saves `epoch_002.pt`, `epoch_010.pt`, `epoch_018.pt`, and `last.pt`. This is separate from the completed baseline. Snapshot epochs are predetermined; no best-checkpoint selection uses audit outcomes.

If the runner prints `TRAJECTORY PAUSED`, repeat the following command in a GPU allocation with sufficient time left:

```bash
python -u pilot_runner.py train --data ../dataset/processed_v1 --design design --out runs/trajectory_seed42_v1 --device cuda --epochs 20 --snapshots 2,10,18 --batch-size 64 --eval-batch-size 64 --workers 2 --threads 2 --seed 42 --max-minutes 75 --resume
```

The 75-minute budget is checked at epoch boundaries; it is not a strict scheduler deadline. A killed run resumes from its last complete epoch. If `last.pt` does not exist, the first epoch never completed: use a fresh output directory and update subsequent paths. Do not change total epochs on resume, because that changes the scheduler. After an allocation expires, repeat Step 3 before continuing.

## Step 6 — Verify identity on the actual GPU and data

```bash
python -u pilot_runner.py identity --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_010.pt --out runs/identity_gpu_seed42_v1 --device cuda --steps 3 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42
```

Required: `IDENTITY CONTROL PASSED | exact states, predictions and metrics`. `identity_control.json` must show all checks true and `max_model_absolute_difference: 0.0`. This compares independently executed identity branches, including model parameters, batch-normalization buffers, optimizer, scheduler, RNG states, first gradient/update, predictions, and metrics. CPU equality does not guarantee GPU equality; if this fails, diagnose it before interpreting any candidate result. Every probe invocation also runs an identity check and aborts candidate trials if it fails.

## Step 7 — Run the first small experiment, including two strengths

```bash
python -u pilot_runner.py probe --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_010.pt --out runs/pilot_smoke_seed42_e010 --device cuda --classes D20 --candidates identity,brightness_low,brightness_high --horizons 1,10 --episodes 1 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42 --max-minutes 75
```

Expected: three completed trials and `PROBE COMPLETE`. Identity gains must be exactly zero. Brightness trials must record changed target pixels; their measured gains may be positive, negative, or zero. **Do not require a favorable augmentation result for the pilot to pass.** Its purpose is to measure faithfully.

If it pauses, the exact resume command is:

```bash
python -u pilot_runner.py probe --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_010.pt --out runs/pilot_smoke_seed42_e010 --device cuda --classes D20 --candidates identity,brightness_low,brightness_high --horizons 1,10 --episodes 1 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42 --max-minutes 75 --resume
```

Bring this output back before making a controller decision. From Windows PowerShell:

```powershell
Set-Location 'C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project'
scp -r solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/pilot_ccast_v1/runs/identity_gpu_seed42_v1 .\reports\
scp -r solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/pilot_ccast_v1/runs/pilot_smoke_seed42_e010 .\reports\
```

## Step 8 — Expand only after the small experiment is technically valid

The next command covers all four target classes, all 11 candidates, two batch-stream episodes, and three horizons: 88 candidate trials, yielding 264 descriptive CSV rows. Reference branches are additional work. It may require several allocations; runtime has not been measured on CCAST for this runner.

```bash
python -u pilot_runner.py probe --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_010.pt --out runs/pilot_grid_seed42_e010 --device cuda --horizons 1,10,50 --episodes 2 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42 --max-minutes 75
```

Resume without duplicating completed trials:

```bash
python -u pilot_runner.py probe --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_010.pt --out runs/pilot_grid_seed42_e010 --device cuda --horizons 1,10,50 --episodes 2 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42 --max-minutes 75 --resume
```

Then use the other fixed training stages:

```bash
python -u pilot_runner.py probe --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_002.pt --out runs/pilot_grid_seed42_e002 --device cuda --horizons 1,10,50 --episodes 2 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42 --max-minutes 75
python -u pilot_runner.py probe --data ../dataset/processed_v1 --design design --checkpoint runs/trajectory_seed42_v1/epoch_018.pt --out runs/pilot_grid_seed42_e018 --device cuda --horizons 1,10,50 --episodes 2 --batch-size 32 --eval-batch-size 64 --threads 2 --seed 42 --max-minutes 75
```

Run one command at a time and complete/resume it before starting the next. To resume either, repeat its exact command with `--resume` appended. The budget is checked at complete-candidate boundaries. Identity checking, reference-gradient construction, and the current candidate can exceed the nominal budget, so leave allocation time for those operations. A scheduler kill discards only an unfinished candidate; completed trial JSON files remain. Restore the same source files, design, settings, and software environment for a resume.

## What to examine before designing the controller

1. Technical validity: identity equality, data checksums, correct target-only changes, finite metrics, completed status, and plausible elapsed time.
2. Magnitude sensitivity: low versus high **within the same class, operation, checkpoint, episode, and horizon**. Do not compare raw magnitude numbers across different operations.
3. Feedback usefulness: compare the inexpensive signals against later audit loss gains. Look for whether they rank interventions better than correctness-only feedback and whether apparent benefits disappear or reverse over longer horizons. Do not choose a signal because it correlates with the same data used to construct it.
4. Stability and collateral effects: compare episode variation, early/middle/late behavior, and all four class outcomes. Add independent trajectories before claiming a reproducible effect. Do not treat crops, horizons, or shared-checkpoint branches as independent training runs.
5. Cost: use per-branch time and invocation time. Reference times appear in several trial records because a reference is shared; do not sum duplicated reference times as separate executions. Full cost includes checkpoint training, validation feedback, repeated controls, and future policy tuning.

Only after those checks should we specify an online rule. A defensible candidate would maintain class-specific utilities for admissible operation–magnitude pairs, permit identity, and change both selection and magnitude using measured benefit with uncertainty. Whether that complexity is justified, whether a simpler correctness rule suffices, and what constitutes a distinct contribution remain open empirical questions.

## Reproduction and output interpretation

The design was prepared locally with:

```powershell
python pilot_runner.py prepare --manifest dataset_splits_v1/crops.csv --out reports/pilot_design_v1 --seed 20261016 --eval-per-class 128
python pilot_runner.py preview --manifest dataset_splits_v1/crops.csv --design reports/pilot_design_v1 --qa-dir tmp/crop_review_extract/crop_review/qa --out reports/pilot_previews_local_v1
```

Those directories already exist. These are provenance commands, not instructions to overwrite them. Any changed bank or panel selection requires a new output directory and a new design hash.

Each probe directory contains `config.json`, environment records, `run_status.json`, `identity_control.json`, exact batch plans, `trials/*.json`, and `results.csv`. Trial files retain reference and candidate predictions/losses per image at each horizon, plus signal values and effect summaries. They support later grouped analysis; the CSV is descriptive and does not perform hypothesis tests, pick a winner, or update an online controller.

`conditional_retention` is null when no target examples are correctly classified in the reference view. Its denominator is recorded. Do not replace this with zero or treat a tiny denominator as reliable evidence. Preserve failed runs and errors. Code/design/configuration mismatches are deliberate hard failures, not reasons to disable integrity checks.
