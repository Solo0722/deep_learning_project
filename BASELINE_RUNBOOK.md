# Initial conventional baseline

This run establishes a working training and validation pipeline. It is not the complete augmentation comparison or evidence that the adaptive method works. The fixed class-specific and adaptive conditions follow after this run is verified.

## Configuration

- Model: `compact_cnn_v1`, randomly initialized; no pretrained weights. Four blocks with two 3x3 convolutions each (16, 32, 64, 128 channels), batch normalization, ReLU, and max pooling; average pooling; 64-unit head with dropout 0.25; four output logits.
- Data: frozen `dataset/processed_v1` crops; 38,513 training and 8,242 validation examples. Image, connected-group, and source-file memberships are checked across all three manifests. Test pixels are not loaded.
- Conventional policy: horizontal reflection with probability 0.5, brightness with probability 0.5 and factor uniform in [0.9, 1.1], contrast with probability 0.5 and the same factor range. Each decision is independent; operations affect the content rectangle while preserving letterbox padding. The same policy applies to all classes and stays fixed throughout training.
- Deterministic preprocessing: existing 224x224 RGB crops, channel scaling `RGB / 127.5 - 1`.
- Training: 20 epochs, seed 42, batch size 64, unweighted cross-entropy, shuffled samples without oversampling, AdamW (learning rate 0.001, weight decay 0.0001), cosine learning-rate schedule to 0.00001. Float32, deterministic operations, TF32 disabled.
- Checkpoint selection: highest validation macro-F1, earliest epoch in a tie. Report accuracy, macro-F1, per-class precision/recall/F1/support, and confusion matrix.
- This mild policy is the initial working baseline. The final shared/fixed/adaptive comparison must use a declared common operation bank and comparable tuning opportunities.

## 1. Copy the script from local Windows PowerShell

```powershell
scp "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\train_baseline.py" solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/
```

## 2. Confirm a GPU allocation in the CCAST SSH terminal

```bash
hostname
printf 'PBS_JOBID=%s\nCUDA_VISIBLE_DEVICES=%s\n' "$PBS_JOBID" "$CUDA_VISIBLE_DEVICES"
```

If back on a login node without an allocation, request one there:

```bash
interact -c 4 -g 1 -m 16 -t 2
```

Do not run that request inside an existing interactive job. Once the allocation starts, activate the environment:

```bash
cd /mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research
source /mmfs1/apps/pyenvs/anaconda3-2022.05/bin/activate pytorch
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('Visible GPUs:', torch.cuda.device_count())"
```

Expect CUDA `True` and exactly one visible GPU. The script also enforces these checks. A previously allocated session may have expired; the earlier successful check does not establish current availability.

## 3. Run the learning sanity check first

```bash
python -u train_baseline.py sanity --data dataset/processed_v1 --out runs/sanity_seed42_v1 --device cuda --steps 200 --seed 42
```

The script selects eight training crops per class and repeatedly fits those same 32 examples. It disables augmentation, dropout, and weight decay for this debugging exercise. Passing requires at least 95% accuracy on that subset and a final evaluation loss less than half the initial loss. This is a memorization check, not a generalization result. No validation or test images are used.

Expected final line: `SANITY PASSED`. Share the final output before beginning the full run. A failure means inspect the pipeline or optimization; do not silently relax the criterion. An augmentation preview is also saved for review, separately from the unaugmented fitting exercise.

Each new output directory must not already exist. Preserve earlier evidence; inspect an existing run rather than deleting it merely to rerun the command.

## 4. Start the complete train/validation run after the sanity check passes

**Status, October 12:** the student reported `SANITY PASSED` on CCAST node `gpu0022`: accuracy 1.0000 on the 32 selected training examples after 200 steps; evaluation loss 1.3894 to 0.0000 (rounded console values). The cuDNN execution-plan warning did not stop the run. The subsequent full train/validation run also completed all 20 epochs. Its selected epoch is 20, with validation accuracy 0.9052 and macro-F1 0.8915 (rounded console values). The commands below are retained for reproducibility; do not restart or resume this completed run. Next is artifact transfer and inspection.

```bash
python -u train_baseline.py train --data dataset/processed_v1 --out runs/conventional_seed42_v1 --device cuda --epochs 20 --batch-size 64 --workers 2 --threads 2 --seed 42 --augmentation shared --max-minutes 75
```

The first epoch provides an actual runtime estimate. The 75-minute setting checks elapsed time **only at epoch boundaries**; it is not a strict 75-minute wall-clock limit. Leave enough allocation time for an epoch to finish. An interrupted epoch is repeated from the last complete checkpoint.

If the script reports `PAUSED`, retain the output directory and script unchanged. After confirming sufficient GPU allocation time, repeat the command with `--resume`:

```bash
python -u train_baseline.py train --data dataset/processed_v1 --out runs/conventional_seed42_v1 --device cuda --epochs 20 --batch-size 64 --workers 2 --threads 2 --seed 42 --augmentation shared --max-minutes 75 --resume
```

Resume checks script, data, environment package versions, and training configuration. Keep the planned 20-epoch budget unchanged so the learning-rate schedule remains consistent. Local CPU resume equivalence has been verified; cross-GPU or cross-version bitwise equivalence is not claimed.

## 5. Evidence to collect

Run this in local Windows PowerShell to copy the completed run into the existing project reports directory, including checkpoints:

```powershell
scp -r solomon.owusu.1@prime.ccast.ndsu.edu:/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/runs/conventional_seed42_v1 "C:\Users\SOLOMON\OneDrive - North Dakota University System\deep_learning_project\reports"
```

The expected local folder is `reports/conventional_seed42_v1`. Its existing `REPORTED_RESULT.md` records the console evidence; the transfer adds the saved run artifacts. Inspect those artifacts before claiming the curves or per-class results have been reviewed.

- `run_status.json`: complete only when all 20 epochs finish; a paused run remains incomplete.
- `config.json`, `environment.json`: configuration, hashes, software versions, allocation, and device information.
- `history.csv` / `history.json`: training and validation losses, accuracy, macro-F1, learning rate, and epoch duration.
- `best.pt`, `last.pt`: selected model and resumable training state.
- `best_validation.json`, `validation_predictions.csv`: selected epoch metrics and individual validation predictions.
- `geometry_metrics.json`: validation performance using bounding-box aspect ratio alone, fitted on training metadata; a diagnostic for an available shortcut, not a CNN result.
- `augmentation_preview.jpg`: training-only policy preview.

**Completed:** transferred artifacts have been verified, and learning curves, per-class figures, and the confusion matrix have been generated and visually inspected. See `reports/conventional_seed42_v1/analysis/RESULTS.md`. Recomputed validation accuracy is 0.9052414463 and macro-F1 is 0.8914572917. D20 recall is 0.8021; its most common error is prediction as D00 (223 examples). The selected checkpoint and final checkpoint are consistent with epoch 20. Hold final test evaluation until the experimental protocol is fixed. Visual inspection of individual validation error images remains future analysis.

To reproduce the local analysis without loading crop pixels or running inference:

```powershell
python summarize_baseline.py --run reports/conventional_seed42_v1 --manifest dataset_splits_v1/crops.csv
```

## Local software verification on October 12

`python -m unittest discover -s tests -p test_train_baseline.py -v` passed four checks under Python 3.14.2, PyTorch 2.10.0+cpu, and Pillow 12.1.1:

- Known-answer accuracy and macro-F1 with unequal class support and a missed class.
- Repeatable augmentation, unchanged padding, deterministic validation, rejection of test-image loading and altered crop files.
- Rejection of manifest edits and source-group overlap across splits.
- Actual two-epoch CNN training on synthetic images: uninterrupted and resumed runs produced identical final model tensors, epoch metrics, and selected validation predictions. Test pixel files were absent from the fixture.

These checks establish software behavior on synthetic inputs. They do not replace the CCAST learning check or provide scientific road-defect classification results.
