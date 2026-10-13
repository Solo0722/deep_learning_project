# Conventional augmentation baseline: verified validation results

The completed seed-42 run selected epoch 20 of 20 by maximum validation macro-F1. Recomputing the saved predictions gives **accuracy 0.905241 (90.52%)** and **macro-F1 0.891457**: 7,461 of 8,242 validation crops were classified correctly. There are 781 errors. These are validation results from one training seed.

## Per-class performance

| Class | Defect | Support | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| D00 | Longitudinal cracks | 3898 | 0.9042 | 0.9472 | 0.9252 |
| D10 | Transverse cracks | 1771 | 0.9524 | 0.9368 | 0.9445 |
| D20 | Alligator cracks | 1592 | 0.8759 | 0.8021 | 0.8374 |
| D40 | Potholes | 981 | 0.8686 | 0.8491 | 0.8588 |

D20 has the lowest recall and F1. The largest directional confusion is D20 to D00: 223 crops, or 14.01% of all D20 validation crops and 28.55% of all validation errors. This identifies an error pattern; it does not establish whether the cause is class imbalance, visual overlap, annotation ambiguity, or a particular augmentation. The next investigation should inspect representative validation errors and compare controlled policies.

## Learning behavior

Training loss decreased from 0.6772 to 0.2670; validation loss decreased from 0.5632 to 0.2735. Validation macro-F1 increased from 0.7386 to 0.8915. There were temporary validation fluctuations, but no sustained validation-loss deterioration over this 20-epoch run. The final epoch is the best under the declared schedule; this does not demonstrate convergence or justify selectively extending this condition.

Training metrics were accumulated on augmented batches while weights changed and dropout was active. Validation metrics were computed after each epoch with deterministic inputs and evaluation mode. Consequently, early validation metrics exceeding training metrics do not themselves indicate a data error, and the plotted train/validation difference is not a controlled estimate of a generalization gap.

## Geometry comparator

The metadata-only comparator uses bounding-box aspect ratio and reaches validation accuracy 0.6703, macro-F1 0.4439. Always predicting the training-majority class reaches 0.4729 and 0.1605, respectively. The CNN exceeds these particular comparators. This does not rule out use of annotation geometry or country-related cues by the CNN. All compared augmentation methods should retain the same preprocessing.

## Reproducibility and verification

The model contains 695,732 trainable parameters and was randomly initialized. The verified configuration uses the shared fixed mild policy, unweighted cross-entropy, AdamW, batch size 64, a 20-epoch cosine learning-rate schedule, and no class resampling. The run used NVIDIA A10 on gpu0022, PBS job 784502.bright04, PyTorch 2.3.0, CUDA build 12.1, Pillow 10.3.0, and NumPy 1.26.4.

Recorded epoch intervals total 1284.4 seconds (21.41 minutes); the median after epoch 1 is 61.0 seconds. These intervals include training and validation and exclude initialization and post-epoch checkpoint writes. They are not a measurement of total allocation time.

The source and manifest hashes match the executed configuration. All 20 epoch records agree across JSON and CSV; their confusion matrices reproduce their metrics and expected class counts. Every selected prediction matches exactly one frozen validation-manifest record, including source, image, group, and label. Selected metrics reproduce from the predictions. Both checkpoints load successfully on CPU; their configurations, histories, selected predictions, and selected weights agree. The selected weights fit the declared architecture and are finite. See `verification.json` for checksums and detailed verification.

No crop pixels were loaded for this numerical review, and inference was not rerun. The saved training augmentation preview was inspected separately. The development run records no test evaluation; the analysis uses validation predictions only. Visual inspection of individual validation errors remains outstanding. Current checks do not prove complete capture-sequence independence beyond the documented grouping procedure.

## Deliverable status and next work

The initial working baseline is complete: the pipeline trains, saves checkpoints, selects a model using validation macro-F1, and supplies verified metrics and figures. The prepared splits are complete for the current protocol. This one-seed result provides no evidence yet that class adaptation improves performance. The fixed class-specific, adaptive, repeated-seed, and ablation comparisons remain to be run after their protocol is finalized. Revised methodology and final writing checks remain separate submission tasks. Preserve the held-out test set until the comparison protocol is fixed.

## Files

- `learning_curves.png` / `.pdf`: training and validation loss, accuracy, and macro-F1.
- `confusion_matrix.png` / `.pdf`: counts and percentages within each true class.
- `per_class_performance.png` / `.pdf`: recall and F1 by class.
- `per_class_metrics.csv`, `confusion_counts.csv`: numerical tables.
- `validation_errors.csv`: all 781 misclassified validation crop identifiers, without pixels or confidence estimates.
- `verification.json`: consistency checks, artifact hashes, timing scope, and analysis versions.

Reproduce with `python summarize_baseline.py --run reports/conventional_seed42_v1 --manifest dataset_splits_v1/crops.csv` from the project directory. The analysis script leaves original run artifacts unchanged.
