# Bounding-box geometry diagnostic

Command: `python train_baseline.py geometry --data dataset_splits_v1 --out reports/baseline_geometry_v1 --device cpu`

The classifier uses only `log(crop_width / crop_height)` from the frozen manifest. Twenty quantile bins and their class-majority predictions are fitted on the 38,513 training crops; evaluation uses the 8,242 validation crops. No pixel data are read. Test metadata are checked for split consistency, but no test predictions or metrics are computed.

| Validation metric | Aspect-ratio classifier | Training-majority prediction (D00) |
| --- | ---: | ---: |
| Accuracy | 0.670347 | 0.472943 |
| Macro-F1 | 0.443889 | 0.160544 |
| D00 recall | 0.964854 | 1.000000 |
| D10 recall | 0.914737 | 0.000000 |
| D20 recall | 0.090452 | 0.000000 |
| D40 recall | 0.000000 | 0.000000 |

**Interpretation:** annotation geometry alone predicts the two directional crack classes strongly. Accuracy can therefore look respectable even when all potholes are missed. This motivates macro-F1 and per-class reporting and cautions against attributing future CNN accuracy entirely to learned defect appearance. This result does not demonstrate that the CNN exploits padding, nor does it establish whether a different preprocessing scheme would perform better. The current preprocessing remains fixed across comparisons.

This is a diagnostic result, not the working CNN baseline or an augmentation comparison. Full metrics and confusion matrices are in `geometry_metrics.json`; configuration and environment are recorded alongside it.
