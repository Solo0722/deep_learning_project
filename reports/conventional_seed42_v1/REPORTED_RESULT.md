# Initial conventional CNN baseline — reported completion

Recorded October 12, 2026 from the student's pasted CCAST console output. Saved run artifacts have not yet been retrieved or independently inspected.

**Subsequent verification:** the run artifacts were later transferred and checked. Exact metrics, checkpoint consistency checks, and inspected figures are now available in `analysis/RESULTS.md` and `analysis/verification.json`. The text below preserves the original console report and its status at the time it was recorded.

The previously supplied command specifies the scratch-initialized `compact_cnn_v1`, seed 42, 20 epochs, batch size 64, and the fixed shared mild augmentation policy. The saved configuration will be checked after transfer.

```text
Epoch 19/20 | 59.3s | train loss 0.2688 | val loss 0.2769 | val acc 0.9018 | val macro-F1 0.8865
Epoch 20/20 | 61.0s | train loss 0.2670 | val loss 0.2735 | val acc 0.9052 | val macro-F1 0.8915
TRAINING COMPLETE | best epoch 20 | val macro-F1 0.8915
```

| Item | Reported value |
| --- | --- |
| Completed epochs | 20/20 |
| Selected epoch | 20 |
| Selected validation accuracy | 0.9052 (90.52%) |
| Selected validation macro-F1 | 0.8915 |
| Epoch 20 training loss | 0.2670 |
| Epoch 20 validation loss | 0.2735 |
| Epoch 20 duration | 61.0 seconds |

All metrics above are rounded console values. Total runtime, earlier learning behavior, per-class results, and individual errors cannot be reconstructed from these final two epochs.

The initial training and validation run is complete. Remaining evidence review: retrieve configuration, environment, run status, history, checkpoints, selected-epoch metrics, validation predictions, and augmentation preview; check consistency; generate learning curves and a confusion matrix; summarize per-class results. The checkpoint remains selected by validation macro-F1 under the declared 20-epoch schedule. Do not extend the run solely because its best checkpoint occurs at the last epoch.

This is one conventional-policy seed. It supplies a working CNN baseline and preliminary validation performance, but does not establish that class adaptation improves performance or quantify variation across seeds. The development script does not evaluate held-out test pixels. Comparative experiments and final test evaluation remain future work.
