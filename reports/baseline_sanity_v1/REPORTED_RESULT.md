# CCAST learning sanity check — reported result

Recorded October 12, 2026 from the student's pasted terminal output. The assistant has not independently retrieved the remote JSON, preview, or environment files.

- Host: `gpu0022`.
- Environment: activated CCAST `pytorch` environment; warning trace refers to Python 3.12.
- Data: `dataset/processed_v1`.
- Remote output: `runs/sanity_seed42_v1`.
- Mode: `sanity`, CUDA, 200 optimizer steps, seed 42.
- Inputs: script-selected eight training crops per class, 32 total. The mode disables augmentation, dropout, and weight decay while fitting. It does not evaluate validation or test images.

Reported output:

```text
Data: {'train': 38513, 'val': 8242} | mode: sanity
Sanity step 25/200 | training loss 0.0006
Sanity step 50/200 | training loss 0.0001
Sanity step 75/200 | training loss 0.0000
Sanity step 100/200 | training loss 0.0000
Sanity step 125/200 | training loss 0.0000
Sanity step 150/200 | training loss 0.0000
Sanity step 175/200 | training loss 0.0000
Sanity step 200/200 | training loss 0.0000
SANITY PASSED | accuracy 1.0000 | loss 1.3894 -> 0.0000
```

A `CUDNN_STATUS_NOT_SUPPORTED` execution-plan warning appeared during backward execution. The run continued through all 200 steps and returned to the shell. No environment or code change is warranted solely by this non-terminating warning; any subsequent training failure should be assessed from its own error. The precise backend fallback was not inspected.

**Conclusion:** the actual prepared crops can be loaded on CCAST and the model can memorize a small balanced training subset. The printed zero loss is rounded, not evidence of mathematically zero loss. This check does not establish generalization, complete the full working-baseline deliverable, or demonstrate augmentation effectiveness. Next is the complete conventional train/validation run.
