# Training crop review — October 12, 2026

Evidence: `processed_v1_qa.tar.gz`, extracted under `tmp/crop_review_extract/crop_review`. All seven QA sheets were viewed: China_Drone, China_MotorBike, Czech, India, Japan, Norway, United_States. There are 112 training-only samples, four per source/class: one extreme-aspect-ratio example plus three deterministic samples. Validation/test pixels were not inspected.

## Observations

- The completed remote report records 55,002 224 x 224 PNG crops from 23,766 images. Its manifest hash matches `dataset_splits_v1/crops.csv`. Counts match every planned split/class cell.
- Previews have the expected aspect-preserving resizing and gray padding. Many typical D10 crops are horizontal strips; some D00 crops are vertical strips. Extreme examples can contain very little image area after resizing.
- Some small pothole examples are visibly blurred after enlargement, and some cracks are faint. These previews cannot establish that every annotation is correct, or estimate the overall rate of ambiguous labels.
- D00 also has diagonal and horizontal-looking examples under perspective. Do not replace the original class with a label inferred solely from the crop's apparent direction.
- Some crops include markings, shadows, repairs, or other cracks. A single box label does not guarantee a visually pure single-class patch. No relabeling is justified from these small previews alone.

## Geometry diagnostic and decision

`training_geometry.json` was calculated only from the training rows in the frozen crop manifest. Width is `right-left`; height is `bottom-top`. Reported proportions use those integer crop dimensions, before resizing. Long/short ratio is `max(width,height)/min(width,height)`.

Median width/height ratios are D00 0.680, D10 5.356, D20 1.382, and D40 1.652. D10 has 73.68% of training samples at long/short ratio >= 4, compared with D00 10.46%, D20 3.08%, and D40 2.45%. These facts demonstrate a relationship between annotated shape and class; they do not measure classifier performance.

Proceed with a preliminary baseline using the existing versioned crops. Add a simple geometry-only comparator, fitted on training examples and assessed on validation examples, to quantify the available shortcut. Keep all augmentation conditions on identical inputs. Do not delete thin examples merely because they are difficult, and do not treat high CNN accuracy as proof that the network recognizes road texture. A later context-crop or padding ablation would require an explicitly versioned preprocessing variant and matched comparisons.

## Environment and next steps

The student's current CCAST base interpreter (`/mmfs1/apps/pyenvs/anaconda3-2022.05/bin/python`) raises `ModuleNotFoundError: No module named 'torch'`. [CCAST's neural-network guide](https://kb.ndsu.edu/it/133762) documents activation of an environment named `pytorch` through the same Anaconda installation. The student subsequently confirmed successful activation at `/mmfs1/apps/pyenvs/anaconda3-2022.05/envs/pytorch/bin/python`: PyTorch 2.3.0, Pillow 10.3.0, CUDA build 12.1. CUDA availability was false in the existing allocation, whose earlier request did not include a GPU. Next request: `interact -c 4 -g 1 -m 16 -t 2`, using the options observed in the student's `interact -h`. GPU availability and actual CUDA execution still need verification after allocation. The newer [Python guide](https://kb.ndsu.edu/it/126857) documents custom virtual environments if needed.

After environment verification: prepare the CNN, check learning on a tiny training subset, then run full training and validation with logged configuration, metrics, and checkpoints. Keep test results unopened. The October 12 literature review and comparison table proceed alongside setup/training and do not require completed experiments.
