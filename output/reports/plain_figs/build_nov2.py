import os
from plain_doc import Report

H = os.path.dirname(os.path.abspath(__file__))
F = lambda n: os.path.join(H, n)

r = Report("Class-Adaptive Data Augmentation for Multi-Class Road Damage Detection",
           "Preliminary Results (November 2, 2026)")

r.h1("1. Overview")
r.p("This report covers the first results for my class-adaptive augmentation method on RDD2022. The idea is "
    "simple. Different damage classes may need different augmentation, so instead of using one augmentation "
    "recipe for every image, the method picks local appearance edits (brightness, contrast, gamma, blur and "
    "noise) separately for each class. To decide which edits are allowed, the controller checks during "
    "training whether the current detector can still find objects of that class after the edit.")
r.p("I ran three sets of experiments. The first compared five training conditions over 20 epochs. The second "
    "added geometric augmentation to every arm and then ran a longer paired comparison to epoch 8. The third "
    "was a short diagnostic to check that training is reproducible. In short, the prototype works, but it did "
    "not improve accuracy. It reached 22.56 AP50:95 compared with 23.06 for a shared online policy, and at epoch "
    "8 it was 1.25 points below training with geometric augmentation alone. The rest of the report shows these "
    "results, looks at where the errors come from, and explains how the plan changes because of them.")

r.h1("2. Setup")
r.p("I use the four RDD2022 classes D00 (longitudinal cracks), D10 (transverse cracks), D20 (alligator cracks) "
    "and D40 (potholes). The images were split by source cluster into 26,863 training and 5,760 validation "
    "images. The validation set has 8,225 boxes, and 2,193 of its images contain no damage at all. I have not "
    "touched the test split.")
r.p("The detector is Faster R-CNN with a ResNet-50 FPN backbone, starting from COCO weights. Images are "
    "letterboxed to 640 pixels and trained with batch size 4, SGD (learning rate 0.0025, momentum 0.9, weight "
    "decay 0.0005) and seed 42 on one NVIDIA A10 on CCAST. I report COCO AP50:95 and AP50 on the validation "
    "set, per-class AP, and false detections on the damage-free images.")
r.p("Before each controller update, the current detector is run on a fixed panel of 256 training images, "
    "once as they are and once with each of 24 candidate edits (8 operations at 3 strengths). An edit is "
    "allowed for a class if the detector still finds at least 80% of the objects of that class it found on "
    "the clean image, with at least 16 objects from 8 clusters as support. The shared policy uses one set of "
    "allowed edits for all classes. The class-adaptive policy keeps a separate set for each class (or "
    "combination of classes) and applies the edit only inside the damage boxes.")

r.h1("3. Comparative Results")
r.h2("3.1 Five conditions over 20 epochs")
r.p("All five runs used the same starting weights, data order, number of steps (134,320) and evaluation. For "
    "each run I kept the checkpoint with the best validation AP50:95. Table 1 shows the results and Figure 1 "
    "shows the training curves.")
r.table("Validation results for the five 20-epoch runs (one run each).",
        ["Condition", "Best epoch", "AP50:95", "AP50", "AP50:95 at epoch 20", "Time (h)"],
        [["No augmentation", 14, "22.82", "51.14", "22.39", "11.1"],
         ["Fixed image-level edits", 14, "22.81", "50.69", "22.58", "11.5"],
         ["Fixed local edits", 18, "22.56", "50.16", "22.49", "13.3"],
         ["Online shared policy", 14, "23.06", "51.03", "22.59", "14.1"],
         ["Online class-adaptive (mine)", 18, "22.56", "50.18", "22.49", "14.5"]],
        [2.1, 0.8, 0.8, 0.7, 1.3, 0.8])
r.figure(F("curves_20epoch.png"), "Validation AP50:95 by epoch for the five runs. The big jump after epoch 12 "
         "is the planned learning-rate drop, and it shows up in every run.")
r.p("The shared online policy came out on top, 0.24 points above no augmentation. My class-adaptive policy was "
    "0.50 points below the shared policy and 0.26 below no augmentation. These gaps are small. Every run is a "
    "single seed and the runs already differed by about 0.8 points after the first epoch, before any "
    "augmentation was applied, so I would not read too much into differences of this size.")

r.h2("3.2 Adding geometric augmentation")
r.p("For the second study I gave every arm the same label-safe geometric augmentation (flips, small resizes "
    "and shifts), so the appearance edits are tested on top of a stronger baseline. All arms started from the "
    "same epoch-1 checkpoint. Table 2 shows the four-arm pilot at epoch 4.")
r.table("Four-arm pilot at epoch 4 (validation).",
        ["Arm", "AP50:95", "AP50", "Difference from geometry only"],
        [["Geometry only", "19.81", "45.05", "0.00"],
         ["Geometry + fixed local edits", "18.86", "43.44", "-0.95"],
         ["Geometry + online shared policy", "19.37", "43.87", "-0.44"],
         ["Geometry + class-adaptive policy", "19.61", "44.76", "-0.20"]],
        [2.6, 0.9, 0.9, 2.0])
r.p("Here the class-adaptive policy did better than the fixed edits and the shared policy, but it was still "
    "slightly below geometry only. Its curve was still climbing at epoch 4, so I extended it and the "
    "geometry-only arm to a fixed endpoint at epoch 8 to see whether it would catch up. It did not. Table 3 and "
    "Figure 2 show that the gap grew to 1.25 points.")
r.table("Paired comparison at epoch 8 (validation, fixed endpoint).",
        ["Metric (%)", "Geometry only", "Class-adaptive", "Difference"],
        [["AP50:95", "19.68", "18.44", "-1.25"], ["AP50", "44.33", "41.97", "-2.36"],
         ["AP75", "14.56", "13.42", "-1.14"], ["AR100", "43.70", "43.01", "-0.70"],
         ["AP small", "8.11", "7.40", "-0.71"], ["AP medium", "16.26", "14.96", "-1.31"],
         ["AP large", "21.18", "20.06", "-1.11"]],
        [1.6, 1.4, 1.4, 1.2])
r.figure(F("curves_epoch8.png"), "Paired run to epoch 8. Left: validation AP50:95 for both arms. Right: "
         "class-adaptive minus geometry only at each epoch.")

r.h1("4. Draft Results Section")
r.p("With the same training budget, the class-adaptive controller did not improve detection on RDD2022. "
    "Without geometric augmentation, the best validation AP50:95 came from the shared online policy (23.06), "
    "followed by no augmentation (22.82) and the class-adaptive policy (22.56). When geometric augmentation was "
    "used as a common base, the class-adaptive edits did better than fixed edits and the shared policy at "
    "epoch 4, but they stayed below geometry-only training. Over eight epochs the gap widened to 1.25 AP50:95, "
    "and all four classes were lower (Figure 3). The class-adaptive detector also made fewer false detections "
    "on damage-free images (2,890 against 3,371 boxes), but it found fewer of the real targets (recall 64.7% "
    "against 67.0%). So the policy mostly moved the detector to a more cautious operating point rather than "
    "making it better overall. All of these are single-seed results on the validation set.")

r.h1("5. Error Analysis")
r.h2("5.1 Per-class results")
r.figure(F("per_class.png"), "Per-class validation AP50:95. (a) The 20-epoch runs. (b) The epoch-8 paired run.")
r.p("In the 20-epoch runs, the class-adaptive policy lost about 0.6 points against the shared policy on "
    "longitudinal cracks, transverse cracks and potholes, and was about even on alligator cracks. Potholes "
    "benefited most from online control in general (the shared policy gained 0.75 points over no "
    "augmentation on D40). In the epoch-8 run every class dropped, and alligator cracks dropped the most "
    "(2.06 points). Because the loss shows up across all classes and not just the rare ones, I think the "
    "problem is in the policy itself rather than in one weak class.")

r.h2("5.2 False detections and recall")
r.table("Detection errors at confidence 0.25 and IoU 0.5 (epoch-8 run).",
        ["Measure", "Geometry only", "Class-adaptive"],
        [["Targets correctly detected", "5,512", "5,323"], ["Recall (%)", "67.0", "64.7"],
         ["Precision (%)", "20.4", "21.2"], ["False boxes on damage-free images", "3,371", "2,890"],
         ["Damage-free images with any detection (out of 2,193)", "1,239", "1,147"]],
        [3.4, 1.3, 1.3])
r.p("The class-adaptive detector is more conservative. It produces about 14% fewer false boxes, but it misses "
    "189 more real targets. I have not gone back and visually checked the damage-free images, so some of these "
    "false boxes could be damage that was never labelled.")

r.h2("5.3 Why the controller did not help")
r.p("Looking through the controller logs, I see three likely reasons. The first and most important is that "
    "the controller measures the wrong thing. It allows any edit the detector can tolerate, but tolerating an "
    "edit says nothing about whether training on it actually helps. In the 20-epoch run, the strongest "
    "strength was used for 76% of the edits simply because it passed the check.")
r.p("Second, leaving the image unchanged was never treated as an option. If an edit passed, it was applied. In "
    "the epoch-8 run, half of all applied edits were blur or noise, and nothing in the method ever tested "
    "whether those were useful.")
r.p("Third, there was not enough evidence for many class combinations. In the 20-epoch run, 9 of the 15 "
    "possible class combinations never had enough examples to get their own policy, and 8.6% of the "
    "opportunities fell back to no edit. By epoch 8 of the paired run, 77% of pothole opportunities were "
    "borrowing evidence from pooled data. On top of this, each controller update costs about 6 minutes of GPU "
    "time, roughly an extra hour per 20-epoch run.")

r.h2("5.4 Reproducibility")
r.p("The 20-epoch runs already disagreed after epoch 1 (16.54, 15.89, 16.42, 15.73 and 16.52 AP50:95), even "
    "though no augmentation had been applied yet, and an earlier smoke test showed arms with identical "
    "settings drifting apart. To check whether this was a bug, I ran a diagnostic on CCAST on November 2 with "
    "fully deterministic GPU settings. I compared plain repeated training, training after a controller probe, "
    "and training after saving and restoring a checkpoint. All 88 recorded comparisons matched exactly, "
    "including the inputs, model weights, optimizer state, gradients and random number states. This suggests "
    "the earlier drift came from GPU numerical non-determinism, and I found no sign that the probe or the "
    "resume step leaks state. The epoch-8 run used these strict settings. I also recomputed every reported "
    "metric from the saved predictions with the official COCO evaluation code, and the numbers matched with "
    "zero difference. What I have not measured yet is how much results vary across different seeds.")

r.h1("6. Plan for the Remaining Experiments")
r.p("These results tell me that choosing augmentation by what the detector can tolerate does not improve "
    "accuracy. For the next stage I want the method to ask a different question: does training on this edit "
    "actually help this class? The plan is as follows.")
r.bullets([
    "Measure first. Train short runs that each add one edit and compare the change in per-class AP with a "
    "no-edit run. This tells me directly whether different classes benefit from different edits.",
    "Rebuild the controller around benefit. Estimate each edit's benefit for each class during training from "
    "how well its gradient agrees with a small held-out set for that class, always keep 'no edit' as an "
    "option, and block edits that clearly hurt a class.",
    "Compare fairly against no augmentation, random (uniform) augmentation, a policy built from the measured "
    "effects, and the same controller with one shared score for all classes. The last one isolates what class "
    "information adds.",
    "Use three seeds for every comparison, a second detector (RetinaNet) and two more datasets (dacl10k for "
    "bridge damage and PASCAL VOC) so the conclusions do not rest on one dataset.",
    "Keep the current controller in the paper as an ablation and report the negative result.",
])
r.p("I estimate about 840 GPU hours on the CCAST A10 nodes. The single-edit runs go first, since everything "
    "else depends on whether classes actually differ in what helps them.")

r.save(os.path.join(H, "..", "CE738_Preliminary_Results_2026-11-02.docx"))
