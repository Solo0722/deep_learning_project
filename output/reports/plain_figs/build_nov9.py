import os
from plain_doc import Report

H = os.path.dirname(os.path.abspath(__file__))
F = lambda n: os.path.join(H, n)

r = Report("Class-Aware Augmentation Selection for Multi-Class Object Detection",
           "Main Results and Validation (November 9, 2026)")

r.h1("1. Overview")
r.p("In the preliminary report I showed that my first class-adaptive controller, which picks augmentations "
    "based on what the detector can tolerate, did not improve road damage detection. This report finishes "
    "that line of work and starts the next one. Sections 2 and 3 give the main comparison, the ablations and "
    "the repeat-run evidence for the first controller. Section 4 describes the revised method, which I am "
    "calling class-utility augmentation. Instead of tolerance, it uses an estimate of how much training on an "
    "edit actually helps each class. Sections 5 and 6 cover the new study design and where things stand. The "
    "code for the new study is finished and tested, and the first experiments are running on CCAST now, so "
    "the new results are not in this report yet.")
r.p("The main result from the first controller is consistent across every comparison I ran. Online control "
    "on its own helps a little (+0.24 AP50:95 over no augmentation for the shared policy). Adding "
    "class-specific tolerance rules on top of it takes that gain away (-0.50), and over a longer fixed "
    "horizon the class-adaptive policy ends up 1.25 points below geometry-only training, with every class "
    "lower. The pipeline itself checks out: repeated training reproduces exactly and all metrics were "
    "recomputed independently. That points to the controller's selection rule, not the implementation, as "
    "the problem, and it is the part the new method replaces.")

r.h1("2. Main Comparison and Ablations")
r.p("All of these runs use RDD2022 (four damage classes, 26,863 training and 5,760 validation images), Faster "
    "R-CNN with a ResNet-50 FPN backbone from COCO weights, and seed 42. I only used the validation set; the "
    "test set is still closed. I set up the conditions so that each one adds a single component to the one "
    "before it. That way the main comparison also works as an ablation of the method's parts (Table 1).")
r.table("Main comparison over 20 epochs, best validation checkpoint. Each row adds one component.",
        ["Condition", "What is added", "AP50:95", "AP50", "Change"],
        [["No augmentation", "-", "22.82", "51.14", "-"],
         ["Fixed image-level edits", "appearance edits", "22.81", "50.69", "-0.01"],
         ["Fixed local edits", "edits only inside boxes", "22.56", "50.16", "-0.26"],
         ["Online shared policy", "online tolerance check", "23.06", "51.03", "+0.50"],
         ["Online class-adaptive", "per-class rules", "22.56", "50.18", "-0.50"]],
        [2.0, 1.8, 0.8, 0.7, 0.8])
r.p("I then repeated the comparison with label-safe geometric augmentation (flips, small resizes and shifts) "
    "in every arm, starting all arms from the same epoch-1 checkpoint. Only the class-adaptive arm and the "
    "geometry-only arm were carried on to epoch 8 (Table 2).")
r.table("Comparison with geometric augmentation in every arm (validation).",
        ["Arm", "AP50:95, epoch 4", "AP50, epoch 4", "AP50:95, epoch 8", "AP50, epoch 8"],
        [["Geometry only", "19.81", "45.05", "19.68", "44.33"],
         ["+ fixed local edits", "18.86", "43.44", "not run", "-"],
         ["+ online shared policy", "19.37", "43.87", "not run", "-"],
         ["+ class-adaptive policy", "19.61", "44.76", "18.44", "41.97"]],
        [1.9, 1.15, 1.1, 1.15, 1.1])
r.figure(F("ablation.png"), "Differences from the reference condition. (a) The 20-epoch runs in Table 1. "
         "(b) The epoch-4 runs in Table 2.")
r.p("Reading the two tables together, online control is the only part that helped. It added 0.50 points over "
    "fixed local edits. Making the rules class-specific gave all of that back. With geometric augmentation "
    "in place, the class-adaptive policy did beat the fixed edits and the shared policy at epoch 4, but it "
    "never got above geometry only, and between epochs 4 and 8 it fell further behind (Figure 2). Figure 3 "
    "breaks this down by class. At epoch 8 all four classes are lower, and alligator cracks lose the most "
    "(2.06 points).")
r.figure(F("curves_epoch8.png"), "Paired run to epoch 8. Left: validation AP50:95. Right: class-adaptive minus "
         "geometry only at each epoch. The class-adaptive arm is behind at every epoch.")
r.figure(F("per_class.png"), "Per-class validation AP50:95. (a) The 20-epoch runs. (b) The epoch-8 paired run.")

r.h1("3. Repeat-Run and Robustness Evidence")
r.p("Since every condition above has only one training run, I put most of the validation effort into making "
    "sure the numbers themselves can be trusted. Table 3 lists the checks.")
r.table("Checks on the reported results.",
        ["Check", "What I did", "Result"],
        [["Repeat-run determinism", "Repeated the same training plainly, after a controller probe, and after a "
          "save and restore, with strict GPU settings (88 comparisons)", "All identical"],
         ["Metric recomputation", "Recomputed every endpoint metric from saved predictions with the official "
          "COCO code, written separately from the training code", "No difference"],
         ["Controller replay", "Re-matched 29,696 saved controller views and rebuilt the gates and routing with "
          "separate code", "Exact match"],
         ["Gate sensitivity", "Recomputed the 96 class-by-edit gates leaving out one image cluster at a time",
          "1 to 2 gates change"],
         ["Consistency over epochs", "Class-adaptive minus geometry only, epochs 2 to 8", "Negative at all 7"],
         ["Spread between starts", "AP50:95 after epoch 1 in five nominally identical runs", "15.73 to 16.54"]],
        [1.5, 3.5, 1.4])
r.p("The pipeline holds up. The weak point is the number of seeds. The runs were already 0.8 points apart after "
    "the first epoch, so the differences in Table 1, which are all under about 0.5 points, should be treated as "
    "unresolved. The 1.25-point drop at epoch 8 is larger and goes the same way at every epoch, but it is still "
    "one seed. For this reason every comparison in the new study uses three paired seeds and reports confidence "
    "intervals.")

r.h1("4. The Revised Method: Class-Utility Augmentation")
r.p("The error analysis in the preliminary report pointed to three problems with the first controller. It "
    "allowed any edit the detector could tolerate without asking whether the edit helps training. It never "
    "considered leaving the image unchanged. And it did not have enough examples for many combinations of "
    "classes.")
r.p("The revised method deals with all three. Every 200 training steps, it takes a small batch of training "
    "images and computes the gradient of the detector-head loss with and without each edit o. It also computes, "
    "for each class c, the gradient of the class-c loss on a small held-out reference set R_c of 64 images. The "
    "benefit of edit o for class c is how much one training step on the edited images would lower the class-c "
    "reference loss, compared with the unedited images:")
eq = r.p("u(o, c) = η <g_o − g_0, h_c> / (L_c(R_c) + δ)")
eq.alignment = 1
r.p("Here g_o and g_0 are the gradients with and without the edit, h_c is the class-c reference gradient, and "
    "δ = 0.01. I smooth these estimates over time and keep a standard error for each one. For a training image "
    "that contains a set of classes S, the policy averages the benefit of each edit over the classes in S. "
    "'No edit' is always a candidate with a benefit of zero. Any edit that is reliably harmful for one of the "
    "classes in S (one-sided 95%) is blocked, and the edit is then drawn from a softmax over the scores with "
    "10% random exploration.")
r.p("I also built a shared-utility version that works exactly the same way, except that it averages the "
    "benefit over all classes, so every image gets the same preferences. Comparing the two tells me how much "
    "the class information is worth on its own. Figure 4 shows how the pieces fit together.")
r.figure(F("method.png"), "Class-utility augmentation. Top: in the training loop, an edit is chosen for each "
         "image based on the classes it contains. Bottom: every 200 steps the benefit table is updated from "
         "how well each edit's gradient agrees with held-out reference images of each class.")

r.h1("5. Study Design")
r.p("To make sure the conclusions are not specific to road damage, the new study uses three datasets (Table 4) "
    "and two detectors, Faster R-CNN and RetinaNet, both with ResNet-50 FPN backbones.")
r.table("Datasets.",
        ["Domain", "Dataset", "Reason for including it"],
        [["Road damage", "RDD2022 (D00, D10, D20, D40)", "Thin cracks next to blob-like potholes; 35% of images "
          "contain more than one class"],
         ["Bridge damage", "dacl10k (damage classes, polygons converted to boxes)", "Classes defined by colour "
          "(rust, efflorescence) next to classes defined by shape (cracks)"],
         ["General objects", "PASCAL VOC 2007 and 2012", "A standard benchmark outside damage detection"]],
        [1.2, 2.3, 2.9])
r.p("Every arm uses horizontal flips and scale jitter. On top of that, the method chooses from 13 edits at one "
    "moderate strength each: translation, rotation (up to 10 degrees), vertical flip, brightness, contrast, "
    "saturation, hue, Gaussian blur, Gaussian noise, JPEG compression, sharpening, random erasing and mosaic. "
    "Vertical flip is left out for RDD2022 and VOC because it does not keep the labels sensible there. The "
    "experiments are listed in Table 5.")
r.table("Planned experiments.",
        ["Experiment", "Design", "Question"],
        [["Single-edit study", "Shared 3-epoch start, then 3 more epochs with no edit or with one edit; 3 "
          "datasets, 2 detectors, 3 seeds (258 runs)", "Do classes benefit from different edits?"],
         ["Short vs full runs", "4 edits trained on the full schedule", "Do the short runs predict full training?"],
         ["Class-targeted study", "Edit applied only inside one class's boxes", "Does an edit aimed at one class "
          "affect the others?"],
         ["Main comparison", "No extra edits, random edits, policy from the single-edit results, shared-utility, "
          "class-utility; full training, 3 seeds", "Does class information beat the same method without it?"],
         ["Ablations", "No harm blocking, no 'no edit' option, weights by inverse AP, the first controller",
          "Which parts of the method matter?"]],
        [1.4, 3.0, 2.0])
r.p("The main metric is macro AP50 on the test set, with AP50:95, per-class AP and false detections as "
    "secondary metrics. All arms share seeds, and each difference gets both a paired bootstrap interval over "
    "test images and a paired t interval over seeds. I will only claim a difference if both intervals exclude "
    "zero. The test set will be opened once, after every setting is fixed.")

r.h1("6. Current Status")
r.p("Over the past week I wrote the code for the new study as a separate package. It includes loaders and data "
    "checks for all three datasets (class co-occurrence, a near-duplicate screen between splits, and the "
    "held-out reference split), all 13 edits with correct box handling, both detectors, a trainer that can "
    "resume from a checkpoint and stop cleanly before a cluster time limit, COCO evaluation with a per-class "
    "bootstrap, automatic job scripts for CCAST, and the analysis for the single-edit study. The code passes 344 "
    "automated tests, and every cluster job runs the tests before it starts training.")
r.p("The experiments are running on CCAST now. I started with data preparation and short smoke tests on "
    "RDD2022, followed by the single-edit study and a check of the training length, both with Faster R-CNN. "
    "VOC and dacl10k come next, once they are downloaded and pass the data checks. Table 6 is the main table "
    "this first batch will fill in. I will add the numbers as the runs finish.")
r.table("Single-edit study on RDD2022 with Faster R-CNN: change in validation AP50 against no edit, mean of "
        "3 seeds (runs in progress).",
        ["Edit", "D00", "D10", "D20", "D40", "Macro"],
        [[e, "", "", "", "", ""] for e in
         ["Translation", "Rotation", "Brightness", "Contrast", "Saturation", "Hue", "Gaussian blur",
          "Gaussian noise", "JPEG compression", "Sharpening", "Random erasing", "Mosaic"]],
        [1.7, 0.9, 0.9, 0.9, 0.9, 0.9])

r.h1("7. Remaining Schedule")
r.table("Plan to the final package.",
        ["Date", "Milestone", "What I plan to have"],
        [["Nov 16", "Complete draft", "Single-edit results for RDD2022 (and VOC and dacl10k if ready), the full "
          "manuscript draft with the method figure, and a list of remaining experiments"],
         ["Nov 23", "Revised draft", "Main comparison on RDD2022 with 3 seeds, ablations, the short vs full run "
          "check, and a revised discussion and limitations"],
         ["Dec 4", "Final package", "Remaining datasets and the second detector as compute allows, plus the "
          "final manuscript, code, data documentation and slides"]],
        [0.8, 1.4, 4.2])
r.p("If GPU time gets tight, I will keep the single-edit study and the main comparison with Faster R-CNN on all "
    "three datasets, and cut from the end of the list: first the ablations, then RetinaNet, then the "
    "class-targeted study. The results of the first controller stay in the paper as the tolerance-based "
    "baseline and as a negative result.")

r.save(os.path.join(H, "..", "CE738_Main_Results_and_Validation_2026-11-09.docx"))
