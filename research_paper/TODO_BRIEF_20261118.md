# Brief for the writing chat: open \todo items we can close now (Nov 18)

Paste everything below the line into the writing chat.

---

Please close these `\todo`s in main.tex using the facts below. Leave the others (VI.B, the Discussion paragraph, total GPU-hours) as `\todo`s; they wait on the Part C runs. Don't change any numbers, and don't make claims about the method (Part C) yet.

**1. Introduction, one-sentence summary (to-do 2): write the measurement half only.**
Finding: whether an augmentation edit helps or hurts depends on the class. The class × edit interaction is significant for Faster R-CNN on all three datasets (RDD2022 p = 4.5e-7, VOC p = 2.4e-5, dacl10k p = 4e-11), but for RetinaNet only on VOC (VOC p < 1e-6; RDD2022 p = 0.77; dacl10k p = 0.34). So the class dependence is real but varies with the detector. Leave the method half as a `\todo` ("CUA results pending").

**2. Table II, RDD2022 counts (to-do 3).**
- Training split: 25,863 images
- Reference set: 1,000 images, held out from the original training split (25,863 + 1,000 = 26,863)
- Validation: 5,760
- Test: 5,760

**3. dacl10k licence (to-do 4).**
Use: "dacl10k is released under CC BY-NC 4.0; we use it for non-commercial research."
(The creators' toolkit README says CC BY-NC 4.0; the Hugging Face mirror's metadata says CC BY 4.0. We follow the creators' licence.)

**4. False-positive threshold (to-do 6).**
A target-free test image counts as a false-positive image if the detector returns at least one box with confidence ≥ 0.5 on it. Report the count and the fraction of target-free test images.

**5. Estimator cost (to-do 5): fill the estimator share only.**
On RDD2022 with Faster R-CNN, re-estimating class utility every K = 200 iterations takes about 15% of epoch wall time, which adds about 18% to training time compared with training without the estimator. Leave total GPU-hours as `\todo`.

**6. Fig. 1 (to-do 1): suggestion only, don't draw it.**
Candidate: one dacl10k image containing both Crack and Rust. With Faster R-CNN, the contrast edit helps Crack and Hollowareas but hurts Rust, ExposedRebars, Restformwork and Weathering (Holm-significant). The caption idea: "the same edit helps one class and hurts another in the same image, so a single global policy has to trade them off." Keep the `\todo` until the image is chosen.
