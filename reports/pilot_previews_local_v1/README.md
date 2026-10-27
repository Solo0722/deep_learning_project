# Training transformation previews

These 35 sheets use 56 distinct training examples recovered from existing JPEG contact sheets. They are lossy visual proxies, not original PNG crops and not training inputs. Regenerate from the original PNGs using Step 4 in PILOT_RUNBOOK.md before judging exact image appearance.

Each sheet shows original, shared conventional augmentation, and low/high extra strengths. Signed operations show both directions. The sample IDs printed below rows match the frozen training manifest.

| Source | Brightness | Contrast | Blur | Horizontal translation | Vertical translation |
|---|---|---|---|---|---|
| China_Drone | [Open](China_Drone_brightness.png) | [Open](China_Drone_contrast.png) | [Open](China_Drone_blur.png) | [Open](China_Drone_translate_x.png) | [Open](China_Drone_translate_y.png) |
| China_MotorBike | [Open](China_MotorBike_brightness.png) | [Open](China_MotorBike_contrast.png) | [Open](China_MotorBike_blur.png) | [Open](China_MotorBike_translate_x.png) | [Open](China_MotorBike_translate_y.png) |
| Czech | [Open](Czech_brightness.png) | [Open](Czech_contrast.png) | [Open](Czech_blur.png) | [Open](Czech_translate_x.png) | [Open](Czech_translate_y.png) |
| India | [Open](India_brightness.png) | [Open](India_contrast.png) | [Open](India_blur.png) | [Open](India_translate_x.png) | [Open](India_translate_y.png) |
| Japan | [Open](Japan_brightness.png) | [Open](Japan_contrast.png) | [Open](Japan_blur.png) | [Open](Japan_translate_x.png) | [Open](Japan_translate_y.png) |
| Norway | [Open](Norway_brightness.png) | [Open](Norway_contrast.png) | [Open](Norway_blur.png) | [Open](Norway_translate_x.png) | [Open](Norway_translate_y.png) |
| United_States | [Open](United_States_brightness.png) | [Open](United_States_contrast.png) | [Open](United_States_blur.png) | [Open](United_States_translate_x.png) | [Open](United_States_translate_y.png) |

Visual checks covered United_States translation-x, China_Drone blur, India brightness, Norway contrast, and Japan translation-y. Thin strips, faint cracks, and low-resolution crops are present. Stronger blur can reduce detail; translation clips boundary content. No blanket label-preservation approval follows from this screening.
