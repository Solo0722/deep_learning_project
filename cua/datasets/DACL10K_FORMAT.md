# dacl10k format (confirmed on CCAST, 2026-11-14)

The official downloads (S3, GigaMove) are gone. We use the Hugging Face mirror `Voxel51/dacl10k`
(FiftyOne export, 8,922 images, CC BY-NC 4.0), downloaded with `hf download --repo-type dataset`.

## Layout

```
<root>/samples.json            # {"samples": [...]}, one entry per image
<root>/data/data_<k>/<name>.jpg
```

Each sample: `filepath` (relative to root), `tags` (official split: train / validation / test),
`metadata.width`, `metadata.height`, and `semseg_ground_truth_filled.polylines`. Each polyline has
`label` and `points` = list of rings, each a list of `[x, y]` normalised to [0, 1].

## Conversion (`parse_fiftyone_sample`)

- One box per polyline: min/max over all its rings, scaled to pixels, clipped to the image.
- Degenerate polygons (<3 distinct points, collinear) and empty clipped boxes are dropped and counted.
- Labels not in `DACL_DAMAGE` (component classes such as Bearing, EJoint, Drainage) are dropped and
  counted in `other_labels`, which goes into the prepare report so label-name mismatches show up.
- Unlabelled samples (field missing/null, or tagged `test` with no polylines) are skipped, so the
  unlabelled test-dev images never become damage-free negatives.
- Official splits are ignored; images are re-split 70/15/15 by near-duplicate group.

The older labelme-style layout (`<root>/annotations/<split>/*.json`) is still read when
`samples.json` is absent.
