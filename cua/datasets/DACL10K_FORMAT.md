# dacl10k format: ASSUMPTION, confirm on CCAST

The dacl10k dataset was NOT available on the development machine when `cua/datasets/dacl10k.py`
was written (plan Task 5 Step 1 is an operator step). The adapter targets an ASSUMED labelme-style
layout. Everything below is unverified until checked against the real download.

## Assumed layout

```
<root>/annotations/<split>/<name>.json     # splits e.g. train, validation (and testdev if labelled)
<root>/images/<split>/<name>.jpg
```

Every `*.json` under `<root>/annotations/*/` is pooled; the image is `<root>/images/<split>/` plus the
basename of `imagePath`. The adapter ignores the official splits and re-splits by duplicate group.

## Assumed JSON keys

Top level: `imageWidth`, `imageHeight`, `imagePath`, `shapes`.
Each shape: `label` (string) and `points` (`[[x, y], ...]`, polygon vertices in image pixels).
Each polygon becomes one box (min/max of its points, clipped to the image). Shapes with fewer than
3 distinct points, or collinear points, are dropped; so are shapes whose clipped box is empty.
Labels not in `DACL_DAMAGE` (component labels such as "Bearing") are dropped.

## To check on CCAST (do these, then update the adapter/tests if anything differs)

1. Key names: are they exactly `imageWidth`, `imageHeight`, `imagePath`, `shapes`, `label`, `points`?
2. Label names: do the damage labels match `DACL_DAMAGE` exactly (spelling, case)? Update the tuple if not.
3. Test labels: are annotations present for the official test split? If not, only train/validation
   are pooled (as designed), but record this here.
4. Bridge id: does any file name encode a bridge identifier? If so, use it as `Record.group`
   instead of `duplicate_groups` and note it here.
5. Directory layout: does it match the assumed layout above? Adjust `load_dacl10k` if not.
