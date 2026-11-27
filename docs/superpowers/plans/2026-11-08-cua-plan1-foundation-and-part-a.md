# Class-Utility Augmentation — Plan 1: Foundation and Part A

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `cua` library (datasets, data gates, augmentation menu, detectors, trainer, evaluation) and run the study's Part A measurement (one augmentation at a time) plus the epoch-budget and proxy checks on CCAST.

**Architecture:** A new package `cua/` at the project root, independent of the frozen v1/v2 release folders (their logic is ported, not imported). Every dataset is converted once to a common JSONL record format; every augmentation is a pure function on a letterboxed sample with a keyed RNG so all arms share identical base draws. One trainer serves every run type; runs are described by `RunSpec` objects and turned into PBS jobs.

**Tech Stack:** Python 3.12, PyTorch 2.3.0 + TorchVision 0.18.0 (CCAST training env), NumPy, Pillow, pycocotools 2.0.10, SciPy (analysis only), pytest.

**Spec:** `docs/superpowers/specs/2026-11-08-class-utility-augmentation-design.md`

**Later plans (not in scope here):** Plan 2 — benefit estimator, policies, Part C main comparison, §9.4 tuning. Plan 3 — Part B, RetinaNet Part C coverage, ablations, paper figures.

## Global Constraints

- Image canvas: 640×640 letterbox, pad value 114; boxes in continuous XYXY pixel coordinates.
- Optimiser: SGD, momentum 0.9, weight decay 5e-4, base LR 0.0025, batch 4, linear LR warm-up over 1,000 iterations from 0.001× base.
- Full-schedule LR: ×0.1 after 60% of epochs, ×0.01 after 80% (rounded to whole epochs).
- Mixed precision on (fp16 autocast + GradScaler); TF32 allowed; no `torch.use_deterministic_algorithms`.
- Seeds {0, 1, 2}, shared across arms.
- Base recipe (every arm): horizontal flip p = 0.5; isotropic scale jitter 0.8–1.2 on the fixed canvas; reject any draw that crops an annotated object (fall back to scale 1.0).
- Menu strengths (spec §6): translation ±10% canvas; rotation ±10°; brightness ×[0.7, 1.3]; contrast ×[0.7, 1.3]; saturation ×[0.5, 1.5]; hue ±0.05; blur σ ∈ [0.5, 1.5] px; noise σ = 0.03 (on [0, 1] scale); JPEG quality ∈ [30, 60]; sharpen factor ∈ [1.5, 2.5]; random erasing 2–10% of canvas, covering ≤ 50% of any box; mosaic 2×2, drop objects < 30% visible.
- Validity: RDD excludes `vflip`; VOC excludes `vflip`; dacl10k excludes none.
- Part A: edit applied with probability 0.5 per training image.
- Reference set: ~1,000 group-disjoint training images, ≥ 100 objects per class where possible; removed from training in every arm.
- Data gates: co-occurrence ≥ 20% of target-bearing training images with ≥ 2 classes; dacl10k class exclusion if < 300 training boxes or > 50% of its boxes cover > 50% of the image.
- Primary metric: macro AP50 (COCO, IoU 0.5, area all, maxDets 100), averaged over classes with ≥ 1 ground-truth object.
- Test split is never evaluated by any code in this plan.

## Review Focus

1. **Target-free images** (RDD has 10,233 in train) must pass through every edit, mosaic, collate and the trainer with `boxes` of shape (0, 4) — no crash, no fabricated boxes. Tests in Tasks 6, 8, 9, 10.
2. **Geometric edits that would push a box off-canvas or collapse it** must be rejected (resampled up to 10 times, then identity), never clipped silently. Tests in Tasks 6 and 8.
3. **Paired randomness:** for the same (seed, epoch, image index), the base geometry must be byte-identical whichever edit an arm uses. Test in Task 6.
4. **A PBS walltime stop** must leave a resumable run that continues from the last finished epoch, and a completed run must refuse to be overwritten. Test in Task 12.
5. **Classes with no validation objects / ignored (VOC "difficult") objects** must give `None` for that class and be excluded from macro AP50, not crash or count as zero. Test in Task 11.

---

### Task 1: Package scaffold and common record format

**Files:**
- Create: `cua/__init__.py`, `cua/records.py`, `pytest.ini`, `cua/requirements-test.txt`, `tests/cua/conftest.py` (test factories such as `mk`, `toy_sample`, `synthetic_train`; each later task adds the factories its tests name)
- Test: `tests/cua/test_records.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Record(image: str, width: int, height: int, boxes: np.ndarray, labels: np.ndarray, ignore: np.ndarray, group: str, source: str)` — `boxes` float64 (N, 4) original-image XYXY; `labels` int64 (N,) 0-based; `ignore` bool (N,).
  - `write_records(path: Path, records: list[Record]) -> None`, `read_records(path: Path) -> list[Record]` (JSONL, one record per line).
  - `write_classes(path: Path, classes: tuple[str, ...])`, `read_classes(path: Path) -> tuple[str, ...]`.
  - `Record.class_set() -> frozenset[int]` over non-ignored labels.

- [ ] **Step 1: Write failing tests**

```python
def test_roundtrip_preserves_arrays(tmp_path):
    r = Record("a/b.jpg", 600, 400, np.array([[1., 2., 30., 40.]]), np.array([2]), np.array([False]), "g1", "Japan")
    write_records(tmp_path / "x.jsonl", [r]); (back,) = read_records(tmp_path / "x.jsonl")
    assert back.boxes.dtype == np.float64 and back.boxes.shape == (1, 4)
    assert np.array_equal(back.labels, [2]) and back.group == "g1"

def test_empty_record_has_shape_0x4(tmp_path):
    r = Record("n.jpg", 10, 10, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s")
    write_records(tmp_path / "e.jsonl", [r])
    assert read_records(tmp_path / "e.jsonl")[0].boxes.shape == (0, 4)

def test_invalid_box_rejected():
    with pytest.raises(ValueError):
        Record("a.jpg", 10, 10, np.array([[5., 5., 5., 9.]]), np.array([0]), np.array([False]), "g", "s")

def test_class_set_ignores_ignored():
    r = Record("a.jpg", 50, 50, np.array([[0,0,9,9],[1,1,8,8.]]), np.array([0, 3]), np.array([False, True]), "g", "s")
    assert r.class_set() == frozenset({0})
```

- [ ] **Step 2: Run** `pytest tests/cua/test_records.py -v` — expected FAIL (module missing).
- [ ] **Step 3: Implement** `cua/records.py`. `__post_init__` validates: x2 > x1, y2 > y1, boxes within [0, width]×[0, height], equal lengths, relative POSIX image path. `pytest.ini` sets `testpaths = tests/cua`. `requirements-test.txt`: torch, torchvision, numpy, Pillow, pycocotools, scipy, pytest (unpinned for the local CPU env).
- [ ] **Step 4: Run** the tests — expected PASS.
- [ ] **Step 5: Commit** `feat(cua): common record format`.

### Task 2: Image loading and letterbox

**Files:**
- Create: `cua/image.py`
- Test: `tests/cua/test_image.py`

**Interfaces:**
- Consumes: `Record`.
- Produces:
  - `@dataclass Sample(rgb: np.ndarray, boxes: np.ndarray, labels: np.ndarray, ignore: np.ndarray)` — rgb uint8 (640, 640, 3), boxes in canvas coordinates.
  - `load_rgb(root: Path, record: Record) -> np.ndarray` (stored orientation, no EXIF transpose, RGB uint8; raises if decoded size ≠ record size).
  - `letterbox(rgb, record: Record, size: int = 640) -> tuple[Sample, dict]` — geometry dict keys exactly as `rdd_v2_pair8_v1/rdd_data.py:prepare_example`.
  - `restore_boxes(boxes: np.ndarray, geometry: dict) -> np.ndarray`.

- [ ] **Step 1: Write failing tests**

```python
def test_letterbox_then_restore_is_identity():
    rec = Record("x.jpg", 1000, 500, np.array([[100., 50., 300., 250.]]), np.array([0]), np.array([False]), "g", "s")
    sample, geo = letterbox(np.zeros((500, 1000, 3), np.uint8), rec)
    assert sample.rgb.shape == (640, 640, 3) and geo["pad_top"] == 160
    assert np.allclose(restore_boxes(sample.boxes, geo), rec.boxes)

def test_padding_value_is_114():
    sample, _ = letterbox(np.zeros((500, 1000, 3), np.uint8), Record("x.jpg", 1000, 500, np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0, bool), "g", "s"))
    assert (sample.rgb[0, 0] == 114).all()
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** by porting `prepare_example` / `restore_boxes` (bilinear resize, rounded sizes, centred padding). **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): letterbox and image loading`.

### Task 3: RDD adapter, reference split and data gates

**Files:**
- Create: `cua/datasets/__init__.py`, `cua/datasets/rdd.py`, `cua/gates.py`, `cua/prepare.py`
- Test: `tests/cua/test_gates.py`, `tests/cua/test_rdd.py`

**Interfaces:**
- Consumes: `Record`, `write_records`, `write_classes`.
- Produces:
  - `RDD_CLASSES = ("D00", "D10", "D20", "D40")`; `load_rdd(splits_dir: Path) -> dict[str, list[Record]]` with keys `train`, `val`, `test` from `detection_splits_v1/{train,val,test}.jsonl`; `group = source_cluster_id`; `source = row["source"]`; `ignore` all False.
  - `cooccurrence_rate(records) -> float` — share of records with ≥1 non-ignored box that have ≥2 distinct classes.
  - `class_box_stats(records, n_classes) -> list[dict(count:int, large_fraction:float)]` — `large_fraction` = share of that class's boxes covering > 50% of the image area.
  - `reference_split(train, n_classes, n_target=1000, min_per_class=100, seed=0) -> tuple[list[Record], list[Record]]` — group-disjoint; returns (train_rest, reference).
  - `prepare_dataset(name: str, splits: dict, classes: tuple, out_dir: Path, image_root: Path) -> dict` — writes `train.jsonl` (minus reference), `reference.jsonl`, `val.jsonl`, `test.jsonl`, `classes.json`, `source.json` (`{"image_root": ...}`), `gates.json`; raises `GateFailure` if co-occurrence < 0.20.
  - CLI: `python -m cua.prepare rdd --splits <dir> --images <raw root> --out <data_root>/rdd`.

- [ ] **Step 1: Write failing tests**

```python
def test_cooccurrence_rate_counts_only_target_images():
    recs = [mk(labels=[0, 1]), mk(labels=[0]), mk(labels=[])]   # mk = small Record factory in conftest
    assert cooccurrence_rate(recs) == 0.5

def test_reference_split_is_group_disjoint_and_meets_quota():
    train = synthetic_train(n_groups=400, per_group=3, n_classes=4, seed=1)
    rest, ref = reference_split(train, 4, n_target=100, min_per_class=20)
    assert {r.group for r in rest}.isdisjoint({r.group for r in ref})
    assert len(rest) + len(ref) == len(train) and len(ref) >= 100
    assert all(sum((r.labels == c).sum() for r in ref) >= 20 for c in range(4))

def test_reference_split_deterministic():
    train = synthetic_train(400, 3, 4, seed=1)
    assert reference_split(train, 4, 100, 20, seed=0)[1] == reference_split(train, 4, 100, 20, seed=0)[1]

def test_gate_failure_below_threshold(tmp_path):
    single = {"train": [mk(labels=[0])] * 50, "val": [mk(labels=[0])], "test": [mk(labels=[0])]}
    with pytest.raises(GateFailure):
        prepare_dataset("toy", single, ("a", "b"), tmp_path, tmp_path)

@pytest.mark.skipif(not FROZEN_SPLITS.exists(), reason="frozen RDD splits not present")
def test_rdd_counts_match_frozen_split():
    s = load_rdd(FROZEN_SPLITS)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (26863, 5760, 5760)
    assert round(cooccurrence_rate(s["train"]), 3) == 0.349
```

- [ ] **Step 2: Run** — FAIL.
- [ ] **Step 3: Implement.** `reference_split`: shuffle groups with `np.random.default_rng(seed)`; first pass adds groups containing any class still below `min_per_class` until all quotas are met or no such group remains; second pass adds shuffled groups until `n_target` images; stop. `gates.json` records co-occurrence, per-class stats, reference counts per class, split sizes.
- [ ] **Step 4: Run** — PASS (the frozen-split test runs locally because `detection_splits_v1/` is in the repo).
- [ ] **Step 5: Commit** `feat(cua): RDD adapter, reference split, data gates`.

### Task 4: PASCAL VOC adapter

**Files:**
- Create: `cua/datasets/voc.py`; Modify: `cua/prepare.py` (add `voc` subcommand)
- Test: `tests/cua/test_voc.py` (fixture: two tiny hand-written VOC XML files under `tests/cua/fixtures/voc/`)

**Interfaces:**
- Produces: `VOC_CLASSES` (the 20 standard names, alphabetical as in VOC devkit); `load_voc(devkit: Path, val_fraction: float = 0.10, seed: int = 0) -> dict[str, list[Record]]` — train/val from VOC2007+2012 `trainval`, test from VOC2007 `test`; `difficult=1` → `ignore=True`; `group` = `"{year}/{image_id}"`; val chosen by keyed hash of group, deterministic.

- [ ] **Step 1: Write failing tests**

```python
def test_difficult_objects_are_ignored():
    recs = parse_voc_xml(FIX / "000005.xml", year="2007")
    assert recs.ignore.tolist() == [False, True]

def test_voc_boxes_are_zero_based_continuous():
    rec = parse_voc_xml(FIX / "000005.xml", year="2007")
    assert rec.boxes[0].tolist() == [262.0, 210.0, 324.0, 339.0]   # xmin-1, ymin-1, xmax, ymax

def test_val_split_is_deterministic_and_disjoint():
    a, b = split_trainval(fake_ids(1000), 0.10, 0), split_trainval(fake_ids(1000), 0.10, 0)
    assert a == b and set(a[0]).isdisjoint(a[1]) and 80 <= len(a[1]) <= 120
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** `parse_voc_xml(path, year) -> Record` (VOC 1-based inclusive pixel coords → continuous: `xmin-1, ymin-1, xmax, ymax`), `split_trainval(ids, fraction, seed) -> (train_ids, val_ids)` using SHA-256 of `f"{seed}:{id}"`, and `load_voc`. **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): PASCAL VOC adapter`.

### Task 5: dacl10k adapter and near-duplicate screen

**Files:**
- Create: `cua/datasets/dacl10k.py`, `cua/dedup.py`; Modify: `cua/prepare.py` (add `dacl10k` subcommand)
- Test: `tests/cua/test_dacl10k.py`, `tests/cua/test_dedup.py`

**Interfaces:**
- Produces:
  - `DACL_DAMAGE = ("Crack", "ACrack", "Wetspot", "Efflorescence", "Rust", "Rockpocket", "Hollowareas", "Cavity", "Spalling", "Graffiti", "Weathering", "Restformwork", "ExposedRebars")` — candidate list, confirmed against the downloaded files in Step 1.
  - `parse_dacl_json(path: Path, classes: tuple[str, ...]) -> Record` — one box per polygon (min/max of its points, clipped to the image); non-damage labels dropped.
  - `dhash(rgb: np.ndarray) -> int` (64-bit difference hash of a 9×8 greyscale resize); `duplicate_groups(hashes: list[int], max_distance: int = 4) -> list[int]` — union-find group id per image (chunked vectorised XOR + popcount).
  - `cross_split_duplicates(a: list[int], b: list[int], max_distance: int = 4) -> int` — number of hash pairs across two splits within `max_distance`; `prepare voc` records train/val-vs-test counts in `gates.json` (official VOC splits are reported, not regrouped).
  - `load_dacl10k(root: Path, seed: int = 0) -> tuple[dict[str, list[Record]], tuple[str, ...]]` — pools labelled official splits, assigns `group` from `duplicate_groups`, splits groups 70/15/15 by keyed hash, then applies the class-exclusion gate and returns the retained classes.

- [ ] **Step 1: Inspect the real format (operator step, on CCAST after download).** Open one annotation JSON and the official split folders; record in `cua/datasets/DACL10K_FORMAT.md`: key names, label names (update `DACL_DAMAGE` if they differ), whether test labels are present, whether a bridge identifier exists in file names. If a bridge id exists, use it as `group` instead of `duplicate_groups` and note it in that file.
- [ ] **Step 2: Write failing tests** (fixture JSON written to mirror the format recorded in Step 1)

```python
def test_polygon_to_box_and_component_labels_dropped():
    rec = parse_dacl_json(FIX / "img1.json", DACL_DAMAGE)
    assert rec.boxes.tolist() == [[10.0, 20.0, 110.0, 60.0]]
    assert [DACL_DAMAGE[i] for i in rec.labels] == ["Rust"]          # "Bearing" (component) dropped

def test_dhash_groups_near_duplicates():
    img = noise_image(seed=0); shifted = brighten(img, 1.05)
    g = duplicate_groups([dhash(img), dhash(shifted), dhash(noise_image(seed=9))])
    assert g[0] == g[1] != g[2]

def test_class_exclusion_rule():
    stats = [{"count": 299, "large_fraction": 0.0}, {"count": 5000, "large_fraction": 0.6}, {"count": 400, "large_fraction": 0.1}]
    assert retained_classes(("a", "b", "c"), stats) == ("c",)
```

- [ ] **Step 3: Run** — FAIL. **Step 4: Implement**; `retained_classes(classes, stats, min_count=300, max_large=0.5) -> tuple[str, ...]`. Labels are re-indexed to the retained classes. **Step 5: Run** — PASS. **Step 6: Commit** `feat(cua): dacl10k adapter and near-duplicate screen`.

### Task 6: Keyed RNG and base geometry

**Files:**
- Create: `cua/augment/__init__.py`, `cua/augment/rng.py`, `cua/augment/base.py`, `cua/augment/boxes.py`
- Test: `tests/cua/test_base.py`

**Interfaces:**
- Consumes: `Sample`.
- Produces:
  - `keyed_rng(seed: int, epoch: int, index: int, stream: str) -> np.random.Generator` — `default_rng` seeded from SHA-256 of the four values; streams used: `"base"`, `"edit"`, `"mosaic"`.
  - `affine_sample(sample: Sample, matrix: np.ndarray) -> Sample | None` (in `boxes.py`) — warps rgb (bilinear, pad 114) and boxes (transform 4 corners → axis-aligned hull); returns `None` if any non-ignored box would extend beyond the canvas or collapse below 1 px. Shared by all geometric edits.
  - `base_geometry(sample: Sample, rng) -> Sample` — hflip p = 0.5, then scale ~ U(0.8, 1.2) about the canvas centre; on rejection retry up to 10 draws, then scale 1.0.

- [ ] **Step 1: Write failing tests**

```python
def test_same_key_same_draws_regardless_of_other_streams():
    a = base_geometry(toy_sample(), keyed_rng(0, 4, 17, "base"))
    _ = keyed_rng(0, 4, 17, "edit").random(100)                    # consuming another stream must not matter
    b = base_geometry(toy_sample(), keyed_rng(0, 4, 17, "base"))
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.boxes, b.boxes)

def test_hflip_box_formula():
    s = toy_sample(boxes=[[100., 50., 200., 150.]])
    out = affine_sample(s, hflip_matrix(640))
    assert out.boxes[0].tolist() == [440.0, 50.0, 540.0, 150.0]

def test_box_touching_edge_rejects_upscale():
    s = toy_sample(boxes=[[0., 0., 640., 640.]])
    assert affine_sample(s, scale_matrix(1.2, 640)) is None

def test_empty_sample_passes_base():
    out = base_geometry(toy_sample(boxes=[]), keyed_rng(0, 0, 0, "base"))
    assert out.boxes.shape == (0, 4)
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (`cv2` is not available; use `PIL.Image.transform` with `Image.AFFINE` and the inverse matrix). **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): keyed RNG and base geometry`.

### Task 7: Colour and image-quality edits

**Files:**
- Create: `cua/augment/pixel.py`
- Test: `tests/cua/test_pixel.py`

**Interfaces:**
- Produces: `Edit = Callable[[Sample, np.random.Generator], Sample]`; functions `brightness`, `contrast`, `saturation`, `hue`, `blur`, `noise`, `jpeg`, `sharpen`, each with the global-constraint strength range sampled uniformly from `rng`. Boxes, labels and ignore are returned unchanged (same array objects copied).

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.parametrize("edit", [brightness, contrast, saturation, hue, blur, noise, jpeg, sharpen])
def test_pixel_edits_keep_boxes_and_dtype(edit):
    s = toy_sample(boxes=[[10., 10., 50., 50.]]); out = edit(s, np.random.default_rng(0))
    assert out.rgb.dtype == np.uint8 and out.rgb.shape == s.rgb.shape
    assert np.array_equal(out.boxes, s.boxes) and not np.shares_memory(out.boxes, s.boxes)

def test_hue_leaves_grey_unchanged():
    grey = toy_sample(fill=128)
    assert np.array_equal(hue(grey, np.random.default_rng(0)).rgb, grey.rgb)

def test_brightness_range():
    s = toy_sample(fill=100)
    vals = {int(brightness(s, np.random.default_rng(i)).rgb[0, 0, 0]) for i in range(200)}
    assert min(vals) >= 70 and max(vals) <= 130
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** with PIL (`ImageEnhance` for brightness/contrast/saturation/sharpness, HSV shift for hue, `GaussianBlur(σ)`, JPEG via in-memory encode/decode) and NumPy for noise (`σ = 0.03 × 255`, clip, round). **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): colour and image-quality edits`.

### Task 8: Geometric edits and random erasing

**Files:**
- Create: `cua/augment/geometric.py`
- Test: `tests/cua/test_geometric.py`

**Interfaces:**
- Consumes: `affine_sample`.
- Produces: `translate`, `rotate`, `vflip`, `erase` (all `Edit`). Geometric edits retry up to 10 draws when `affine_sample` returns `None`, then return the input unchanged. `erase` fills one rectangle (area 2–10% of canvas, aspect ratio U(0.3, 3.3)) with per-pixel uniform noise, resampling (≤ 10 tries, then identity) when it would cover > 50% of any non-ignored box.

- [ ] **Step 1: Write failing tests**

```python
def test_rotate_boxes_stay_inside_or_identity():
    s = toy_sample(boxes=[[5., 5., 635., 635.]])
    out = rotate(s, np.random.default_rng(0))
    assert np.array_equal(out.rgb, s.rgb) and np.array_equal(out.boxes, s.boxes)   # any rotation would leave canvas

def test_rotate_angle_within_10_degrees():
    angles = [sampled_rotation_angle(np.random.default_rng(i)) for i in range(500)]
    assert max(abs(a) for a in angles) <= 10.0

def test_erase_never_covers_more_than_half_a_box():
    s = toy_sample(boxes=[[300., 300., 340., 340.]])
    for i in range(200):
        out = erase(s, np.random.default_rng(i))
        assert covered_fraction(out, s, box=0) <= 0.5

def test_geometric_edits_on_empty_sample():
    for edit in (translate, rotate, vflip, erase):
        assert edit(toy_sample(boxes=[]), np.random.default_rng(0)).boxes.shape == (0, 4)
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**; expose `sampled_rotation_angle(rng) -> float` for the test. **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): geometric edits and erasing`.

### Task 9: Mosaic, menu registry, validity and the arm transform

**Files:**
- Create: `cua/augment/mosaic.py`, `cua/augment/menu.py`
- Test: `tests/cua/test_menu.py`

**Interfaces:**
- Consumes: Tasks 6–8.
- Produces:
  - `mosaic(sample: Sample, others: list[Sample], rng) -> Sample` — centre (cx, cy) ~ U(160, 480)²; each of the 4 samples is resized to 320×320 and placed so its inner corner meets the centre; tiles cropped to the canvas; boxes go through `clip_and_filter`.
  - `clip_and_filter(boxes: np.ndarray, areas: np.ndarray, canvas: int = 640, min_visible: float = 0.30) -> np.ndarray` — clips to the canvas and returns the keep mask (visible area / pre-clip area ≥ `min_visible`).
  - `MENU_ORDER = ("translate", "rotate", "vflip", "brightness", "contrast", "saturation", "hue", "blur", "noise", "jpeg", "sharpen", "erase", "mosaic")`.
  - `INVALID = {"rdd": {"vflip"}, "voc": {"vflip"}, "dacl10k": set()}`; `valid_menu(dataset: str) -> tuple[str, ...]` (in `MENU_ORDER`).
  - `class ArmTransform(dataset: str, edit: str | None, seed: int, p: float = 0.5)` with `__call__(sample: Sample, epoch: int, index: int, fetch: Callable[[int], Sample], n_items: int) -> Sample` — applies `base_geometry` (stream `"base"`), then with probability `p` (stream `"edit"`) applies `edit`; for `mosaic`, draws 3 other indices from stream `"mosaic"` and calls `fetch(j)` (which returns base-transformed samples). Raises `ValueError` if `edit` not in `valid_menu(dataset)`.

- [ ] **Step 1: Write failing tests**

```python
def test_invalid_edit_rejected():
    with pytest.raises(ValueError):
        ArmTransform("rdd", "vflip", seed=0)

def test_base_identical_across_arms_when_edit_not_applied():
    outs = [ArmTransform("rdd", e, p=0.0, seed=0)(toy_sample(), 3, 11, toy_fetch, 100) for e in (None, "blur", "mosaic")]
    assert all(np.array_equal(o.rgb, outs[0].rgb) for o in outs)

def test_clip_and_filter_visibility_rule():
    boxes = np.array([[-100., 0., 20., 10.], [0., 0., 100., 100.]]); areas = np.array([1200., 10000.])
    assert clip_and_filter(boxes, areas).tolist() == [False, True]

def test_mosaic_on_empty_samples():
    t = ArmTransform("rdd", "mosaic", seed=0, p=1.0)
    assert t(toy_sample(boxes=[]), 4, 0, lambda j: toy_sample(boxes=[]), 10).boxes.shape == (0, 4)

def test_valid_menu_sizes():
    assert len(valid_menu("rdd")) == 12 and len(valid_menu("dacl10k")) == 13
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): mosaic, menu and arm transform`.

### Task 10: Dataset, collate and detector builders

**Files:**
- Create: `cua/data.py`, `cua/models.py`
- Test: `tests/cua/test_data_models.py`

**Interfaces:**
- Consumes: `Record`, `load_rgb`, `letterbox`, `ArmTransform`.
- Produces:
  - `class DetectionDataset(torch.utils.data.Dataset)(records, root, transform: ArmTransform | None, size=640)` with `set_epoch(epoch: int)`; `__getitem__(i) -> (image: FloatTensor[3,640,640] in [0,1], target: dict(boxes, labels (1-based int64), image_id, ignore))`; ignored boxes are removed from training targets.
  - `collate(batch) -> (list[Tensor], list[dict])`.
  - `build_detector(name: Literal["fasterrcnn", "retinanet"], num_classes: int, size: int = 640, pretrained: bool = True) -> torch.nn.Module` — torchvision `fasterrcnn_resnet50_fpn` / `retinanet_resnet50_fpn` with `weights="COCO_V1"` when `pretrained`, `trainable_backbone_layers=5`, `min_size=max_size=size`, score threshold 0.001, NMS 0.5, 100 detections; predictor/classification head replaced for `num_classes + 1` labels.
  - `trainable_head_parameters(model, name) -> list[Parameter]` — ROI box head + predictor (Faster R-CNN) or classification + regression heads (RetinaNet). Used by Plan 2's estimator; defined here so model code stays in one file.

- [ ] **Step 1: Write failing tests**

```python
def test_empty_target_tensor_shapes(tmp_path):
    ds = DetectionDataset([empty_record(tmp_path)], tmp_path, transform=None)
    _, t = ds[0]
    assert t["boxes"].shape == (0, 4) and t["labels"].dtype == torch.int64

@pytest.mark.parametrize("name", ["fasterrcnn", "retinanet"])
def test_detector_trains_one_step_on_cpu(name):
    m = build_detector(name, num_classes=4, size=320, pretrained=False)
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [{"boxes": torch.tensor([[10., 10., 100., 100.]]), "labels": torch.tensor([3])},
               {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}]
    loss = sum(m(images, targets).values()); loss.backward()
    assert torch.isfinite(loss)
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): dataset, collate and detectors`.

### Task 11: Evaluation and fast AP50 bootstrap

**Files:**
- Create: `cua/evaluate.py`, `cua/analysis/__init__.py`, `cua/analysis/ap50.py`
- Test: `tests/cua/test_evaluate.py`

**Interfaces:**
- Consumes: `Record`, `restore_boxes`.
- Produces:
  - `predict(model, dataset: DetectionDataset, device, batch_size=8) -> list[dict(image_index, boxes (orig coords), scores, labels 0-based)]`.
  - `coco_metrics(records, predictions, classes) -> dict(macro_ap50, ap50_95, ap75, ap_small, ap_medium, ap_large, per_class={name: {ap50, ap50_95, n_gt}})` — pycocotools; ignored objects as `iscrowd=1`; classes with 0 GT give `None` and are excluded from `macro_ap50`.
  - `image_matches(records, predictions, n_classes, iou=0.5, max_det=100) -> Matches` — per image and class: sorted scores, TP flags, ignored-match flags, GT counts (non-ignored), following COCO's greedy matching.
  - `ap50_from_matches(m: Matches, image_idx: np.ndarray) -> np.ndarray[n_classes]` (NaN for classes with 0 GT in the resample; 101-point interpolation).
  - `paired_bootstrap(ma: Matches, mb: Matches, n_boot=1000, seed=0) -> dict(diff, lo, hi)` — macro AP50 (b − a), same image resample for both.

- [ ] **Step 1: Write failing tests**

```python
def test_fast_ap50_matches_pycocotools():
    recs, preds = random_detection_problem(n_images=60, n_classes=3, seed=4, with_ignored=True)
    coco = coco_metrics(recs, preds, ("a", "b", "c"))
    fast = ap50_from_matches(image_matches(recs, preds, 3), np.arange(60))
    for k, name in enumerate("abc"):
        assert abs(fast[k] - coco["per_class"][name]["ap50"]) < 1e-9

def test_class_without_gt_is_none_and_excluded():
    recs, preds = problem_where_class_2_has_no_gt()
    out = coco_metrics(recs, preds, ("a", "b", "c"))
    assert out["per_class"]["c"]["ap50"] is None
    assert out["macro_ap50"] == pytest.approx(np.mean([out["per_class"][n]["ap50"] for n in "ab"]))

def test_bootstrap_of_identical_models_is_zero():
    recs, preds = random_detection_problem(40, 2, seed=1)
    m = image_matches(recs, preds, 2)
    r = paired_bootstrap(m, m, n_boot=200)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**; AP numbers are on the 0–100 scale everywhere. **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): evaluation and fast AP50 bootstrap`.

### Task 12: Trainer

**Files:**
- Create: `cua/train.py`
- Test: `tests/cua/test_train.py`

**Interfaces:**
- Consumes: Tasks 9–11.
- Produces:
  - `@dataclass(frozen=True) TrainConfig(run_id: str, dataset: str, detector: str, seed: int, epoch_lr: tuple[float, ...], edit: str | None, edit_p: float = 0.5, init_run: str | None = None, first_epoch: int = 1, lr_warmup_iters: int = 1000, base_lr: float = 0.0025, batch_size: int = 4, amp: bool = True, workers: int = 4)` with `digest() -> str` (SHA-256 of its JSON).
  - `full_schedule(epochs: int) -> tuple[float, ...]` — 1.0, then 0.1 from epoch `round(0.6E)+1`, 0.01 from `round(0.8E)+1`. `full_schedule(20)` = 12×1.0, 4×0.1, 4×0.01.
  - `train(cfg: TrainConfig, data_root: Path, runs_root: Path, resume: bool = False, deadline: float | None = None, max_train_images: int | None = None) -> int` — reads `data_root/<dataset>/` (jsonl + `source.json`), writes to `runs_root/<run_id>/`; returns 0 (complete) or 3 (stopped at deadline). `TrainConfig` holds no paths, so its digest is location-independent. Writes `config.json`, `history.jsonl` (one line per epoch: epoch, lr multiplier, train losses, val `coco_metrics`, seconds), `last.pt` and `best.pt` (by val macro AP50; includes model, optimizer, scaler, epoch, cfg digest), `val_predictions_last.json`, `val_predictions_best.json`, `COMPLETE.json`. Epoch numbering starts at `first_epoch` (Part A arms use 4–6 so their keyed draws continue the warm run). With `init_run`, loads model + optimizer from `runs_root/<init_run>/last.pt`. Warm-up applies only when `init_run is None`. Training uses `records` from `train.jsonl` (reference set excluded by construction).
  - Refuses to start if `out_dir` exists unless `resume`; refuses to resume if cfg digest differs or `COMPLETE.json` exists. Stops cleanly before an epoch if `deadline - now < 1.2 × last epoch duration`.

- [ ] **Step 1: Write failing tests** (CPU, `pretrained=False`, size 320, 6 synthetic images written by a fixture)

```python
def test_full_schedule_20():
    assert full_schedule(20) == (1.0,) * 12 + (0.1,) * 4 + (0.01,) * 4

# tiny_data: fixture returning (data_root, runs_root) with 6 synthetic images; tiny_cfg uses run_id "tiny"
def test_train_writes_history_and_complete(tiny_data):
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0)), *tiny_data) == 0
    assert [h["epoch"] for h in read_history(tiny_data[1] / "tiny")] == [1, 2]
    assert (tiny_data[1] / "tiny" / "COMPLETE.json").exists()

def test_refuses_overwrite_and_resume_of_complete(tiny_data):
    train(tiny_cfg(epoch_lr=(1.0,)), *tiny_data)
    with pytest.raises(RuntimeError): train(tiny_cfg(epoch_lr=(1.0,)), *tiny_data)
    with pytest.raises(RuntimeError): train(tiny_cfg(epoch_lr=(1.0,)), *tiny_data, resume=True)

def test_resume_rejects_changed_config(tiny_data):
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0)), *tiny_data, deadline=after_first_epoch()) == 3
    with pytest.raises(RuntimeError): train(tiny_cfg(epoch_lr=(1.0, 0.1)), *tiny_data, resume=True)

def test_deadline_stop_then_resume_continues(tiny_data):
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0, 1.0)), *tiny_data, deadline=after_first_epoch()) == 3
    assert train(tiny_cfg(epoch_lr=(1.0, 1.0, 1.0)), *tiny_data, resume=True) == 0
    assert [h["epoch"] for h in read_history(tiny_data[1] / "tiny")] == [1, 2, 3]
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement**; `after_first_epoch()` in the test patches `time.monotonic`; AMP is disabled automatically on CPU. **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): trainer with resume and deadline stop`.

### Task 13: Run specifications and CLI

**Files:**
- Create: `cua/experiments/__init__.py`, `cua/experiments/specs.py`, `cua/run.py`
- Test: `tests/cua/test_specs.py`

**Interfaces:**
- Consumes: `TrainConfig`, `full_schedule`, `valid_menu`.
- Produces:
  - `BUDGET_EPOCHS = {"rdd": 20, "voc": 20, "dacl10k": 30}`.
  - `budget_specs(dataset, detector, epochs: int | None = None) -> list[TrainConfig]` — one base-recipe run, seed 0, `full_schedule(epochs or BUDGET_EPOCHS[dataset])`; run id `budget-{dataset}-{detector}-e{epochs}`.
  - `part_a_specs(dataset, detector, seeds=(0, 1, 2)) -> list[TrainConfig]` — per seed: warm `A-{dataset}-{detector}-s{seed}-warm` (`epoch_lr=(1.0, 1.0, 1.0)`, edit None) and arms `A-{dataset}-{detector}-s{seed}-{edit or 'base'}` (`init_run`=warm id, `first_epoch=4`, `epoch_lr=(1.0, 1.0, 0.1)`) for `None` and every `valid_menu(dataset)` edit.
  - `proxy_specs(edits: list[str], epochs: int) -> list[TrainConfig]` — RDD, fasterrcnn, seed 0, `full_schedule(epochs)`, edits plus base; run id `proxy-rdd-fasterrcnn-{edit or 'base'}`.
  - CLI: `python -m cua.run --spec <run_id> --data-root <dir> --runs-root <dir> [--resume] [--deadline-seconds N] [--max-train-images N]` (looks the id up in all spec families; exit code passes through from `train`).

- [ ] **Step 1: Write failing tests**

```python
def test_part_a_rdd_counts_and_links():
    specs = part_a_specs("rdd", "fasterrcnn")
    assert len(specs) == 3 * (1 + 1 + 12)
    arm = next(s for s in specs if s.run_id == "A-rdd-fasterrcnn-s1-blur")
    assert arm.init_run == "A-rdd-fasterrcnn-s1-warm" and arm.first_epoch == 4 and arm.epoch_lr == (1.0, 1.0, 0.1)

def test_run_ids_unique_across_families():
    ids = [s.run_id for d in ("rdd", "voc", "dacl10k") for s in part_a_specs(d, "fasterrcnn") + budget_specs(d, "fasterrcnn")]
    assert len(ids) == len(set(ids))
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): run specifications and CLI`.

### Task 14: CCAST job generation

**Files:**
- Create: `cua/ccast/job_template.pbs`, `cua/ccast/make_jobs.py`, `cua/ccast/README.md`
- Test: `tests/cua/test_make_jobs.py`

**Interfaces:**
- Consumes: spec families from Task 13.
- Produces: `make_jobs(specs: list[TrainConfig], out_dir: Path, project: str = "/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research", smoke: bool = False) -> Path` (`smoke` adds `--max-train-images 64` and a 1-hour walltime) — writes one `<run_id>.pbs` per spec and `submit.sh`, which submits warm/budget runs first and arms with `-W depend=afterok:<warm job id>`. Walltime: 6 h for `A-*`, 24 h for `budget-*`/`proxy-*` (deadline = walltime − 20 min). Template keeps the existing resource line (`select=1:ncpus=4:mem=32gb:ngpus=1:glist=a10`, queue `gpus`, group `x-ccast-prj-aaboah`), activates `conda pytorch` + `.venvs/rdd_detection_v1`, runs `pytest -q tests/cua` once per job, then `python -m cua.run`; on exit 3 it prints the resume command and does not resubmit.
- CLI: `python -m cua.ccast.make_jobs --family part_a --dataset rdd --detector fasterrcnn --out jobs/A-rdd-fasterrcnn`.

- [ ] **Step 1: Write failing tests**

```python
def test_arm_jobs_depend_on_warm(tmp_path):
    submit = make_jobs(part_a_specs("rdd", "fasterrcnn", seeds=(0,)), tmp_path).read_text()
    warm_line = submit.index("A-rdd-fasterrcnn-s0-warm.pbs"); arm_line = submit.index("A-rdd-fasterrcnn-s0-blur.pbs")
    assert warm_line < arm_line and "depend=afterok:$WARM_s0" in submit

def test_walltime_and_deadline(tmp_path):
    make_jobs(budget_specs("rdd", "fasterrcnn"), tmp_path)
    job = (tmp_path / "budget-rdd-fasterrcnn-e20.pbs").read_text()
    assert "walltime=24:00:00" in job and "--deadline-seconds 85200" in job
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement.** **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): CCAST job generation`.

### Task 15: Part A analysis

**Files:**
- Create: `cua/analysis/part_a.py`
- Test: `tests/cua/test_part_a_analysis.py`

**Interfaces:**
- Consumes: `history.jsonl` from Task 12 runs.
- Produces:
  - `load_final(runs_root: Path, prefix: str) -> list[dict(dataset, detector, seed, edit, cls, ap50)]` — last epoch of each `A-*` arm (edit `"base"` for the base arm).
  - `deltas(rows) -> list[dict(dataset, detector, seed, edit, cls, delta)]` — arm − base, same dataset/detector/seed/class.
  - `interaction_test(deltas) -> dict(F, df1, df2, p)` — per dataset × detector, nested OLS: Δ ~ seed + edit + cls vs. + edit:cls; F-test via `scipy.stats.f`.
  - `edit_class_tests(deltas) -> list[dict(edit, F, p, p_holm)]` — per edit, Δ ~ seed + cls, F-test for `cls`; Holm across edits.
  - `budget_plateaued(curve: list[float], tol: float = 0.3, tail_frac: float = 0.2) -> bool` — `curve[-1] - curve[round((1 - tail_frac) * len(curve)) - 1] < tol`.
  - `proxy_agreement(short: list[dict], full: list[dict], n_boot: int = 2000, seed: int = 0) -> dict(rho, lo, hi)` — Spearman over (edit, cls) cells, bootstrap over cells.
  - `heatmap(deltas, out_png: Path)` — mean Δ per edit × class, annotated.
  - CLI: `python -m cua.analysis.part_a --runs-root <dir> --out <dir>` writes `deltas.csv`, `tests.json`, `heatmap_<dataset>_<detector>.png`.

- [ ] **Step 1: Write failing tests**

```python
def test_interaction_detected_when_classes_disagree():
    d = synthetic_deltas(edits=4, classes=4, seeds=3, conflict_edit=0, effect=3.0, noise=0.3, seed=0)
    assert interaction_test(d)["p"] < 0.01
    assert edit_class_tests(d)[0]["p_holm"] < 0.05

def test_no_interaction_under_null():
    d = synthetic_deltas(edits=4, classes=4, seeds=3, conflict_edit=None, effect=0.0, noise=0.3, seed=0)
    assert interaction_test(d)["p"] > 0.05

def test_budget_plateau_rule():
    assert budget_plateaued([10, 20, 30, 40, 41.0, 41.1, 41.2, 41.25, 41.3, 41.3])
    assert not budget_plateaued([10, 20, 30, 35, 38, 40, 42, 44, 46, 48.0])

def test_holm_is_monotone():
    p = [0.01, 0.04, 0.03]; adj = holm(p)
    assert adj == pytest.approx([0.03, 0.06, 0.06])
```

- [ ] **Step 2: Run** — FAIL. **Step 3: Implement** (`holm(pvalues) -> list[float]` is a module-level helper). **Step 4: Run** — PASS. **Step 5: Commit** `feat(cua): Part A analysis`.

### Task 16: Execution on CCAST (operator task)

No new library code; each step has a check. Results are written under `$PROJECT/runs/cua/` and copied back to `reports/cua_part_a/`.

- [ ] **Step 1: Environment.** On CCAST, install `scipy` into `.venvs/rdd_detection_v1` if missing; pre-download torchvision COCO weights for both detectors on the login node into `$TORCH_HOME`. Check: `python -c "import torchvision; torchvision.models.detection.retinanet_resnet50_fpn(weights='COCO_V1')"` succeeds offline on a compute node.
- [ ] **Step 2: Data.** Download VOC 2007 trainval/test and 2012 trainval and dacl10k (official source, record licence) into `$PROJECT/dataset/`. Complete Task 5 Step 1. Run `python -m cua.prepare {rdd,voc,dacl10k} ...`. Check: each `gates.json` shows co-occurrence ≥ 0.20 (else apply the spec §3 replacement rule), reference sets ≈ 1,000 images, dacl10k retained-class list recorded, VOC cross-split duplicate counts recorded.
- [ ] **Step 3: Smoke.** Generate the RDD seed-0 warm and blur-arm jobs with `make_jobs(..., smoke=True)` and submit them. Check: completes, `history.jsonl` has finite losses, measured seconds/epoch recorded in `reports/cua_part_a/timing.json`. Update the spec §12 estimates if timings differ by > 30%.
- [ ] **Step 4: Budget runs.** Submit `budget-{rdd,voc,dacl10k}-fasterrcnn`. Check with `budget_plateaued` on each val AP50 curve; if not plateaued, resubmit with `epochs = round(1.5 × epochs)`. Record accepted budgets in `reports/cua_part_a/budgets.json`.
- [ ] **Step 5: Part A, Faster R-CNN.** Submit `part_a` jobs for all three datasets. Check: `COMPLETE.json` count = Σ over datasets of 3 × (2 + |valid_menu(dataset)|) = 42 + 42 + 45 = 129.
- [ ] **Step 6: Proxy check.** From the RDD Faster R-CNN Part A heatmap choose 4 edits (best, worst, two closest to the median by mean |Δ|); submit `proxy_specs`. Check: `proxy_agreement` ρ written to `reports/cua_part_a/proxy.json`; if ρ < 0.5 mark short-run results "indicative" in the report (spec §7).
- [ ] **Step 7: Part A, RetinaNet** (priority 6 in spec §12; skip if budget is short). Same as Step 5 with `--detector retinanet`.
- [ ] **Step 8: Analysis and report.** Run `python -m cua.analysis.part_a`; write `reports/cua_part_a/RESULTS.md` with the heatmaps, interaction tests (H1) and proxy agreement. Commit the report and CSV/JSON outputs (not checkpoints).
