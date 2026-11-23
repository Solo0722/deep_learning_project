"""Class-adaptive copy-paste (CAP).

Objects are cut from training images (their box rectangle) and pasted into other training images,
with a box and label added. Every arm pastes the same amount (probability `p` per image, 1..`max_objects`
objects); arms differ only in the class distribution the pasted objects are drawn from:

- "random":   a uniformly random object from the whole bank (follows class frequency; plain copy-paste)
- "balanced": class first, uniformly over classes (frequency-balanced), then an object of that class
- "adaptive": class first, with probabilities set from each class's current reference-split AP50
              (lower AP -> more pastes), refreshed by the trainer through `set_probs`

Pasting happens on the letterboxed image before base geometry and the policy edit, so pasted objects
go through the same augmentation as original ones. A paste is placed inside the image content (not the
letterbox padding) where it covers at most MAX_COVER of any existing or earlier pasted box, and is
blended with a linear feather of FEATHER pixels. Draws are keyed by (seed, epoch, index), so a run
is reproducible and resumable.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from cua.augment.rng import keyed_rng
from cua.image import Sample
from cua.records import Record

MODES = ("random", "balanced", "adaptive")
MIN_SIDE = 8.0  # canvas pixels; smaller objects are not banked
SCALE_RANGE = (0.8, 1.2)
MAX_COVER = 0.3
FEATHER = 4
PLACE_TRIES = 10


@dataclass(frozen=True)
class Content:
    """Non-padding region of a letterboxed canvas."""
    left: int
    top: int
    width: int
    height: int


def content_of(rec: Record, size: int) -> Content:
    scale = min(size / rec.width, size / rec.height)
    rw, rh = max(1, round(rec.width * scale)), max(1, round(rec.height * scale))
    return Content((size - rw) // 2, (size - rh) // 2, rw, rh)


def build_bank(records: list[Record], n_classes: int, size: int = 640,
               min_side: float = MIN_SIDE) -> list[np.ndarray]:
    """Per class, an (n, 2) int array of (record index, box index) of non-ignored boxes whose
    letterboxed sides are both >= min_side."""
    bank: list[list[tuple[int, int]]] = [[] for _ in range(n_classes)]
    for i, r in enumerate(records):
        scale = min(size / r.width, size / r.height)
        for b, (box, lab, ign) in enumerate(zip(r.boxes, r.labels, r.ignore)):
            if ign:
                continue
            if (box[2] - box[0]) * scale >= min_side and (box[3] - box[1]) * scale >= min_side:
                bank[int(lab)].append((i, b))
    return [np.asarray(x, dtype=np.int64).reshape(-1, 2) for x in bank]


def adaptive_probs(ap50: list[float | None], has_objects: np.ndarray, epsilon: float) -> np.ndarray:
    """p = (1-eps) softmax(-ap/tau) + eps/C over classes that have bank objects, tau = std of their
    AP (floored at 1 AP point). Classes with no AP (None) get the mean AP of the others."""
    has = np.asarray(has_objects, dtype=bool)
    p = np.zeros(len(ap50))
    if not has.any():
        return p
    known = [a for a, h in zip(ap50, has) if h and a is not None]
    fill = float(np.mean(known)) if known else 0.0
    a = np.array([fill if v is None else float(v) for v in ap50])[has]
    tau = max(float(np.std(a)), 1.0)
    e = np.exp(-(a - a.min()) / tau)
    q = (1.0 - epsilon) * e / e.sum() + epsilon / len(a)
    p[has] = q / q.sum()
    return p


class CopyPaste:
    """Pastes objects from `bank` into training samples. `load(j)` returns the letterboxed Sample of
    training record j (no augmentation)."""

    def __init__(self, records: list[Record], n_classes: int, mode: str, seed: int, *, p: float = 0.5,
                 max_objects: int = 3, epsilon: float = 0.2, size: int = 640):
        if mode not in MODES:
            raise ValueError(f"copy-paste mode must be one of {MODES}, got {mode!r}")
        if not 0.0 <= p <= 1.0 or max_objects < 1 or not 0.0 <= epsilon <= 1.0:
            raise ValueError("need 0 <= p <= 1, max_objects >= 1 and 0 <= epsilon <= 1")
        self.records, self.mode, self.seed = records, mode, int(seed)
        self.p, self.max_objects, self.epsilon, self.size = float(p), int(max_objects), float(epsilon), int(size)
        self.bank = build_bank(records, n_classes, size)
        self.has_objects = np.array([len(b) > 0 for b in self.bank])
        if not self.has_objects.any():
            raise ValueError("no object is large enough to paste")
        self.counts = np.array([len(b) for b in self.bank], dtype=np.float64)
        self.probs = self.class_probs_initial()

    def class_probs_initial(self) -> np.ndarray:
        """Class distribution of a pasted object before any AP is known."""
        if self.mode == "random":
            return self.counts / self.counts.sum()
        return self.has_objects / self.has_objects.sum()

    def set_probs(self, probs) -> None:
        if self.mode != "adaptive":
            raise ValueError("only the adaptive mode changes its class probabilities")
        probs = np.asarray(probs, dtype=np.float64)
        if probs.shape != self.counts.shape or np.any(probs < 0) or np.any(probs[~self.has_objects] > 0) \
                or not np.isclose(probs.sum(), 1.0):
            raise ValueError(f"invalid class probabilities {probs}")
        self.probs = probs

    def plan(self, epoch: int, index: int) -> list[tuple[int, int, int]]:
        """[(class, record index, box index), ...] to paste into item `index` (may be empty)."""
        rng = keyed_rng(self.seed, epoch, index, "paste")
        if rng.random() >= self.p:
            return []
        k = int(rng.integers(1, self.max_objects + 1))
        out = []
        for _ in range(k):
            c = int(rng.choice(len(self.probs), p=self.probs))
            j, b = self.bank[c][int(rng.integers(0, len(self.bank[c])))]
            if j == index and len(self.records) > 1:  # never paste an image into itself
                continue
            out.append((c, int(j), int(b)))
        return out

    def __call__(self, sample: Sample, epoch: int, index: int, load) -> tuple[Sample, list[int]]:
        """Returns the sample with pastes applied and the list of pasted classes."""
        todo = self.plan(epoch, index)
        if not todo:
            return sample, []
        rng = keyed_rng(self.seed, epoch, index, "paste_place")
        content = content_of(self.records[index], self.size)
        rgb = sample.rgb.copy()
        boxes = [np.asarray(sample.boxes, dtype=np.float64).reshape(-1, 4)]
        labels = [np.asarray(sample.labels, dtype=np.int64).reshape(-1)]
        ignore = [np.asarray(sample.ignore, dtype=bool).reshape(-1)]
        occupied = list(boxes[0])
        pasted: list[int] = []
        for c, j, b in todo:
            src = load(j)
            box = src.boxes[b]
            x0, y0 = int(np.floor(box[0])), int(np.floor(box[1]))
            x1, y1 = int(np.ceil(box[2])), int(np.ceil(box[3]))
            crop = src.rgb[y0:y1, x0:x1]
            if crop.shape[0] < 2 or crop.shape[1] < 2:
                continue
            s = float(rng.uniform(*SCALE_RANGE))
            w, h = crop.shape[1] * s, crop.shape[0] * s
            fit = min(1.0, 0.9 * content.width / w, 0.9 * content.height / h)
            w, h = max(2, int(round(w * fit))), max(2, int(round(h * fit)))
            placed = None
            for _ in range(PLACE_TRIES):
                px = content.left + int(rng.integers(0, content.width - w + 1))
                py = content.top + int(rng.integers(0, content.height - h + 1))
                cand = np.array([px, py, px + w, py + h], dtype=np.float64)
                if all(_cover(o, cand) <= MAX_COVER for o in occupied):
                    placed = cand
                    break
            if placed is None:
                continue
            with Image.fromarray(crop) as im, im.resize((w, h), Image.Resampling.BILINEAR) as r:
                patch = np.asarray(r, dtype=np.float32)
            px, py = int(placed[0]), int(placed[1])
            alpha = _feather(h, w, FEATHER)[..., None]
            region = rgb[py:py + h, px:px + w].astype(np.float32)
            rgb[py:py + h, px:px + w] = np.clip(alpha * patch + (1 - alpha) * region + 0.5, 0, 255).astype(np.uint8)
            boxes.append(placed[None])
            labels.append(np.array([c], dtype=np.int64))
            ignore.append(np.array([False]))
            occupied.append(placed)
            pasted.append(c)
        if not pasted:
            return sample, []
        return Sample(rgb, np.concatenate(boxes), np.concatenate(labels), np.concatenate(ignore)), pasted


def _cover(box: np.ndarray, cand: np.ndarray) -> float:
    """Fraction of `box` covered by `cand`."""
    iw = max(0.0, min(box[2], cand[2]) - max(box[0], cand[0]))
    ih = max(0.0, min(box[3], cand[3]) - max(box[1], cand[1]))
    area = max((box[2] - box[0]) * (box[3] - box[1]), 1e-9)
    return iw * ih / area


def _feather(h: int, w: int, f: int) -> np.ndarray:
    """Alpha mask, 1 inside and ramping linearly to ~0 over f pixels at each edge."""
    f = max(1, min(f, h // 2, w // 2))
    ys = np.minimum(np.arange(h), np.arange(h)[::-1]) + 1
    xs = np.minimum(np.arange(w), np.arange(w)[::-1]) + 1
    return np.minimum(np.minimum(ys[:, None], xs[None, :]) / f, 1.0).astype(np.float32)
