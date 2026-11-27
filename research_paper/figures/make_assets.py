"""Render real dataset images for the paper figures (run on CCAST, where the data lives).

python research_paper/figures/make_assets.py --data-root $P/cua_data --out assets
Writes, per dataset (rdd, dacl10k, voc), a few multi-class training images and every valid edit of
each, as rendered by the training code (cua.augment), plus copy-paste examples (box rectangle vs
polygon mask, same plan) and per-class object crops from the dacl10k reference set. Images are saved
without annotation; boxes and labels go to meta.json so figures draw them locally.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from cua.augment.menu import EDITS, valid_menu
from cua.augment.mosaic import mosaic
from cua.augment.paste import CopyPaste, content_of
from cua.augment.rng import keyed_rng
from cua.datasets.dacl10k_masks import MASKS_FILE, load_masks
from cua.image import letterbox, load_rgb
from cua.records import read_classes, read_records

N_CANDIDATES = 4
DRAWS = 2


def _root(ds_dir: Path) -> Path:
    r = Path(json.loads((ds_dir / "source.json").read_text(encoding="utf-8"))["image_root"])
    return r if r.is_absolute() else ds_dir / r


def _key(rec) -> str:
    return hashlib.sha256(rec.image.encode()).hexdigest()


def _candidates(recs, size=640):
    """Training images with 2-3 classes, 2-6 boxes, every box at least 40 px on the canvas."""
    out = []
    for i, r in enumerate(recs):
        labs = set(int(x) for x in r.labels)
        if not (2 <= len(labs) <= 3 and 2 <= len(r.boxes) <= 6) or r.ignore.any():
            continue
        s = min(size / r.width, size / r.height)
        wh = (r.boxes[:, 2:] - r.boxes[:, :2]) * s
        if wh.min() < 40 or (wh[:, 0] * wh[:, 1]).max() > 0.5 * size * size:
            continue
        out.append(i)
    return sorted(out, key=lambda i: _key(recs[i]))[:N_CANDIDATES]


def _save(sample, path: Path, content, meta: dict, name: str, extra=None):
    """Save the content region (letterbox padding cropped) and record boxes in crop coordinates."""
    l, t, w, h = content.left, content.top, content.width, content.height
    Image.fromarray(np.ascontiguousarray(sample.rgb[t:t + h, l:l + w])).save(path, quality=92)
    b = np.asarray(sample.boxes, dtype=float).reshape(-1, 4) - [l, t, l, t]
    meta[name] = {"file": path.name, "boxes": np.round(b, 1).tolist(),
                  "labels": [int(x) for x in sample.labels], **(extra or {})}


def edits_for(ds: str, data_root: Path, out: Path) -> dict:
    d = data_root / ds
    classes, recs, root = read_classes(d / "classes.json"), read_records(d / "train.jsonl"), _root(d)
    load = lambda j: letterbox(load_rgb(root, recs[j]), recs[j], 640)[0]  # noqa: E731
    meta = {"classes": list(classes), "images": {}}
    for k, i in enumerate(_candidates(recs)):
        sub = out / ds / f"img{k}"
        sub.mkdir(parents=True, exist_ok=True)
        m = {"record": recs[i].image}
        c = content_of(recs[i], 640)
        sample = load(i)
        _save(sample, sub / "orig.jpg", c, m, "orig")
        for edit in valid_menu(ds):
            for draw in range(DRAWS):
                rng = keyed_rng(0, 1, 1000 * draw + i, "edit")
                if edit == "mosaic":
                    mr = keyed_rng(0, 1, 1000 * draw + i, "mosaic")
                    others = [load(int(mr.integers(0, len(recs)))) for _ in range(3)]
                    res = mosaic(sample, others, rng)
                    full = type(c)(0, 0, 640, 640)
                    _save(res, sub / f"{edit}_{draw}.jpg", full, m, f"{edit}_{draw}")
                else:
                    _save(EDITS[edit](sample, rng), sub / f"{edit}_{draw}.jpg", c, m, f"{edit}_{draw}")
        meta["images"][f"img{k}"] = m
    return meta


def paste_examples(ds: str, data_root: Path, out: Path, n: int = 6) -> dict:
    d = data_root / ds
    classes, recs, root = read_classes(d / "classes.json"), read_records(d / "train.jsonl"), _root(d)
    load = lambda j: letterbox(load_rgb(root, recs[j]), recs[j], 640)[0]  # noqa: E731
    region = "lower" if ds == "rdd" else "full"
    masks = load_masks(d / MASKS_FILE, recs) if (d / MASKS_FILE).exists() else None
    rect = CopyPaste(recs, len(classes), "balanced", 0, p=1.0, region=region)
    mask = CopyPaste(recs, len(classes), "balanced", 0, p=1.0, region=region, masks=masks) if masks else None
    meta = {"classes": list(classes), "examples": {}}
    sub = out / f"paste_{ds}"
    sub.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    done = 0
    for i in rng.permutation(len(recs)):
        i = int(i)
        if len(recs[i].boxes) == 0 or len(recs[i].boxes) > 4:
            continue
        s = load(i)
        r_out, r_p = rect(s, 1, i, load)
        if not r_p:
            continue
        m = {"record": recs[i].image}
        c = content_of(recs[i], 640)
        _save(s, sub / f"ex{done}_orig.jpg", c, m, "orig")
        _save(r_out, sub / f"ex{done}_rect.jpg", c, m, "rect", {"n_pasted": len(r_p)})
        if mask is not None:
            m_out, m_p = mask(s, 1, i, load)
            _save(m_out, sub / f"ex{done}_mask.jpg", c, m, "mask", {"n_pasted": len(m_p)})
        meta["examples"][f"ex{done}"] = m
        done += 1
        if done >= n:
            break
    return meta


def reference_crops(data_root: Path, out: Path) -> dict:
    d = data_root / "dacl10k"
    classes, recs, root = read_classes(d / "classes.json"), read_records(d / "reference.jsonl"), _root(d)
    sub = out / "ref_dacl10k"
    sub.mkdir(parents=True, exist_ok=True)
    best = {}
    for i, r in enumerate(recs):
        for b, lab in zip(r.boxes, r.labels):
            w, h = b[2] - b[0], b[3] - b[1]
            score = min(w / r.width, h / r.height)  # favour objects that fill a fair part of the image
            if 0.15 < w / r.width < 0.7 and 0.15 < h / r.height < 0.7 and 0.5 < w / h < 2.0:
                if score > best.get(int(lab), (0,))[0]:
                    best[int(lab)] = (score, i, b)
    meta = {}
    for lab, (_, i, b) in sorted(best.items()):
        rgb = load_rgb(root, recs[i])
        x0, y0, x1, y1 = (int(v) for v in b)
        with Image.fromarray(rgb[y0:y1, x0:x1]) as im:
            im.thumbnail((256, 256))
            im.save(sub / f"{classes[lab]}.jpg", quality=92)
        meta[classes[lab]] = recs[i].image
    return meta


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    meta = {}
    for ds in ("dacl10k", "rdd", "voc"):
        meta[ds] = edits_for(ds, a.data_root, a.out)
        print(ds, "edits done", flush=True)
    for ds in ("dacl10k", "rdd"):
        meta[f"paste_{ds}"] = paste_examples(ds, a.data_root, a.out)
        print(ds, "paste done", flush=True)
    meta["ref_dacl10k"] = reference_crops(a.data_root, a.out)
    (a.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
