"""Polygon sidecar for mask-based copy-paste on dacl10k (cua.augment.paste with masks).

`prepare` keeps only boxes. This rebuilds, for every record of a prepared split, the polygon rings of
each box from the FiftyOne export, with the same filtering as cua.datasets.dacl10k._record (damage
labels only, degenerate and empty-after-clipping polygons dropped, excluded classes dropped), and checks
that every rebuilt box equals the stored one.

python -m cua.datasets.dacl10k_masks --root <dir with samples.json> --data-dir <cua_data>/dacl10k
writes <data-dir>/train_masks.json: {"version": 1, "masks": {image: [[ring, ...] per box]}}, rings as
[[x, y], ...] in original-image pixels.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cua.datasets.dacl10k import DACL_DAMAGE, FIFTYONE_FIELD, _degenerate
from cua.records import Record, read_classes, read_records

MASKS_FILE = "train_masks.json"


def sample_polygons(sample: dict, retained: tuple[str, ...]) -> tuple[str, list[tuple[int, np.ndarray, list]]]:
    """(filepath, [(retained label index, clipped box, rings in pixels), ...]) in _record order."""
    width, height = int(sample["metadata"]["width"]), int(sample["metadata"]["height"])
    out = []
    for poly in (sample.get(FIFTYONE_FIELD) or {}).get("polylines") or []:
        label = poly["label"]
        if label not in DACL_DAMAGE:
            continue
        rings = [np.asarray(r, dtype=np.float64).reshape(-1, 2) * [width, height] for r in poly["points"]]
        pts = np.concatenate(rings) if rings else np.zeros((0, 2))
        if len(pts) == 0 or _degenerate(pts):
            continue
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        cx1, cx2 = (min(max(v, 0.0), float(width)) for v in (x1, x2))
        cy1, cy2 = (min(max(v, 0.0), float(height)) for v in (y1, y2))
        if cx2 <= cx1 or cy2 <= cy1 or label not in retained:
            continue
        out.append((retained.index(label), np.array([cx1, cy1, cx2, cy2]), [r for r in rings if len(r) >= 3]))
    return str(sample["filepath"]), out


def build_masks(samples: list[dict], records: list[Record], retained: tuple[str, ...]) -> dict[str, list]:
    """image -> per-box ring lists, aligned with each record's boxes. Raises on any mismatch."""
    by_path = {}
    for s in samples:
        if s.get("filepath") is not None:
            by_path[str(s["filepath"])] = s
    masks = {}
    for r in records:
        if r.image not in by_path:
            raise ValueError(f"{r.image}: not in samples.json")
        _, polys = sample_polygons(by_path[r.image], retained)
        if len(polys) != len(r.boxes):
            raise ValueError(f"{r.image}: {len(polys)} polygons for {len(r.boxes)} boxes")
        for (lab, box, _), rb, rl in zip(polys, r.boxes, r.labels):
            if lab != int(rl) or not np.allclose(box, rb, atol=1e-6):
                raise ValueError(f"{r.image}: polygon box {box.tolist()} / {lab} != record {rb.tolist()} / {rl}")
        masks[r.image] = [[np.round(ring, 2).tolist() for ring in rings] for _, _, rings in polys]
    return masks


def load_masks(path: Path, records: list[Record]) -> list[list[list[np.ndarray]]]:
    """Per record, per box, the rings as (n, 2) arrays (input to CopyPaste(masks=...))."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))["masks"]
    out = []
    for r in records:
        if r.image not in data or len(data[r.image]) != len(r.boxes):
            raise ValueError(f"{path}: no polygons, or a different box count, for {r.image}")
        out.append([[np.asarray(ring, dtype=np.float64).reshape(-1, 2) for ring in rings] for rings in data[r.image]])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.datasets.dacl10k_masks")
    ap.add_argument("--root", required=True, type=Path, help="directory holding samples.json")
    ap.add_argument("--data-dir", required=True, type=Path, help="prepared cua_data/dacl10k directory")
    a = ap.parse_args(argv)
    samples = json.loads((a.root / "samples.json").read_text(encoding="utf-8"))["samples"]
    retained = tuple(read_classes(a.data_dir / "classes.json"))
    records = read_records(a.data_dir / "train.jsonl")
    masks = build_masks(samples, records, retained)
    out = a.data_dir / MASKS_FILE
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps({"version": 1, "masks": masks}), encoding="utf-8")
    tmp.replace(out)
    n = sum(len(v) for v in masks.values())
    print(f"wrote {out}: {len(masks)} images, {n} polygons")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
