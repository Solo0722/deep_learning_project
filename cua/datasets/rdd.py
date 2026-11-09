"""RDD adapter: loads the frozen detection splits into common Records."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from cua.records import Record

RDD_CLASSES = ("D00", "D10", "D20", "D40")


def _load_split(path: Path) -> list[Record]:
    out: list[Record] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            targets = row["targets"]
            boxes = np.asarray([t["box_xyxy"] for t in targets], dtype=np.float64).reshape(-1, 4)
            labels = np.asarray([t["class_index"] for t in targets], dtype=np.int64)
            out.append(
                Record(
                    image=row["image"],
                    width=int(row["width"]),
                    height=int(row["height"]),
                    boxes=boxes,
                    labels=labels,
                    ignore=np.zeros(len(targets), dtype=bool),
                    group=row["source_cluster_id"],
                    source=row["source"],
                )
            )
    return out


def load_rdd(splits_dir: Path) -> dict[str, list[Record]]:
    """Load train/val/test from detection_splits_v1-style JSONL files."""
    splits_dir = Path(splits_dir)
    return {s: _load_split(splits_dir / f"{s}.jsonl") for s in ("train", "val", "test")}
