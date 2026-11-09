"""Common in-memory and on-disk record format shared by all CUA stages."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np


@dataclass(frozen=True, eq=False)
class Record:
    """One annotated image.

    boxes:  float64 (N, 4) original-image XYXY.
    labels: int64 (N,) 0-based class ids.
    ignore: bool (N,) True marks a box that is excluded from training targets.
    """

    image: str
    width: int
    height: int
    boxes: np.ndarray
    labels: np.ndarray
    ignore: np.ndarray
    group: str
    source: str

    def __post_init__(self) -> None:
        boxes = np.asarray(self.boxes, dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(self.labels, dtype=np.int64).reshape(-1)
        ignore = np.asarray(self.ignore, dtype=bool).reshape(-1)
        if not (len(boxes) == len(labels) == len(ignore)):
            raise ValueError("boxes, labels and ignore must have equal length")
        if not np.isfinite(boxes).all():
            raise ValueError("boxes must be finite (no NaN or inf)")
        if len(boxes):
            x1, y1, x2, y2 = boxes.T
            if np.any(x2 <= x1) or np.any(y2 <= y1):
                raise ValueError("every box must satisfy x2 > x1 and y2 > y1")
            if np.any(x1 < 0) or np.any(y1 < 0) or np.any(x2 > self.width) or np.any(y2 > self.height):
                raise ValueError("boxes must lie within the image bounds")
        _check_relative_posix(self.image)
        object.__setattr__(self, "boxes", boxes)
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "ignore", ignore)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Record):
            return NotImplemented
        return (
            self.image == other.image
            and self.width == other.width
            and self.height == other.height
            and self.group == other.group
            and self.source == other.source
            and np.array_equal(self.boxes, other.boxes)
            and np.array_equal(self.labels, other.labels)
            and np.array_equal(self.ignore, other.ignore)
        )

    def __hash__(self) -> int:
        return hash((self.image, self.width, self.height, self.group, self.source))

    def class_set(self) -> frozenset[int]:
        """Class ids present among non-ignored boxes."""
        return frozenset(int(c) for c in self.labels[~self.ignore])


def _check_relative_posix(image: str) -> None:
    if "\\" in image or ":" in image:
        raise ValueError(f"image path must be a relative POSIX path: {image!r}")
    path = PurePosixPath(image)
    if path.is_absolute() or ".." in path.parts or not image:
        raise ValueError(f"image path must be a relative POSIX path: {image!r}")


def write_records(path: Path, records: list[Record]) -> None:
    """Write records as JSONL, one record per line."""
    with Path(path).open("w", encoding="utf-8") as f:
        for r in records:
            row = {
                "image": r.image,
                "width": r.width,
                "height": r.height,
                "boxes": r.boxes.tolist(),
                "labels": r.labels.tolist(),
                "ignore": r.ignore.tolist(),
                "group": r.group,
                "source": r.source,
            }
            f.write(json.dumps(row) + "\n")


def read_records(path: Path) -> list[Record]:
    """Read records written by write_records."""
    out: list[Record] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            out.append(
                Record(
                    image=d["image"],
                    width=int(d["width"]),
                    height=int(d["height"]),
                    boxes=np.asarray(d["boxes"], dtype=np.float64).reshape(-1, 4),
                    labels=np.asarray(d["labels"], dtype=np.int64),
                    ignore=np.asarray(d["ignore"], dtype=bool),
                    group=d["group"],
                    source=d["source"],
                )
            )
    return out


def write_classes(path: Path, classes: tuple[str, ...]) -> None:
    """Write the ordered class-name tuple as a JSON list."""
    Path(path).write_text(json.dumps(list(classes)), encoding="utf-8")


def read_classes(path: Path) -> tuple[str, ...]:
    """Read the class-name tuple written by write_classes."""
    return tuple(json.loads(Path(path).read_text(encoding="utf-8")))
