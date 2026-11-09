"""Torch dataset and collate function for CUA records."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from cua.augment.menu import ArmTransform
from cua.image import Sample, letterbox, load_rgb
from cua.records import Record


class DetectionDataset(Dataset):
    """Letterboxed detection samples with optional keyed augmentation.

    Target keys: boxes float32 (N,4), labels int64 (N,) 1-based (both exclude ignored boxes),
    image_id python int (the dataset index), ignore float32 (K,4) ignored boxes in canvas
    coordinates, and geometry (letterbox dict) only when transform is None (eval; augmented
    geometry is not invertible).
    """

    def __init__(self, records: list[Record], root: Path, transform: ArmTransform | None, size: int = 640):
        if transform is not None and size != 640:
            raise ValueError(f"Augmentation (mosaic canvas) is 640-only; got size={size} with a transform")
        self.records, self.root, self.transform, self.size = list(records), Path(root), transform, size
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    def _load(self, j: int) -> tuple[Sample, dict]:
        rec = self.records[j]
        return letterbox(load_rgb(self.root, rec), rec, self.size)

    def __getitem__(self, i: int):
        sample, geometry = self._load(i)
        if self.transform is not None:
            def fetch(j: int) -> Sample:
                return self.transform.base(self._load(j)[0], self.epoch, j)

            sample = self.transform(sample, self.epoch, i, fetch, len(self.records))
        image = torch.from_numpy(np.ascontiguousarray(sample.rgb)).permute(2, 0, 1).float().div(255.0).contiguous()
        ign = np.asarray(sample.ignore, dtype=bool)
        boxes = np.asarray(sample.boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(sample.labels, dtype=np.int64).reshape(-1)
        target = {
            "boxes": torch.from_numpy(boxes[~ign].copy()),
            "labels": torch.from_numpy(labels[~ign] + 1),
            "image_id": int(i),
            "ignore": torch.from_numpy(boxes[ign].copy()),
        }
        if self.transform is None:
            target["geometry"] = geometry
        return image, target


def collate(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
