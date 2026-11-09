"""Inference in original-image coordinates and COCO (pycocotools) metrics. AP values are 0-100."""
from __future__ import annotations

import contextlib
import io

import numpy as np
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from cua.analysis.ap50 import predictions_by_image, xyxy_to_xywh
from cua.data import DetectionDataset, collate
from cua.image import restore_boxes
from cua.records import Record


def predict(model, dataset: DetectionDataset, device, batch_size: int = 8, num_workers: int = 0) -> list[dict]:
    """Run the detector (moved to `device`); boxes restored to and clipped within the original image,
    labels 0-based (label 0 dropped). The model's train/eval mode is restored afterwards."""
    if dataset.transform is not None:
        raise ValueError("predict needs an un-augmented dataset (transform=None)")
    device = torch.device(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=num_workers)
    model.to(device)
    was_training = model.training
    model.eval()
    out = []
    try:
        with torch.no_grad():
            for images, targets in loader:
                outputs = model([im.to(device) for im in images])
                for t, o in zip(targets, outputs):
                    boxes = o["boxes"].detach().cpu().double().numpy().reshape(-1, 4)
                    scores = o["scores"].detach().cpu().double().numpy().reshape(-1)
                    labels = o["labels"].detach().cpu().numpy().astype(np.int64).reshape(-1)
                    keep = labels != 0  # index 0 is unused by both detectors
                    g = t["geometry"]
                    restored = restore_boxes(boxes[keep].reshape(-1, 4), g)
                    restored[:, [0, 2]] = np.clip(restored[:, [0, 2]], 0, g["original_width"])
                    restored[:, [1, 3]] = np.clip(restored[:, [1, 3]], 0, g["original_height"])
                    out.append({"image_index": int(t["image_id"]), "boxes": restored,
                                "scores": scores[keep].astype(np.float64), "labels": labels[keep] - 1})
    finally:
        model.train(was_training)
    return out


def _coco_inputs(records: list[Record], predictions, n_classes: int):
    images = [{"id": i + 1, "width": int(r.width), "height": int(r.height)} for i, r in enumerate(records)]
    cats = [{"id": k + 1, "name": str(k)} for k in range(n_classes)]
    gt_anns = []
    for i, r in enumerate(records):
        if len(r.labels) and (r.labels.min() < 0 or r.labels.max() >= n_classes):
            raise ValueError(f"GT label outside [0, {n_classes}) in image {i}")
        for box, lab, ign in zip(xyxy_to_xywh(r.boxes).tolist(), r.labels.tolist(), r.ignore.tolist()):
            gt_anns.append({"id": len(gt_anns) + 1, "image_id": i + 1, "category_id": lab + 1,
                            "bbox": box, "area": box[2] * box[3], "iscrowd": int(ign)})
    dt_anns = []
    for i, p in enumerate(predictions_by_image(predictions, len(records))):
        if p is None:
            continue
        labels = np.asarray(p["labels"], dtype=np.int64).reshape(-1)
        if len(labels) and (labels.min() < 0 or labels.max() >= n_classes):
            raise ValueError(f"predicted label outside [0, {n_classes}) in image {i}")
        scores = np.asarray(p["scores"], dtype=np.float64).reshape(-1).tolist()
        for box, lab, s in zip(xyxy_to_xywh(p["boxes"]).tolist(), labels.tolist(), scores):
            dt_anns.append({"image_id": i + 1, "category_id": lab + 1, "bbox": box, "score": s})
    return images, cats, gt_anns, dt_anns


def _mean_valid(s) -> float | None:
    v = s[s > -1]
    return float(np.mean(v)) * 100 if v.size else None


def coco_metrics(records: list[Record], predictions, classes) -> dict:
    """pycocotools bbox metrics; ignored objects are iscrowd=1; classes with 0 GT give None."""
    n_classes = len(classes)
    images, cats, gt_anns, dt_anns = _coco_inputs(records, predictions, n_classes)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = {"images": images, "categories": cats, "annotations": gt_anns}
        gt.createIndex()
        if dt_anns:
            dt = gt.loadRes(dt_anns)
        else:  # loadRes cannot take an empty list
            dt = COCO()
            dt.dataset = {"images": list(images), "categories": list(cats), "annotations": []}
            dt.createIndex()
        ev = COCOeval(gt, dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
    p = ev.params
    prec = ev.eval["precision"]  # [T, R, K, A, M]
    t50 = np.where(p.iouThrs == 0.5)[0]
    t75 = np.where(p.iouThrs == 0.75)[0]
    a_idx = {lbl: i for i, lbl in enumerate(p.areaRngLbl)}
    m100 = p.maxDets.index(100)
    per_class = {}
    for k, name in enumerate(classes):
        n_gt = sum(int(((r.labels == k) & ~r.ignore).sum()) for r in records)
        per_class[name] = {
            "ap50": _mean_valid(prec[t50, :, k, a_idx["all"], m100]),
            "ap50_95": _mean_valid(prec[:, :, k, a_idx["all"], m100]),
            "n_gt": n_gt,
        }
    valid = [v["ap50"] for v in per_class.values() if v["n_gt"] >= 1]
    return {
        "macro_ap50": float(np.mean(valid)) if valid else None,
        "ap50_95": _mean_valid(prec[:, :, :, a_idx["all"], m100]),
        "ap75": _mean_valid(prec[t75, :, :, a_idx["all"], m100]),
        "ap_small": _mean_valid(prec[:, :, :, a_idx["small"], m100]),
        "ap_medium": _mean_valid(prec[:, :, :, a_idx["medium"], m100]),
        "ap_large": _mean_valid(prec[:, :, :, a_idx["large"], m100]),
        "per_class": per_class,
    }
