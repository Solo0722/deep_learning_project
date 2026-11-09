"""Fast per-class AP50 that reproduces pycocotools exactly, and a paired image bootstrap.

image_matches mirrors COCOeval.evaluateImg (IoU 0.5, area "all", maxDet 100) once per image and
class; ap50_from_matches mirrors COCOeval.accumulate for any (possibly duplicated) image resample.
AP values are on the 0-100 scale.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils

from cua.records import Record

# Exactly COCOeval Params.recThrs.
REC_THRS = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
_AREA_ALL = (0 ** 2, 1e5 ** 2)


@dataclass(frozen=True)
class Matches:
    """Per-class detection outcomes in CSR layout over images (image order).

    For class k, detections of image i are rows offsets[k][i]:offsets[k][i+1] of scores[k] (float64,
    sorted by descending score in COCO's stable order, truncated to maxDet), tp[k] (bool, matched to
    a non-ignored GT) and ignored[k] (bool, matched to an ignored GT). gt_count[i, k] counts the
    non-ignored GT boxes.
    """

    n_images: int
    n_classes: int
    scores: tuple
    tp: tuple
    ignored: tuple
    offsets: tuple
    gt_count: np.ndarray


def xyxy_to_xywh(boxes) -> np.ndarray:
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def predictions_by_image(predictions, n_images: int) -> list:
    """Index predictions by image_index; images without an entry get None."""
    out = [None] * n_images
    for p in predictions:
        i = int(p["image_index"])
        if not 0 <= i < n_images:
            raise ValueError(f"image_index {i} outside [0, {n_images})")
        if out[i] is not None:
            raise ValueError(f"duplicate predictions for image_index {i}")
        out[i] = p
    return out


def _match_one(gt_xywh, gt_ign, dt_xywh, iou_thr):
    """COCOeval.evaluateImg for one image/class/IoU threshold; gt is already sorted ignore-last."""
    D, G = len(dt_xywh), len(gt_xywh)
    matched = np.zeros(D, dtype=bool)
    dt_ig = np.zeros(D, dtype=bool)
    if D and G:
        crowd = [int(v) for v in gt_ign]  # ignored objects are iscrowd=1
        ious = np.asarray(mask_utils.iou(dt_xywh, gt_xywh, crowd)).tolist()
        gt_ig = [bool(v) for v in gt_ign]
        gtm = [False] * G
        thr0 = min(iou_thr, 1 - 1e-10)
        for d in range(D):
            iou, m = thr0, -1
            row = ious[d]
            for g in range(G):
                # already matched and not a crowd
                if gtm[g] and not crowd[g]:
                    continue
                # matched to a regular gt and reached the ignored gts
                if m > -1 and not gt_ig[m] and gt_ig[g]:
                    break
                if row[g] < iou:
                    continue
                iou, m = row[g], g
            if m == -1:
                continue
            dt_ig[d] = gt_ig[m]
            matched[d] = True
            gtm[m] = True
    area = dt_xywh[:, 2] * dt_xywh[:, 3]
    dt_ig |= ~matched & ((area < _AREA_ALL[0]) | (area > _AREA_ALL[1]))
    return matched & ~dt_ig, dt_ig


def image_matches(records: list[Record], predictions, n_classes: int, iou: float = 0.5,
                  max_det: int = 100) -> Matches:
    """COCO greedy matching per image and class (labels 0-based, boxes in original XYXY)."""
    n = len(records)
    by_image = predictions_by_image(predictions, n)
    per_k = [([], [], [], [0]) for _ in range(n_classes)]
    gt_count = np.zeros((n, n_classes), dtype=np.int64)
    for i, (rec, p) in enumerate(zip(records, by_image)):
        if len(rec.labels) and (rec.labels.min() < 0 or rec.labels.max() >= n_classes):
            raise ValueError(f"GT label outside [0, {n_classes}) in image {i}")
        g_xywh_all = xyxy_to_xywh(rec.boxes)
        g_area = g_xywh_all[:, 2] * g_xywh_all[:, 3]
        g_ign_all = rec.ignore | (g_area < _AREA_ALL[0]) | (g_area > _AREA_ALL[1])
        if p is None:
            d_xywh_all, d_scores_all, d_labels_all = np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
        else:
            d_xywh_all = xyxy_to_xywh(p["boxes"])
            d_scores_all = np.asarray(p["scores"], dtype=np.float64).reshape(-1)
            d_labels_all = np.asarray(p["labels"], dtype=np.int64).reshape(-1)
            if not len(d_xywh_all) == len(d_scores_all) == len(d_labels_all):
                raise ValueError(f"prediction arrays of image {i} differ in length")
            if len(d_labels_all) and (d_labels_all.min() < 0 or d_labels_all.max() >= n_classes):
                raise ValueError(f"predicted label outside [0, {n_classes}) in image {i}")
        for k in range(n_classes):
            gsel = rec.labels == k
            g_ign = g_ign_all[gsel]
            gt_count[i, k] = int((~g_ign).sum())
            gind = np.argsort(g_ign, kind="mergesort")
            dsel = d_labels_all == k
            scores = d_scores_all[dsel]
            dind = np.argsort(-scores, kind="mergesort")[:max_det]
            tp, ig = _match_one(g_xywh_all[gsel][gind], g_ign[gind], d_xywh_all[dsel][dind], iou)
            s_l, tp_l, ig_l, off = per_k[k]
            s_l.append(scores[dind])
            tp_l.append(tp)
            ig_l.append(ig)
            off.append(off[-1] + len(dind))

    def cat(parts, dtype):
        return np.concatenate(parts).astype(dtype) if parts else np.zeros(0, dtype=dtype)

    return Matches(
        n_images=n, n_classes=n_classes,
        scores=tuple(cat(v[0], np.float64) for v in per_k),
        tp=tuple(cat(v[1], bool) for v in per_k),
        ignored=tuple(cat(v[2], bool) for v in per_k),
        offsets=tuple(np.asarray(v[3], dtype=np.int64) for v in per_k),
        gt_count=gt_count,
    )


def ap50_from_matches(m: Matches, image_idx: np.ndarray) -> np.ndarray:
    """Per-class AP (0-100) over images image_idx (duplicates count as distinct images), as
    COCOeval.accumulate computes it; NaN for classes without non-ignored GT in the resample."""
    idx = np.asarray(image_idx, dtype=np.int64).reshape(-1)
    if len(idx) and (idx.min() < 0 or idx.max() >= m.n_images):
        raise ValueError("image index out of range")
    npig = m.gt_count[idx].sum(axis=0)
    out = np.full(m.n_classes, np.nan)
    R = len(REC_THRS)
    for k in range(m.n_classes):
        if npig[k] == 0:
            continue
        off = m.offsets[k]
        starts = off[idx]
        lens = off[idx + 1] - starts
        total = int(lens.sum())
        q = np.zeros(R)
        if total:
            # Concatenate image segments in resample order (as COCO concatenates evalImgs).
            pos = np.arange(total) + np.repeat(starts - (np.cumsum(lens) - lens), lens)
            keep = ~m.ignored[k][pos]
            # Ignored detections add to neither TP nor FP; dropping them keeps the stable order of
            # the rest and yields the same interpolated precision as COCO.
            s = m.scores[k][pos][keep]
            tp = m.tp[k][pos][keep]
            nd = len(s)
            if nd:
                tp = tp[np.argsort(-s, kind="mergesort")]
                tp_sum = np.cumsum(tp).astype(dtype=float)
                fp_sum = np.cumsum(~tp).astype(dtype=float)
                rc = tp_sum / npig[k]
                pr = tp_sum / (fp_sum + tp_sum + np.spacing(1))
                pr = np.maximum.accumulate(pr[::-1])[::-1]
                inds = np.searchsorted(rc, REC_THRS, side="left")
                valid = inds < nd
                q[valid] = pr[inds[valid]]
        out[k] = np.mean(q) * 100
    return out


def _macro(m: Matches, idx: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN resample
        return float(np.nanmean(ap50_from_matches(m, idx)))


def paired_bootstrap(ma: Matches, mb: Matches, n_boot: int = 1000, seed: int = 0) -> dict:
    """Paired image bootstrap of the macro AP50 difference (b - a): point diff and 95% percentile CI."""
    if ma.n_images != mb.n_images:
        raise ValueError(f"image counts differ: {ma.n_images} vs {mb.n_images}")
    if ma.n_classes != mb.n_classes:
        raise ValueError(f"class counts differ: {ma.n_classes} vs {mb.n_classes}")
    n = ma.n_images
    full = np.arange(n)
    diff = _macro(mb, full) - _macro(ma, full)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = _macro(mb, idx) - _macro(ma, idx)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff": float(diff), "lo": float(lo), "hi": float(hi)}
