"""Whole-image geometry on the prepared square canvas; annotation-safe fallback.

Boxes use continuous pixel edges. All geometry is axis aligned: reflection,
uniform scaling and translation. No rotation or change of class labels.
"""
import copy
from itertools import product

import numpy as np
from PIL import Image

from controller import digest


class InvalidAnnotation(ValueError):
    pass


def bank(settings):
    n = settings["size"]
    result = []
    for flip, m, tx, ty in product(settings["horizontal_flip"], settings["resized_canvas_sizes"],
                                  settings["translations_pixels"], settings["translations_pixels"]):
        result.append({"flip": flip, "resized_size": m, "tx": tx, "ty": ty,
                       "scale": m/n, "left": (n-m)//2+tx, "top": (n-m)//2+ty})
    if len(result) != settings["bank_size"] or len({digest(v) for v in result}) != len(result):
        raise ValueError("Invalid geometry bank")
    return result


def prepared_annotations(row, size):
    """Convert targets AND non-target annotations; retain all original metadata."""
    width, height = row["width"], row["height"]
    scale = min(size/width, size/height)
    rw, rh = max(1, round(width*scale)), max(1, round(height*scale))
    left, top = (size-rw)//2, (size-rh)//2
    result = []
    for kind, items, coordinate, identity in (("target", row["targets"], "box_xyxy", "object_index"),
                                               ("other", row["other_annotations"], "box", "index")):
        for item in items:
            box = np.asarray(item[coordinate], dtype=np.float64)
            if (box.shape != (4,) or not np.isfinite(box).all() or not (box[2:] > box[:2]).all()
                    or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height):
                raise InvalidAnnotation(f"Invalid {kind} annotation {item.get(identity)} in {row['image']}")
            transformed = box * np.array([rw/width, rh/height]*2) + np.array([left, top]*2)
            result.append({"kind": kind, "id": item[identity], "original": copy.deepcopy(item),
                           "box_xyxy": transformed.tolist()})
    if len({r["id"] for r in result}) != len(result):
        raise InvalidAnnotation("Repeated original XML object identity")
    return result


def forward_boxes(boxes, candidate, size):
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    if candidate["flip"]:
        boxes[:, [0, 2]] = size - boxes[:, [2, 0]]
    return boxes * candidate["scale"] + np.array([candidate["left"], candidate["top"]]*2)


def inverse_boxes(boxes, candidate, size):
    boxes = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
             - np.array([candidate["left"], candidate["top"]]*2))/candidate["scale"]
    if candidate["flip"]:
        boxes[:, [0, 2]] = size - boxes[:, [2, 0]]
    return boxes


def evaluate(annotations, candidate, size):
    boxes = forward_boxes([r["box_xyxy"] for r in annotations], candidate, size)
    outside = ((boxes[:, :2] < 0).any(axis=1) | (boxes[:, 2:] > size).any(axis=1))
    return {"accepted": not bool(outside.any()), "reason": "cropped_annotation" if outside.any() else "accepted",
            "cropped_ids": [r["id"] for r, bad in zip(annotations, outside) if bad], "boxes": boxes}


def apply(rgb, annotations, candidate):
    """Return pixels, annotations and decision; a rejected draw is exact identity."""
    n = rgb.shape[0]
    if rgb.shape != (n, n, 3) or rgb.dtype != np.uint8:
        raise ValueError("Expected square RGB uint8 prepared image")
    verdict = evaluate(annotations, candidate, n)
    if not verdict["accepted"]:
        return rgb.copy(), copy.deepcopy(annotations), verdict
    original = rgb[:, ::-1].copy() if candidate["flip"] else rgb
    m = candidate["resized_size"]
    if m != n:
        with Image.fromarray(original) as im:
            with im.resize((m, m), Image.Resampling.BILINEAR) as resized:
                original = np.array(resized)
    left, top = candidate["left"], candidate["top"]
    canvas = np.full_like(rgb, 114)
    x1, y1, x2, y2 = max(left, 0), max(top, 0), min(left+m, n), min(top+m, n)
    canvas[y1:y2, x1:x2] = original[y1-top:y2-top, x1-left:x2-left]
    updated = copy.deepcopy(annotations)
    for item, box in zip(updated, verdict["boxes"]):
        item["box_xyxy"] = box.tolist()
    return canvas, updated, verdict


def chosen_candidate(row, settings, seed, epoch):
    if row["split"] != "train":
        raise ValueError("Geometry sampling is training only")
    options = bank(settings)
    # Rejection sampling avoids modulo bias; no re-draw after annotation rejection.
    for counter in range(1024):
        word = int(digest(["v2_geometry", seed, epoch, row["image"], counter]), 16)
        bound = 2**256 - 2**256 % len(options)
        if word < bound:
            return options[word % len(options)]
    raise RuntimeError("Failed deterministic integer sampling")
