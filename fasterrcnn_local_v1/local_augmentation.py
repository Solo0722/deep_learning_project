"""One uniformly selected raster overlap-group union; shared appearance draws."""
import copy
from dataclasses import replace
from functools import lru_cache
import hashlib

import numpy as np

import shared_augmentation as shared

OPERATIONS = shared.OPERATIONS
LOCAL_VERSION = "fixed_local_v1"
TAPER_PIXELS = 2


def prepared_geometry(row, size):
    if type(size) is not int or size <= 0 or size % 32:
        raise ValueError("Invalid prepared image size")
    width, height = row["width"], row["height"]
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image dimensions")
    scale = min(size/width, size/height)
    rw, rh = max(1, round(width*scale)), max(1, round(height*scale))
    return {"original_width": width, "original_height": height, "resized_width": rw,
            "resized_height": rh, "pad_left": (size-rw)//2, "pad_top": (size-rh)//2, "size": size}


def prepared_boxes(row, size):
    geometry = prepared_geometry(row, size)
    boxes = np.array([t["box_xyxy"] for t in row["targets"]], dtype=np.float64).reshape(-1, 4)
    boxes[:, [0, 2]] = boxes[:, [0, 2]]*(geometry["resized_width"]/row["width"]) + geometry["pad_left"]
    boxes[:, [1, 3]] = boxes[:, [1, 3]]*(geometry["resized_height"]/row["height"]) + geometry["pad_top"]
    return boxes


def raster_rectangles(row, size):
    geometry = prepared_geometry(row, size)
    boxes = prepared_boxes(row, size)
    ids = [t["object_index"] for t in row["targets"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate object indices")
    if len(boxes) and (not np.isfinite(boxes).all() or not (boxes[:, 2:] > boxes[:, :2]).all()):
        raise ValueError("Invalid target geometry")
    lower = np.floor(boxes[:, :2]).astype(np.int64)
    upper = np.ceil(boxes[:, 2:]).astype(np.int64)
    start = np.array([geometry["pad_left"], geometry["pad_top"]])
    end = start + np.array([geometry["resized_width"], geometry["resized_height"]])
    lower, upper = np.maximum(lower, start), np.minimum(upper, end)
    if len(boxes) and not (upper > lower).all():
        raise ValueError("Target has no raster support")
    return tuple(sorted((int(i), *map(int, lo), *map(int, hi)) for i, lo, hi in zip(ids, lower, upper)))


def union_weights(rectangles):
    """Return bounds and alpha*2; the union is composed once, never per box."""
    if not rectangles:
        raise ValueError("Empty group")
    left, top = min(r[0] for r in rectangles), min(r[1] for r in rectangles)
    right, bottom = max(r[2] for r in rectangles), max(r[3] for r in rectangles)
    if right <= left or bottom <= top:
        raise ValueError("Empty union bounds")
    mask = np.zeros((bottom-top, right-left), dtype=bool)
    for x1, y1, x2, y2 in rectangles:
        if not (x1 < x2 and y1 < y2):
            raise ValueError("Invalid raster rectangle")
        mask[y1-top:y2-top, x1-left:x2-left] = True
    padded = np.pad(mask, 1, constant_values=False)
    interior = mask.copy()
    height, width = mask.shape
    for dy in range(3):
        for dx in range(3):
            interior &= padded[dy:dy+height, dx:dx+width]
    weights = mask.astype(np.uint8) + interior.astype(np.uint8)
    return (left, top, right, bottom), weights


@lru_cache(maxsize=32768)
def groups_from_rectangles(rectangles):
    # Cache compact geometry and counts, not image-sized masks.
    parent = list(range(len(rectangles)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i, a in enumerate(rectangles):
        for j in range(i):
            b = rectangles[j]
            if max(a[1], b[1]) < min(a[3], b[3]) and max(a[2], b[2]) < min(a[4], b[4]):
                parent[find(i)] = find(j)
    components = {}
    for i, rect in enumerate(rectangles):
        components.setdefault(find(i), []).append(rect)
    groups = []
    for records in components.values():
        records = sorted(records)
        ids = tuple(r[0] for r in records)
        regions = tuple(r[1:] for r in records)
        _, weights = union_weights(regions)
        groups.append((ids, regions, int(np.count_nonzero(weights)), int(weights.sum())))
    return tuple(sorted(groups))


def groups(row, size):
    return groups_from_rectangles(raster_rectangles(row, size))


def attach_local(row, selected, seed, epoch, size, group_index=None):
    result = copy.deepcopy(selected)
    result["local"] = None
    if result["operation"] == "identity":
        return result
    options = groups(row, size)
    if row["split"] != "train" or not options:
        raise ValueError("Local edit requires target-bearing training data")
    if group_index is None:
        key = shared.encoded([LOCAL_VERSION, seed, epoch, row["image"]])
        group_index = shared.choice_index(key, "group", len(options))
    if type(group_index) is not int or not 0 <= group_index < len(options):
        raise ValueError("Invalid group index")
    ids, regions, pixels, units = options[group_index]
    geometry = prepared_geometry(row, size)
    result["local"] = {"version": LOCAL_VERSION, "image_size": size, "group_index": group_index,
                       "group_count": len(options), "object_indices": list(ids),
                       "rectangles": [list(r) for r in regions], "union_pixels": pixels,
                       "alpha_units": units, "taper_pixels": TAPER_PIXELS,
                       "content_pixels": geometry["resized_width"]*geometry["resized_height"]}
    return result


def decision(row, policy, seed, epoch, size=640):
    return attach_local(row, shared.decision(row, policy, seed, epoch), seed, epoch, size)


def apply_example(example, selected):
    if selected["image"] != example.record["image"]:
        raise ValueError("Decision belongs to a different image")
    canvas = example.rgb.copy()
    stats = {"union_pixels": 0, "alpha_units": 0, "changed_pixels": 0}
    if selected["operation"] != "identity":
        if example.geometry is None or example.record["split"] != "train" or not len(example.boxes):
            raise ValueError("Local edit requires prepared target-bearing training data")
        local = selected.get("local")
        if not isinstance(local, dict):
            raise ValueError("Missing local group decision")
        size = example.geometry["size"]
        expected = attach_local(example.record, selected, 0, 0, size, local["group_index"])["local"]
        if local != expected or example.geometry != prepared_geometry(example.record, size):
            raise ValueError("Local decision/geometry differs from target metadata")
        if not np.array_equal(example.boxes, prepared_boxes(example.record, size)):
            raise ValueError("Prepared target boxes differ from metadata")
        geometry = example.geometry
        left, top = geometry["pad_left"], geometry["pad_top"]
        width, height = geometry["resized_width"], geometry["resized_height"]
        reference = example.rgb[top:top+height, left:left+width]
        candidate = shared.transform_content(reference, selected["operation"], selected["parameters"], selected["noise_seed"])
        (x1, y1, x2, y2), units = union_weights(local["rectangles"])
        original = example.rgb[y1:y2, x1:x2].astype(np.float64)
        transformed = candidate[y1-top:y2-top, x1-left:x2-left].astype(np.float64)
        composed = np.rint(original + (units[..., None]/2.)*(transformed-original)).clip(0, 255).astype(np.uint8)
        canvas[y1:y2, x1:x2] = composed
        stats = {"union_pixels": int(np.count_nonzero(units)), "alpha_units": int(units.sum()),
                 "changed_pixels": int(np.any(composed != example.rgb[y1:y2, x1:x2], axis=2).sum())}
        if any(stats[k] != local[k] for k in ("union_pixels", "alpha_units")):
            raise ValueError("Local mask area changed")
    elif selected.get("local") is not None:
        raise ValueError("Identity decision cannot select a local group")
    result = replace(example, rgb=canvas)
    result.augmentation_decision = copy.deepcopy(selected)
    result.augmentation_changed = bool(stats["changed_pixels"])
    result.augmentation_pixel_stats = stats
    return result


def empty_trace():
    return {**shared.empty_trace(), "shared_decision_sha256": "0"*64, "local_changed_pixels": 0}


def record_decision(trace, selected, changed=False, pixel_stats=None):
    shared.record_decision(trace, selected, changed)
    appearance = {k: v for k, v in selected.items() if k != "local"}
    trace["shared_decision_sha256"] = hashlib.sha256(bytes.fromhex(trace["shared_decision_sha256"])+shared.encoded(appearance)).hexdigest()
    if selected.get("local"):
        local = selected["local"]
        for key, amount in {"selected_groups": 1, "selected_boxes": len(local["object_indices"]),
                            "selected_multi_box_groups": int(len(local["object_indices"]) > 1),
                            "local_union_pixels": local["union_pixels"], "local_alpha_units": local["alpha_units"],
                            "local_reference_pixels": local["content_pixels"]}.items():
            trace["counts"][key] = trace["counts"].get(key, 0)+amount
    if pixel_stats:
        count = pixel_stats["changed_pixels"]
        if not 0 <= count <= pixel_stats["union_pixels"] or bool(count) != changed:
            raise ValueError("Invalid changed-pixel accounting")
        trace["local_changed_pixels"] += count


def verify_pairing(configuration, reference, shared_reference=None):
    allowed = {"augmentation", "augmentation_policy", "local_policy", "version", "purpose", "recipe_status", "comparator", "sources"}
    strip = lambda c: {k: v for k, v in c.items() if k not in allowed}
    if strip(configuration) != strip(reference):
        raise ValueError("Non-augmentation settings differ from comparator")
    if configuration["augmentation"] != "fixed_shared_local":
        raise ValueError("Unexpected local condition")
    shared.validate_policy(configuration["augmentation_policy"])
    local = configuration["local_policy"]
    if local["version"] != LOCAL_VERSION or local["taper_pixels"] != TAPER_PIXELS:
        raise ValueError("Unsupported local mask protocol")
    if shared_reference is not None:
        if strip(configuration) != strip(shared_reference):
            raise ValueError("Training settings differ from shared image-level control")
        strip_scope = lambda p: {k: v for k, v in p.items() if k != "scope"}
        if strip_scope(configuration["augmentation_policy"]) != strip_scope(shared_reference["augmentation_policy"]):
            raise ValueError("Appearance policy differs from image-level control")


def verify_shared_exposure(history, reference):
    for row in history:
        actual, expected = row["augmentation"], reference["epochs"][row["epoch"]-1]
        if actual["shared_decision_sha256"] != expected["decision_sha256"]:
            raise ValueError("Appearance decisions differ from the completed shared control")
        if any(actual["counts"].get(k, 0) != v for k, v in expected["counts"].items()):
            raise ValueError("Appearance exposure differs from shared control")
