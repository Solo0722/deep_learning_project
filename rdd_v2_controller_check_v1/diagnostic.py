"""Training-only panel planning, paired rendering and auditable retention analysis."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import controller
import geometry
import local_augmentation as local
import rdd_data


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def plan_panels(rows, cfg):
    """Hash-ranked class quotas, without inspecting pixels or detector outcomes.

    Allocate rare classes first; alternate A/B slots at each class quota index.
    A source cluster can contribute one image to the entire two-panel design.
    """
    require(len({r["image"] for r in rows}) == len(rows), "Duplicate training image")
    require(all(r["split"] == "train" for r in rows), "Non-training input")
    pools = {c: [r for r in rows if any(t["class_index"] == c for t in r["targets"])]
             for c in range(4)}
    class_order = sorted(pools, key=lambda c: (len({r["source_cluster_id"] for r in pools[c]}), c))
    panels, used = {"A": [], "B": []}, set()
    for c in class_order:
        candidates = iter(sorted(pools[c], key=lambda r: controller.digest(
            ["v2_panel", cfg["seed"], c, r["image"]])))
        for slot in range(cfg["panels"]["primary_groups_per_class"]):
            for name in ("A", "B"):
                row = next((r for r in candidates if r["source_cluster_id"] not in used), None)
                require(row is not None, f"Insufficient distinct clusters for class {c}")
                used.add(row["source_cluster_id"])
                options = local.groups(row, cfg["geometry"]["size"])
                lookup = {t["object_index"]: t["class_index"] for t in row["targets"]}
                eligible = [(i, group) for i, group in enumerate(options)
                            if any(lookup[j] == c for j in group[0])]
                index, group = min(eligible, key=lambda item: controller.digest(
                    ["v2_panel_group", cfg["seed"], row["image"], item[1][0]]))
                selected = {"panel": name, "slot": slot, "anchor_class": c,
                            "image": row["image"], "source": row["source"],
                            "source_cluster_id": row["source_cluster_id"],
                            "group_index": index, "object_indices": list(group[0]),
                            "group_classes": sorted({lookup[j] for j in group[0]}),
                            "target_positions": [j for j, t in enumerate(row["targets"])
                                                 if t["object_index"] in group[0]]}
                panels[name].append(selected)
    for name in panels:
        panels[name].sort(key=lambda p: controller.digest(["v2_panel_order", cfg["seed"], p["image"]]))
    return {"panels": panels, "class_allocation_order": class_order,
            "distinct_clusters": len(used),
            "sources": {k: dict(Counter(p["source"] for p in v)) for k, v in panels.items()},
            "selection": "Class quota is for the anchor group, not the number of detected objects; incidental classes are retained"}


def parameters(operation, level):
    require(operation in controller.OPERATIONS and level in (1, 2, 3), "Invalid appearance action")
    if operation == "gaussian_noise":
        return {"sigma_unit": (.01, .02, .03)[level-1]}
    if operation == "gaussian_blur":
        return {"sigma_pixels": (.4, .8, 1.2)[level-1]}
    low = operation in ("brightness_down", "contrast_down", "gamma_brighten")
    return {"gamma" if operation.startswith("gamma_") else "factor":
            ((.9, .8, .7) if low else (1.1, 1.2, 1.3))[level-1]}


def action_views(example, selection, cfg):
    """Existing local renderer FIRST, identical whole-image geometry SECOND.

    This preserves the tested content mean and union-mask implementation. The
    geometry-only reference and each appearance candidate share one affine map.
    Rejected geometry keeps that view's input, including its appearance edit.
    """
    row, size = example.record, cfg["geometry"]["size"]
    require(row["image"] == selection["image"] and row["split"] == "train", "Wrong selected image")
    require(example.geometry is not None and example.geometry["size"] == size, "Prepare image first")
    original = example.rgb.copy()
    annotations = geometry.prepared_annotations(row, size)
    candidate = geometry.chosen_candidate(row, cfg["geometry"], cfg["seed"], cfg["geometry_epoch_index"])
    reference, mapped, verdict = geometry.apply(original, annotations, candidate)
    targets = [a["box_xyxy"] for a in mapped if a["kind"] == "target"]
    metadata = {"candidate": candidate, "accepted": verdict["accepted"],
                "reason": verdict["reason"], "cropped_ids": verdict["cropped_ids"],
                "annotations": mapped, "target_boxes": targets,
                "clean_boxes": example.boxes.tolist(), "labels": example.labels.tolist()}
    yield "clean", original.copy(), {**metadata, "boxes": example.boxes.tolist()}
    yield "reference", reference.copy(), {**metadata, "boxes": targets}
    for op, level in controller.ACTIONS:
        for draw in range(2 if op == "gaussian_noise" else 1):
            noise_seed = int(controller.digest(["v2_probe_noise", cfg["seed"], row["image"], level, draw])[:16], 16)
            chosen = {"image": row["image"], "operation": op,
                      "parameters": parameters(op, level),
                      "noise_seed": noise_seed if op == "gaussian_noise" else None}
            chosen = local.attach_local(row, chosen, cfg["seed"], 0, size, selection["group_index"])
            require(chosen["local"]["object_indices"] == selection["object_indices"], "Selected group changed")
            edited = local.apply_example(example, chosen)
            canvas, updated, decision = geometry.apply(edited.rgb, annotations, candidate)
            require(updated == mapped and decision["accepted"] == verdict["accepted"], "Unpaired action geometry")
            require(np.array_equal(example.rgb, original), "Local renderer mutated reference pixels")
            yield f"{op}/{level}/{draw}", canvas, {
                "boxes": targets, "labels": example.labels.tolist(), "decision": chosen,
                "changed_pixels_before_geometry": edited.augmentation_pixel_stats["changed_pixels"],
                "changed_pixels_vs_reference": int(np.any(canvas != reference, axis=2).sum())}


def match_prediction(prediction, boxes, labels, settings):
    """Class-correct, confidence-ordered, one-to-one ordinary-detection matching.

    Saved detector labels and manifest labels are both zero-based foreground IDs.
    GT is used here only, after detector inference; never as detector input.
    """
    pb = np.asarray(prediction["boxes"], dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(prediction["scores"], dtype=np.float64)
    pl = np.asarray(prediction["labels"])
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    labels = np.asarray(labels)
    require(scores.shape == pl.shape == (len(pb),), "Prediction length mismatch")
    require(labels.shape == (len(boxes),), "GT length mismatch")
    require(np.isfinite(pb).all() and np.isfinite(scores).all(), "Nonfinite predictions")
    require(((scores >= 0) & (scores <= 1)).all(), "Invalid confidence")
    require(((pl >= 0) & (pl < 4) & (pl == np.floor(pl))).all(), "Invalid predicted classes")
    require((pb[:, 2:] > pb[:, :2]).all(), "Invalid predicted boxes")
    require(((pb >= 0) & (pb <= settings.get("size", 640))).all(), "Predicted boxes outside canvas")
    require(np.isfinite(boxes).all() and (boxes[:, 2:] > boxes[:, :2]).all(), "Invalid GT boxes")
    require(((labels >= 0) & (labels < 4) & (labels == np.floor(labels))).all(), "Invalid GT classes")
    order = sorted((i for i in range(len(scores)) if scores[i] >= settings["reference_confidence"]),
                   key=lambda i: (-scores[i], i))
    lo = np.maximum(pb[:, None, :2], boxes[None, :, :2])
    hi = np.minimum(pb[:, None, 2:], boxes[None, :, 2:])
    intersection = np.maximum(hi-lo, 0).prod(axis=2)
    union = (pb[:, 2:]-pb[:, :2]).prod(axis=1)[:, None] + (boxes[:, 2:]-boxes[:, :2]).prod(axis=1)[None, :] - intersection
    iou = intersection / np.maximum(union, np.finfo(np.float64).tiny)
    detected, overlaps, false_positives = [False]*len(boxes), [0.]*len(boxes), 0
    for i in order:
        choices = [j for j in range(len(boxes)) if not detected[j] and labels[j] == pl[i]
                   and iou[i, j] >= settings["reference_iou"]]
        if not choices:
            false_positives += 1
        else:
            j = max(choices, key=lambda j: (iou[i, j], -j))
            detected[j], overlaps[j] = True, float(iou[i, j])
    return {"detected": detected, "iou": overlaps, "false_positives": false_positives,
            "confident_predictions": len(order)}


def check_prediction_parity(first, second, tolerance):
    require(len(first) == len(second), "Repeat batch size changed")
    max_error = 0.
    for a, b in zip(first, second):
        require(a["labels"] == b["labels"], "Repeated prediction labels/order changed")
        for key in ("boxes", "scores"):
            x, y = np.asarray(a[key]), np.asarray(b[key])
            require(x.shape == y.shape, "Repeated prediction count changed")
            error = float(np.max(np.abs(x-y))) if x.size else 0.
            max_error = max(max_error, error)
            require(error <= tolerance, "Repeated prediction values changed beyond tolerance")
    return max_error


def analyze(records, rows, cfg):
    """Policies for A and B separately, never pool their data to rescue support."""
    by_name = {r["image"]: r for r in rows}
    all_records = records["A"] + records["B"]
    require(len({p["image"] for p in all_records}) == len(all_records), "Images reused across panels")
    require(len({p["source_cluster_id"] for p in all_records}) == len(all_records), "Source clusters reused across panels")
    output = {"panels": {}, "between_panels": {}}
    for name in ("A", "B"):
        panel = records[name]
        units = controller.units_from_panel(panel, by_name)
        policy = controller.build_policy(units, cfg["controller"])
        sensitivity = controller.stability(units, cfg["controller"], cfg["seed"], cfg["bootstrap_draws"])
        counts = Counter(controller.key(sorted({p["classes"][i] for i in p["target_positions"]})) for p in panel)
        routing = []
        for key, n in sorted(counts.items()):
            rule = policy["routed"][key]
            routing.append({"class_set": key, "selected_groups": n, "allowed_actions": sum(rule["allowed"]),
                            "borrowed_evidence": rule["borrowed_evidence"],
                            "tv_to_shared": controller.tv(rule["distribution"], policy["shared_distribution"])})
        per_class = {}
        for c in range(4):
            flags = [(p["clean"]["detected"][i], p["reference"]["detected"][i])
                     for p in panel for i in p["target_positions"] if p["classes"][i] == c]
            per_class[str(c)] = {"selected_objects": len(flags),
                                 "clean_detected": sum(a for a, b in flags),
                                 "geometry_detected": sum(b for a, b in flags),
                                 "clean_found_geometry_missed": sum(a and not b for a, b in flags),
                                 "clean_missed_geometry_found": sum(not a and b for a, b in flags)}
        output["panels"][name] = {"policy": policy, "sensitivity": sensitivity, "routing": routing,
                                  "geometry_recall_diagnostic": per_class,
                                  "geometry_acceptance": dict(Counter(p["geometry"]["reason"] for p in panel)),
                                  "group_weighted_tv_to_shared": sum(r["selected_groups"]*r["tv_to_shared"] for r in routing)/len(panel),
                                  "all_classes_supported": all(v["supported"] for v in policy["pooled"].values())}
    a, b = [output["panels"][n]["policy"] for n in ("A", "B")]
    for c in range(4):
        x, y = a["pooled"][str(c)], b["pooled"][str(c)]
        differences = [list(action) for action, p, q in zip(controller.ACTIONS, x["allowed"], y["allowed"]) if p != q]
        output["between_panels"][str(c)] = {"supported_in_both": x["supported"] and y["supported"],
            "differing_actions": differences,
            "action_agreement_if_supported": 1-len(differences)/24 if x["supported"] and y["supported"] else None,
            "distribution_tv": controller.tv(controller.balanced_distribution(x["allowed"]), controller.balanced_distribution(y["allowed"]))}
    output["shared_distribution_tv_between_panels"] = controller.tv(a["shared_distribution"], b["shared_distribution"])
    output["between_panel_routes"] = {s: {
        "distribution_tv": controller.tv(a["routed"][s]["distribution"], b["routed"][s]["distribution"]),
        "direct_supported_A": a["routed"][s]["direct_supported"],
        "direct_supported_B": b["routed"][s]["direct_supported"]} for s in a["routed"]}
    output["technical_status"] = "complete"
    output["scientific_status"] = "review_required; no automatic training authorization"
    output["support_in_both_panels"] = all(p["all_classes_supported"] for p in output["panels"].values())
    output["limits"] = ["One fixed epoch-1 detector; no optimizer or AP evaluation",
                        "Two cluster-disjoint, class-enriched panels are not a population-level confidence guarantee",
                        "Observed class differences and retained detections do not establish training utility",
                        "Missing direct mixed-set support remains borrowed evidence; no panel pooling or threshold relaxation"]
    return output


def verify_package(folder):
    listed = {}
    for line in (folder/"SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        path = rdd_data.inside(folder, name)
        require(name not in listed and file_hash(path) == digest, f"Package checksum failed: {name}")
        listed[name] = digest
    required = {p.relative_to(folder).as_posix() for p in folder.rglob("*")
                if p.is_file() and (p.suffix in (".py", ".json", ".sh", ".pbs", ".md"))}
    require(required == set(listed), "Unlisted or missing package source")
    return listed
