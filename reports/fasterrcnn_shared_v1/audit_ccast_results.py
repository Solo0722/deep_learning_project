"""Audit the returned fixed shared-augmentation baseline without a detector, raw pixels or test data.

All archive members and all 20 epochs' coverage/counts are checked. Direct COCO
evaluation independently recomputes the selected and final epochs. The original
archive, released package and split manifests are read-only.
"""
from collections import Counter
import contextlib
import csv
import gzip
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path, PurePosixPath
import sys
import tarfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "fasterrcnn_shared_seed42_v1_records_31a774d9401a.tar.gz"
EXPECTED_SHA = "28b28bee01376cc10fe0a60dca7ddfb9cd97ec9b548597fe316d94db164bfd7c"
PREFIX = "fasterrcnn_shared_seed42_v1"
CLASSES = ("D00", "D10", "D20", "D40")
METRICS = ("AP50_95", "AP50", "AP75", "AR100")


def verify_augmentation(rows, protocol, history, run):
    """Replay metadata-only decisions and their order; do not reopen image pixels."""
    import torch
    sys.path.insert(0, str(ROOT / "fasterrcnn_shared_v1"))
    import shared_augmentation as aug
    identity = json.loads((ROOT / "fasterrcnn_baseline_v1/protocol.json").read_text())
    aug.verify_pairing(protocol, identity)
    require(file_hash(ROOT / "fasterrcnn_baseline_v1/protocol.json") ==
            protocol["comparator"]["identity_protocol_sha256"], "Identity comparator protocol changed")
    plan = json.loads((OUT / "expected_exposure.json").read_text())
    total, changed, checks = Counter(), 0, []
    for index, row in enumerate(history):
        generator = torch.Generator().manual_seed(protocol["seed"] + 1000003*index)
        order = torch.randperm(len(rows), generator=generator).tolist()
        trace = aug.empty_trace()
        for ident in order:
            aug.record_decision(trace, aug.decision(rows[ident], protocol["augmentation_policy"], protocol["seed"], index))
        actual = row["augmentation"]
        require(actual["counts"] == trace["counts"] and actual["decision_sha256"] == trace["decision_sha256"],
                f"Recorded augmentation differs from replay: epoch {index+1}")
        require(actual["counts"] == plan["epochs"][index]["counts"], "Exposure differs from prospective plan")
        require(0 <= actual["changed_images"] <= actual["counts"].get("applied_images", 0), "Invalid changed-image count")
        if index == 0:
            require(actual["counts"].get("applied_images", 0) == 0, "Augmentation warmup violated")
        total.update(actual["counts"])
        changed += actual["changed_images"]
        checks.append({"epoch": index+1, **actual})
    require(dict(total) == plan["total"], "Total exposure differs from prospective plan")
    require(run["augmentation_applied_images"] == total["applied_images"], "Run exposure differs")
    report = {"status": "passed", "decision_replay_and_order": "exact_all_20_epochs", "total": dict(total),
              "reported_changed_images": changed, "epochs": checks, "warmup_epoch_unaugmented": True,
              "scope": "Decisions and counts verified from training metadata; changed-pixel counts are recorded by CCAST, not recomputed from raw images"}
    save("augmentation_verification.json", encoded(report))
    return {k: v for k, v in report.items() if k != "epochs"}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(name, content):
    path = OUT / name
    if path.exists():
        require(path.read_bytes() == content, f"Preserve different existing output: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(content)


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def frozen_data(protocol):
    rows = {}
    for split in ("train", "val"):
        path = ROOT / "detection_splits_v1" / f"{split}.jsonl"
        require(file_hash(path) == protocol["manifests"][split], f"Changed {split} manifest")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        require(len(rows[split]) == protocol["expected_rows"][split], f"Wrong {split} size")
        require(all(r["split"] == split for r in rows[split]), f"Wrong {split} assignments")
        require(len({r['image'] for r in rows[split]}) == len(rows[split]), "Duplicate image")
    for key in ("image", "source_cluster_id", "rgb_sha256"):
        require(not {r[key] for r in rows["train"]} & {r[key] for r in rows["val"]},
                f"Train/validation overlap: {key}")
    require(file_hash(ROOT / "detection_splits_v1/config.json") == protocol["split_config_sha256"],
            "Changed split configuration")
    images, annotations = [], []
    for image_id, row in enumerate(rows["val"]):
        images.append(dict(id=image_id, file_name=row["image"], width=row["width"], height=row["height"]))
        for obj in row["targets"]:
            index = obj["class_index"]
            require(CLASSES[index] == obj["label"], "Wrong class mapping")
            x1, y1, x2, y2 = obj["box_xyxy"]
            require(0 <= x1 < x2 <= row["width"] and 0 <= y1 < y2 <= row["height"], "Invalid target geometry")
            annotations.append(dict(id=len(annotations)+1, image_id=image_id, category_id=index+1,
                                    bbox=[x1, y1, x2-x1, y2-y1], area=(x2-x1)*(y2-y1), iscrowd=0))
    gt = dict(info={"description": "Frozen RDD detection val partition"}, images=images,
              annotations=annotations, categories=[dict(id=i+1, name=name) for i, name in enumerate(CLASSES)])
    return rows, gt


def inspect_predictions(predictions, gt, protocol):
    dimensions = {r["id"]: (r["width"], r["height"]) for r in gt["images"]}
    positive_ids = {a["image_id"] for a in gt["annotations"]}
    negatives = set(dimensions) - positive_ids
    counts, confident, by_class = Counter(), Counter(), Counter()
    for pred in predictions:
        ident, category = pred["image_id"], pred["category_id"]
        require(type(ident) is int and ident in dimensions and type(category) is int and 1 <= category <= 4,
                "Unknown prediction image/class")
        score = pred["score"]
        require(math.isfinite(score) and protocol["evaluation"]["score_threshold"] <= score <= 1,
                "Invalid prediction score")
        box = pred["bbox"]
        require(len(box) == 4 and all(math.isfinite(x) for x in box), "Nonfinite prediction box")
        x, y, w, h = box
        width, height = dimensions[ident]
        require(x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= width+1e-3 and y+h <= height+1e-3,
                "Prediction outside original image geometry")
        counts[ident] += 1
        by_class[category] += 1
        if score >= protocol["evaluation"]["negative_confidence"]:
            confident[ident] += 1
    require(max(counts.values(), default=0) <= protocol["evaluation"]["max_detections"], "Detection cap exceeded")
    negative_counts = {str(i): confident[i] for i in sorted(negatives)}
    return dict(prediction_boxes=len(predictions), per_image_counts=dict(counts),
                predictions_by_class={name: by_class[i+1] for i, name in enumerate(CLASSES)},
                no_target_images=len(negatives), no_target_predicted_boxes=negative_counts,
                no_target_images_with_predictions=sum(n > 0 for n in negative_counts.values()))


def read_archive(gt, protocol, selected_epochs):
    require(file_hash(ARCHIVE) == EXPECTED_SHA, "Archive checksum mismatch")
    contents, hashes, checks, retained_predictions = {}, {}, {}, {}
    with tarfile.open(ARCHIVE, "r|gz") as bundle:
        total_bytes = 0
        for member in bundle:
            name = PurePosixPath(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), "Unexpected member type")
            require(not name.is_absolute() and len(name.parts) > 1 and name.parts[0] == PREFIX,
                    "Unexpected archive prefix")
            require(all(p not in (".", "..") and ":" not in p and "\\" not in p for p in name.parts), "Unsafe path")
            relative = name.relative_to(PREFIX).as_posix()
            require(relative not in hashes and not relative.endswith((".pt", ".tmp")), "Duplicate/binary checkpoint in records")
            total_bytes += member.size
            require(total_bytes <= 600_000_000 and member.size <= 40_000_000, "Unexpected archive size")
            data = bundle.extractfile(member).read()
            hashes[relative] = hashlib.sha256(data).hexdigest()
            if relative.endswith("_val_predictions.json.gz"):
                epoch = int(relative.split("_")[1])
                predictions = json.loads(gzip.decompress(data))
                checks[epoch] = inspect_predictions(predictions, gt, protocol)
                if epoch in selected_epochs:
                    retained_predictions[epoch] = predictions
                print(f"Prediction coverage/geometry checked: epoch {epoch:02d}, {len(predictions):,} boxes", flush=True)
            else:
                contents[relative] = data
    sums = json.loads(contents["checksums.json"])
    require(set(sums) == set(hashes)-{"checksums.json"}, "Result checksum coverage mismatch")
    require(all(hashes[n] == expected for n, expected in sums.items()), "Result file checksum mismatch")
    return contents, hashes, checks, retained_predictions, total_bytes


def direct_coco(gt, predictions):
    """Use the direct COCO API, without importing released metric/training code."""
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = gt
        truth.createIndex()
        detections = truth.loadRes(predictions)
        evaluation = COCOeval(truth, detections, "bbox")
        evaluation.params.imgIds = [r["id"] for r in gt["images"]]
        evaluation.params.catIds = [1, 2, 3, 4]
        evaluation.params.maxDets = [1, 10, 100]
        evaluation.evaluate()
        evaluation.accumulate()
        evaluation.summarize()
    precision = evaluation.eval["precision"]

    def average(values):
        values = values[values >= 0]
        return float(values.mean()) if len(values) else None

    counts = Counter(a["category_id"] for a in gt["annotations"])
    return {"AP50_95": float(evaluation.stats[0]), "AP50": float(evaluation.stats[1]),
            "AP75": float(evaluation.stats[2]), "AR100": float(evaluation.stats[8]),
            "per_class": {name: {"AP50_95": average(precision[:, :, i, 0, -1]),
                                  "AP50": average(precision[0, :, i, 0, -1]),
                                  "target_count": counts[i+1]} for i, name in enumerate(CLASSES)}}


def main():
    protocol = json.loads((ROOT / "fasterrcnn_shared_v1/protocol.json").read_text())
    rows, gt = frozen_data(protocol)
    # Read the declared selection first; verify it against the entire history below.
    with tarfile.open(ARCHIVE, "r:gz") as bundle:
        declared_best = json.load(bundle.extractfile(f"{PREFIX}/best.json"))
    selected_epochs = sorted({declared_best["epoch"], protocol["epochs"]})
    contents, hashes, prediction_checks, predictions, total_bytes = read_archive(gt, protocol, selected_epochs)
    load = lambda name: json.loads(contents[name])
    run, context, history, best, last = [load(n) for n in ("run.json", "context.json", "history.json", "best.json", "last.json")]
    code_names = {n[5:] for n in contents if n.startswith("code/")}
    for name in code_names:
        require(contents["code/"+name] == (ROOT / "fasterrcnn_shared_v1" / name).read_bytes(), f"Changed released code: {name}")
    package_sums = dict(line.split("  ", 1)[::-1] for line in contents["code/SHA256SUMS"].decode().splitlines())
    require(code_names == set(package_sums) | {"SHA256SUMS"}, "Package checksum coverage mismatch")
    require(all(hashes["code/"+n] == h for n, h in package_sums.items()), "Package checksum mismatch")
    require(context["package_hashes"] == {n: hashes["code/"+n] for n in code_names}, "Execution code hashes differ")
    require(context["configuration"] == protocol and context["versions"] == protocol["versions"] and context["profile"] == "full",
            "Execution configuration/environment mismatch")
    require(load("validation_ground_truth.json") == gt, "Ground truth differs from frozen validation manifest")
    require(run["status"] == "complete" and run["profile"] == "full" and run["test_data_read"] is False and not run["resume"],
            "Unexpected run status/profile")
    require(load("attempts/31a774d9401a.json") == run, "Attempt/run records differ")
    epochs, train_n, val_n = protocol["epochs"], len(rows["train"]), len(rows["val"])
    steps_per_epoch = math.ceil(train_n/protocol["batch_size"])
    require(run["completed_epochs"] == epochs and run["optimizer_steps"] == epochs*steps_per_epoch
            and run["image_presentations"] == epochs*train_n, "Training budget mismatch")
    require(len(history) == epochs and set(prediction_checks) == set(range(1, epochs+1)), "Missing epoch")
    csv_rows = list(csv.DictReader(io.StringIO(contents["history.csv"].decode())))
    require(len(csv_rows) == epochs, "History CSV length differs")
    checkpoint_names = [n for n in contents if n.startswith("checkpoints/")]
    require(len(checkpoint_names) == epochs, "Checkpoint metadata count differs")
    epoch_checks = []
    for epoch, (row, csv_row) in enumerate(zip(history, csv_rows), 1):
        require(row["epoch"] == epoch and row["step"] == epoch*steps_per_epoch and row["images_seen"] == epoch*train_n
                and row["train_images"] == train_n and row["val_images"] == val_n, "Epoch coverage/budget differs")
        require(all(math.isfinite(x) and x >= 0 for x in row["training_loss"].values()), "Invalid epoch loss")
        record = load(f"epoch_{epoch:03d}_validation.json")
        values, clipping = record["metrics"], record["clipping"]
        require(values == row["validation"], "Epoch/history metrics differ")
        require(values["evaluation_images"] == val_n and values["target_boxes"] == len(gt["annotations"]), "Validation denominator differs")
        require([r["image_id"] for r in clipping] == list(range(val_n)), "Incomplete/reordered validation coverage")
        check = prediction_checks[epoch]
        for item in clipping:
            before, after, collapsed = [item[k] for k in ("before_clipping", "after_clipping", "collapsed_after_clipping")]
            require(0 <= after <= before <= 100 and after == before-collapsed, "Invalid clipping counts")
            require(after == check["per_image_counts"].get(item["image_id"], 0), "Saved prediction count differs from coverage record")
        for key in ("prediction_boxes", "no_target_images", "no_target_predicted_boxes", "no_target_images_with_predictions"):
            require(values[key] == check[key], f"Prediction-derived count differs: {key}")
        for key in METRICS:
            require(math.isfinite(values[key]) and 0 <= values[key] <= 1, "Invalid AP/AR")
            require(float(csv_row[f"val_{key}"]) == values[key], "History CSV metric differs")
        for key in ("AP50", "AP50_95"):
            require(abs(sum(values["per_class"][c][key] for c in CLASSES)/4-values[key]) < 1e-12,
                    "Class macro average differs")
            for name in CLASSES:
                require(float(csv_row[f"val_{name}_{key}"]) == values["per_class"][name][key], "Per-class CSV differs")
        for key in ("epoch", "step", "images_seen"):
            require(int(csv_row[key]) == row[key], "History CSV progress differs")
        require(float(csv_row["train_total_loss"]) == sum(row["training_loss"].values()), "History CSV loss differs")
        checkpoint = load("checkpoints/"+row["checkpoint_file"].replace(".pt", ".json"))
        require(checkpoint["file"] == row["checkpoint_file"] and checkpoint["completed_epochs"] == epoch
                and checkpoint["step"] == row["step"] and len(checkpoint["sha256"]) == 64 and checkpoint["bytes"] > 0,
                "Checkpoint metadata differs")
        epoch_checks.append({"epoch": epoch, "images": val_n, "predictions": values["prediction_boxes"],
                             "clipped_boxes": sum(x["boxes_clipped"] for x in clipping),
                             "collapsed_predictions": sum(x["collapsed_after_clipping"] for x in clipping)})
    chosen = max(history, key=lambda r: r["validation"]["AP50_95"])
    require(best["epoch"] == chosen["epoch"] and best["validation_AP50_95"] == chosen["validation"]["AP50_95"], "Best selection differs")
    best_meta = load("checkpoints/"+chosen["checkpoint_file"].replace(".pt", ".json"))
    require(all(best[k] == v for k, v in best_meta.items()), "Best checkpoint pointer differs")
    require(run["best"] == {k: best[k] for k in ("epoch", "checkpoint_file", "validation_AP50_95")}, "Run selection differs")
    require(last == load("checkpoints/"+history[-1]["checkpoint_file"].replace(".pt", ".json")), "Last pointer differs")
    augmentation = verify_augmentation(rows["train"], protocol, history, run)
    print(f"Artifact/protocol/history checks passed: {len(hashes)} files, {epochs} epochs", flush=True)
    recomputed, max_error = {}, 0.
    for epoch in selected_epochs:
        print(f"Direct COCO recomputation starting: epoch {epoch}", flush=True)
        values = direct_coco(gt, predictions.pop(epoch))
        expected = history[epoch-1]["validation"]
        errors = [abs(values[k]-expected[k]) for k in METRICS]
        for name in CLASSES:
            require(values["per_class"][name]["target_count"] == expected["per_class"][name]["target_count"], "Class target count differs")
            errors += [abs(values["per_class"][name][key]-expected["per_class"][name][key]) for key in ("AP50_95", "AP50")]
        require(max(errors) < 1e-12, f"Recomputed COCO metrics differ: epoch {epoch}")
        max_error = max(max_error, max(errors))
        recomputed[str(epoch)] = values
        print(f"Direct COCO recomputation PASS: epoch {epoch}, max difference {max(errors):.3g}", flush=True)
    selected_values = chosen["validation"]
    negative_boxes = sum(selected_values["no_target_predicted_boxes"].values())
    report = {
        "status": "verified_with_selected_and_final_metrics_recomputed",
        "archive": ARCHIVE.name, "archive_sha256": EXPECTED_SHA, "archive_bytes": ARCHIVE.stat().st_size,
        "archive_files": len(hashes), "result_checksum_entries": len(hashes)-1, "uncompressed_bytes": total_bytes,
        "released_code_files_matched": len(code_names), "protocol_and_environment_match": True,
        "frozen_train_val_separation": "image/source-cluster/decoded-RGB hashes disjoint",
        "ground_truth_matches_frozen_validation": True, "train_images": train_n, "validation_images": val_n,
        "validation_targets": len(gt["annotations"]), "completed_epochs": epochs,
        "optimizer_steps": run["optimizer_steps"], "image_presentations": run["image_presentations"],
        "all_epoch_prediction_geometry_counts_and_coverage": epoch_checks,
        "best_epoch": chosen["epoch"], "best_checkpoint": best,
        "best_metrics": recomputed[str(chosen["epoch"])], "final_metrics": recomputed[str(epochs)],
        "recomputed_epochs": selected_epochs, "maximum_recomputed_metric_error": max_error,
        "best_negative_images": selected_values["no_target_images"],
        "best_negative_images_with_predictions_at_0_25": selected_values["no_target_images_with_predictions"],
        "best_negative_predicted_boxes_at_0_25": negative_boxes,
        "best_negative_image_fp_fraction_at_0_25": selected_values["no_target_images_with_predictions"]/selected_values["no_target_images"],
        "run": {k: run[k] for k in ("pbs_job", "host", "gpu", "started_utc", "finished_utc", "elapsed_seconds", "peak_cuda_memory_allocated_bytes", "resume")},
        "training_loss_first_final": [sum(history[i]["training_loss"].values()) for i in (0, -1)],
        "validation_AP50_95_epoch12_to13_change": history[12]["validation"]["AP50_95"]-history[11]["validation"]["AP50_95"],
        "best_to_final_AP50_95_change": history[-1]["validation"]["AP50_95"]-selected_values["AP50_95"],
        "local_versions": {p: version(p) for p in ("numpy", "pycocotools", "matplotlib")},
        "audit_source_sha256": file_hash(__file__), "test_data_read": False,
        "augmentation_verification": augmentation,
        "limits": [
            "Only selected and final epochs' AP/AR were independently recomputed; all 20 epochs' hashes, coverage, geometry, counts and history consistency were verified.",
            "Full checkpoint binaries and the live stdout/stderr log are not in this records archive; checkpoint hashes are metadata, not a fresh audit of checkpoint bytes.",
            "Raw images were not reopened locally and detector inference was not rerun.",
            "One seed and one initialization/recipe; no seed-variation estimate or final-test evaluation.",
            "The best epoch is selected on this validation set, so its result is a development estimate rather than an untouched final-test estimate.",
            "No-target-image detections are false positives relative to retained annotations; annotation completeness was not visually re-audited."
        ],
        "recommendation": "Compare this fixed shared-augmentation run with the verified identity control under the declared validation checkpoint selection; do not infer general efficacy from one seed.",
        "next_experiment_executed": False,
    }
    save("ccast_verification.json", encoded(report))
    save("recomputed_metrics.json", encoded(recomputed))
    save("epoch_metrics.csv", contents["history.csv"])
    save("verified_run.json", contents["run.json"])
    save("verified_best_checkpoint.json", contents["best.json"])
    save("verified_context.json", contents["context.json"])
    save("verified_history.json", contents["history.json"])
    table = io.StringIO(newline="")
    writer = csv.DictWriter(table, fieldnames=["class", "validation_targets", "AP50_95", "AP50"])
    writer.writeheader()
    for name, values in report["best_metrics"]["per_class"].items():
        writer.writerow({"class": name, "validation_targets": values["target_count"],
                         "AP50_95": values["AP50_95"], "AP50": values["AP50"]})
    save("best_per_class.csv", table.getvalue().encode())
    print(json.dumps({k: report[k] for k in ("status", "best_epoch", "best_metrics", "maximum_recomputed_metric_error",
                                          "best_negative_image_fp_fraction_at_0_25")}, indent=2))


if __name__ == "__main__":
    main()
