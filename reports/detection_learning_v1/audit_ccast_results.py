"""Verify the returned records and recompute metrics without loading a detector.

Run from any directory with the local Python environment. Inputs are immutable;
returned bytes and analysis outputs are written only if absent or identical.
No held-out data, raw images, or model checkpoints are opened.
"""
from collections import Counter
import contextlib
import copy
import csv
import hashlib
from importlib.metadata import version
import io
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import tarfile

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPECTED = "0ab14951c60bf566e2585ea66648d3bc743e2c427558b1f85b762ed50dd9693a"
MODELS = ("fasterrcnn_resnet50_fpn", "yolov8n")
CLASSES = ("D00", "D10", "D20", "D40")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"Preserve existing different output: {path}")
    else:
        with path.open("xb") as handle:
            handle.write(data)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def read_archive(path, expected, prefix):
    require(digest(path.read_bytes()) == expected, f"Archive checksum differs: {path}")
    files = {}
    with tarfile.open(path, "r:gz") as bundle:
        members = bundle.getmembers()
        require(sum(m.size for m in members) < 50_000_000, "Unexpected archive size")
        for member in members:
            require(member.isfile() and not member.issym() and not member.islnk(), "Unexpected member type")
            name = PurePosixPath(member.name)
            require(not name.is_absolute() and name.parts[0] == prefix and len(name.parts) > 1,
                    "Unexpected archive prefix")
            require(all(p not in (".", "..") and ":" not in p and "\\" not in p for p in name.parts),
                    "Unsafe archive path")
            relative = "/".join(name.parts[1:])
            require(relative not in files, "Duplicate archive member")
            files[relative] = bundle.extractfile(member).read()
    return files


def coco_metrics(gt, predictions):
    """Direct COCO API call, independent of the released metrics wrapper."""
    with contextlib.redirect_stdout(io.StringIO()):
        truth = COCO()
        truth.dataset = copy.deepcopy(gt)
        truth.createIndex()
        if predictions:
            detections = truth.loadRes(copy.deepcopy(predictions))
        else:
            detections = COCO()
            detections.dataset = {"images": copy.deepcopy(gt["images"]),
                                  "categories": copy.deepcopy(gt["categories"]), "annotations": []}
            detections.createIndex()
        evaluator = COCOeval(truth, detections, "bbox")
        evaluator.params.imgIds = sorted(x["id"] for x in gt["images"])
        evaluator.params.catIds = [1, 2, 3, 4]
        evaluator.params.maxDets = [1, 10, 100]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]

    def mean_valid(values):
        values = values[values >= 0]
        return float(values.mean()) if len(values) else None

    per_class = {name: {"AP50": mean_valid(precision[0, :, i, 0, -1]),
                        "AP50_95": mean_valid(precision[:, :, i, 0, -1])}
                 for i, name in enumerate(CLASSES)}
    return {"AP50_95": float(evaluator.stats[0]), "AP50": float(evaluator.stats[1]),
            "AP75": float(evaluator.stats[2]), "AR100": float(evaluator.stats[8]),
            "per_class": per_class}


def iou(a, b):
    intersection = max(0., min(a[0]+a[2], b[0]+b[2])-max(a[0], b[0])) * max(
        0., min(a[1]+a[3], b[1]+b[3])-max(a[1], b[1]))
    return intersection / (a[2]*a[3]+b[2]*b[3]-intersection)


def prediction_diagnostics(gt, predictions):
    per_target = []
    for target in gt["annotations"]:
        same_image = [p for p in predictions if p["image_id"] == target["image_id"]]
        same_class = [p for p in same_image if p["category_id"] == target["category_id"]]
        per_target.append({"target_id": target["id"], "image_id": target["image_id"],
                           "category_id": target["category_id"],
                           "best_any_class_iou": max((iou(target["bbox"], p["bbox"]) for p in same_image), default=0.),
                           "best_same_class_iou": max((iou(target["bbox"], p["bbox"]) for p in same_class), default=0.)})
    scores = [p["score"] for p in predictions]
    return {"detections": len(predictions),
            "detections_by_class": {name: sum(p["category_id"] == i+1 for p in predictions)
                                    for i, name in enumerate(CLASSES)},
            "score_min_median_max": [min(scores), statistics.median(scores), max(scores)] if scores else None,
            "targets_with_any_class_iou_at_least_0_5": sum(p["best_any_class_iou"] >= .5 for p in per_target),
            "targets_with_same_class_iou_at_least_0_5": sum(p["best_same_class_iou"] >= .5 for p in per_target),
            "per_target": per_target,
            "scope": "Coverage of the saved post-threshold/post-NMS predictions; not raw anchor recall, one-to-one recall or AP"}


def main():
    archive = ROOT / "detection_learning_check_v1.tar.gz"
    files = read_archive(archive, EXPECTED, "detection_learning_check_v1")
    load = lambda name: json.loads(files[name])
    sums = load("checksums.json")
    require(set(sums) == set(files)-{"checksums.json"}, "Result manifest coverage differs")
    for name, expected in sums.items():
        require(digest(files[name]) == expected, f"Result file hash differs: {name}")
    code_names = [name for name in files if name.startswith("code/")]
    for name in code_names:
        require(files[name] == (ROOT / "detection_learning_v1" / name[5:]).read_bytes(), f"Released code differs: {name}")
    package_sums = {}
    for line in files["code/SHA256SUMS"].decode().splitlines():
        expected, name = line.split("  ", 1)
        require(name not in package_sums and digest(files["code/"+name]) == expected, "Package checksum mismatch")
        package_sums[name] = expected
    require(set(package_sums) | {"SHA256SUMS"} == {p[5:] for p in code_names}, "Package manifest coverage differs")
    run, protocol, selection = load("run.json"), load("code/protocol.json"), load("code/selection.json")
    require(run["status"] == "complete" and run["protocol"] == protocol, "Run/protocol mismatch")
    require(run["versions"] == protocol["versions"] and not run["held_out_data_read"], "Run context differs")
    require(run["paired_inputs_verified"], "Paired input check absent")
    for key, name in [("protocol_sha256", "protocol.json"), ("selection_sha256", "selection.json"),
                      ("package_checksums_sha256", "SHA256SUMS")]:
        require(run["context"][key] == digest(files["code/"+name]), f"Context mismatch: {key}")
    manifest_bytes = (ROOT / "detection_splits_v1/train.jsonl").read_bytes()
    require(digest(manifest_bytes) == run["context"]["manifest_sha256"], "Frozen train manifest differs")
    rows = [json.loads(line) for line in manifest_bytes.splitlines()]
    inputs, gt = load("inputs.json"), load("ground_truth.json")
    entries = [image for batch in inputs for image in batch["images"]]
    require(len(entries) == 15 and len({i["image_id"] for i in entries}) == 15, "Selected image coverage differs")
    require([i["image"] for i in entries] == [i["image"] for i in selection["selected"]], "Selection order differs")
    expected_gt_images, expected_annotations = [], []
    for entry in entries:
        row = rows[entry["image_id"]]
        require(row["split"] == "train" and row["image"] == entry["image"], "Unexpected image identity")
        require(row["image_sha256"] == entry["source_image_sha256"], "Image hash record differs")
        require(entry["target_count"] == len(row["targets"]) and
                entry["canonical_labels"] == [t["class_index"] for t in row["targets"]], "Target mapping differs")
        expected_gt_images.append({"id": entry["image_id"], "file_name": row["image"],
                                   "width": row["width"], "height": row["height"]})
        for target in row["targets"]:
            x1, y1, x2, y2 = target["box_xyxy"]
            expected_annotations.append({"id": len(expected_annotations)+1, "image_id": entry["image_id"],
                                         "category_id": target["class_index"]+1, "bbox": [x1,y1,x2-x1,y2-y1],
                                         "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    require(gt["images"] == expected_gt_images and gt["annotations"] == expected_annotations,
            "Evaluation ground truth differs from frozen training manifest")
    require(gt["categories"] == [{"id": i+1, "name": name} for i,name in enumerate(CLASSES)], "GT classes differ")
    for batch in inputs:
        require(batch["signature"] == digest(json.dumps(batch["images"], sort_keys=True).encode()), "Input signature differs")
    prior = ROOT / "detection_integration_check_v1.tar.gz"
    require(digest(prior.read_bytes()) == "aa63883e24fecdbf7f68f993562310935f9b935247940e27edaafbd46e98e0a1",
            "Prior integration archive hash differs")
    with tarfile.open(prior) as bundle:
        old_steps = [json.loads(line) for line in bundle.extractfile("detection_integration_check_v1/steps.jsonl")]
    for name in MODELS:
        require([s["inputs"] for s in old_steps if s["model"] == name] == inputs,
                "Input signatures/pixel hashes/geometry differ from prior verified integration")
    steps = [json.loads(line) for line in files["steps.jsonl"].splitlines()]
    require(len(steps) == 400, "Step count differs")
    checkpoints = load("checkpoint_records.json")
    require(len(checkpoints) == 4 and not any(n.startswith("checkpoints/") for n in files), "Checkpoint archive scope differs")
    images = {x["id"]: x for x in gt["images"]}
    positive_ids = {x["image_id"] for x in gt["annotations"]}
    per_model, metric_rows, pass_rows = {}, [], []
    for name in MODELS:
        status = run["models"][name]
        sequence = [s for s in steps if s["model"] == name]
        require(status["status"] == "complete" and status["steps_completed"] == 200 and
                status["image_presentations"] == 375, "Incomplete training")
        require([s["step"] for s in sequence] == list(range(1,201)), "Missing/duplicate steps")
        for step in sequence:
            batch = (step["step"]-1) % 8
            require(step["batch_index"] == batch and step["input_signature"] == inputs[batch]["signature"], "Step input mismatch")
            update = step["update"]
            values = [update["total_loss"], update["classifier_max_absolute_update"], *update["loss_components"].values()]
            require(all(math.isfinite(x) for x in values) and update["classifier_max_absolute_update"] > 0,
                    "Missing/invalid update")
            factor = len(inputs[batch]["images"]) if name == "yolov8n" else 1
            require(math.isclose(update["total_loss"], factor*sum(update["loss_components"].values()),
                                 rel_tol=2e-6, abs_tol=1e-6), "Loss aggregation differs")
        require([r["completed_steps"] for r in status["checkpoint_reloads"]] == [100,200], "Checkpoint restore schedule differs")
        for restored in status["checkpoint_reloads"]:
            require(all(restored[k] for k in ("model_state_exact", "optimizer_state_exact", "rng_state_exact")), "Restore check failed")
        require(load(name+"_reload_predictions.json") == load(name+"_step_200_predictions.json"), "Reload prediction records differ")
        require([(c["model_name"],c["completed_steps"]) for c in checkpoints if c["model_name"]==name] == [(name,100),(name,200)], "Checkpoint records differ")
        model_metrics, diagnostics, max_metric_error = [], [], 0.
        for at in protocol["evaluate_steps"]:
            stem = f"{name}_step_{at:03d}"
            result = load(stem+"_metrics.json")
            saved, predictions = result["metrics"], load(stem+"_predictions.json")
            for pred in predictions:
                require(pred["image_id"] in images and type(pred["category_id"]) is int and 1<=pred["category_id"]<=4,
                        "Invalid prediction identity")
                x,y,w,h = pred["bbox"]; image = images[pred["image_id"]]
                require(all(math.isfinite(v) for v in [x,y,w,h,pred["score"]]) and min(w,h)>0 and
                        0<=x<=x+w<=image["width"]+1e-8 and 0<=y<=y+h<=image["height"]+1e-8 and
                        protocol["evaluation"]["score_threshold"]<=pred["score"]<=1, "Invalid prediction geometry/score")
            require(max(Counter(p["image_id"] for p in predictions).values(), default=0) <= 100, "Detection cap differs")
            recomputed = coco_metrics(gt, predictions)
            for key in ("AP50", "AP50_95", "AP75", "AR100"):
                max_metric_error = max(max_metric_error, abs(saved[key]-recomputed[key]))
                require(abs(saved[key]-recomputed[key]) < 1e-12, f"Recomputed {key} differs")
            for cls in CLASSES:
                for key in ("AP50", "AP50_95"):
                    require(abs(saved["per_class"][cls][key]-recomputed["per_class"][cls][key]) < 1e-12, "Per-class AP differs")
            negative_counts = {str(i): sum(p["image_id"]==i and p["score"]>=.25 for p in predictions)
                               for i in sorted(set(images)-positive_ids)}
            require(saved["no_target_predicted_boxes"] == negative_counts and saved["prediction_boxes"]==len(predictions),
                    "Prediction counts differ")
            require({"step": at, **saved} == next(x for x in status["evaluations"] if x["step"]==at),
                    "Run evaluation record differs")
            diagnostics.append({"step": at, **prediction_diagnostics(gt,predictions)})
            model_metrics.append({"step": at, **recomputed})
            metric_rows.append({"model":name,"step":at,**{k:recomputed[k] for k in ("AP50","AP50_95","AP75","AR100")},
                                **{cls+"_AP50":recomputed["per_class"][cls]["AP50"] for cls in CLASSES}})
        for index in range(25):
            block = sequence[index*8:(index+1)*8]
            pass_rows.append({"model": name,"pass":index+1,"first_step":block[0]["step"],"last_step":block[-1]["step"],
                              "mean_total_loss": statistics.mean(x["update"]["total_loss"] for x in block),
                              "max_classifier_absolute_update":max(x["update"]["classifier_max_absolute_update"] for x in block)})
        biggest = max(sequence,key=lambda x:x["update"]["classifier_max_absolute_update"])
        per_model[name] = {"metrics":model_metrics,"prediction_diagnostics":diagnostics,
                           "maximum_recomputed_scalar_metric_error":max_metric_error,
                           "largest_classifier_update":{**biggest,"batch_target_counts":[i["target_count"] for i in inputs[biggest["batch_index"]]["images"]]},
                           "learning_review":status["learning_review"],"checkpoint_reloads":status["checkpoint_reloads"]}
    record = {"status":"verified_records_both_models_need_learning_review", "archive":archive.name,
              "archive_sha256":EXPECTED,"archive_bytes":archive.stat().st_size,"archive_files":len(files),
              "result_checksum_entries":len(sums),"package_files_matched_to_release":len(code_names),
              "package_checksum_entries":len(package_sums),"step_records_verified":len(steps),
              "training_images":15,"target_boxes":16,"no_target_images":8,
              "ground_truth_matches_frozen_train_manifest":True,"inputs_match_prior_verified_integration":True,
              "evaluation_recomputed_directly_with_pycocotools":version("pycocotools"),
              "remote_pycocotools":run["versions"]["pycocotools"],"numpy":version("numpy"),
              "pbs_job":run["pbs_job"],"host":run["host"],"gpu":run["gpu_name"],
              "elapsed_seconds_reported":run["elapsed_seconds"],"models":per_model,
              "checkpoints":checkpoints,"full_checkpoint_binaries_verified_locally":False,
              "limits":["No raw image decoding or GPU/model execution was repeated locally.",
                        "Checkpoint restore results are inspected records, not an independent reload of the binaries.",
                        "These are training-subset engineering diagnostics; no ranking, generalization or controller-efficacy conclusion.",
                        "Post-NMS records do not expose raw logits, positive assignment counts, batch statistics or full gradient norms."]}
    for name,data in files.items():
        save(OUT/"returned_records"/name,data)
    save(OUT/"ccast_verification.json",json_bytes(record))
    for filename,table in [("metrics.csv",metric_rows),("loss_by_pass.csv",pass_rows)]:
        buf=io.StringIO(newline="")
        writer=csv.DictWriter(buf,fieldnames=list(table[0]),lineterminator="\n")
        writer.writeheader();writer.writerows(table)
        save(OUT/filename,buf.getvalue().encode())
    print(json.dumps({"status":record["status"],"files":len(files),"checksums":len(sums),"code_files":len(code_names),
                      "step_records":len(steps),"max_metric_error":max(m["maximum_recomputed_scalar_metric_error"] for m in per_model.values()),
                      "final_diagnostics":{n:m["prediction_diagnostics"][-1] for n,m in per_model.items()}},indent=2))


if __name__ == "__main__":
    main()
