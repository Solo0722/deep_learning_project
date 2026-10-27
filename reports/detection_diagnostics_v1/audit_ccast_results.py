"""Audit immutable diagnostic records; no model, raw image or held-out data access."""
from collections import Counter
import csv
from importlib import util
from importlib.metadata import version
import io
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
EXPECTED = "14dd16aad80c0dbf546994aa4c2aad80c6dfb215ca5d7f7a222e218b5557193f"
MODES = ("running_stats", "batch_stats", "running_stats_after")
SPEC = util.spec_from_file_location("learning_record_audit", ROOT / "reports/detection_learning_v1/audit_ccast_results.py")
audit = util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)
require, digest, save = audit.require, audit.digest, audit.save


def verify_manifest(files):
    sums = json.loads(files["checksums.json"])
    require(set(sums) == set(files) - {"checksums.json"}, "Result manifest coverage differs")
    for name, expected in sums.items():
        require(digest(files[name]) == expected, f"Result checksum differs: {name}")
    return len(sums)


def verify_code(files, directory):
    code = {name[5:]: data for name, data in files.items() if name.startswith("code/")}
    for name, data in code.items():
        require(data == (directory / name).read_bytes(), f"Local release differs: {name}")
    sums = {}
    for line in code["SHA256SUMS"].decode().splitlines():
        expected, name = line.split("  ", 1)
        require(name not in sums and digest(code[name]) == expected, "Package checksum differs")
        sums[name] = expected
    require(set(sums) | {"SHA256SUMS"} == set(code), "Package manifest coverage differs")
    return len(code)


def verify_ground_truth(gt, inputs, context):
    manifest = (ROOT / "detection_splits_v1/train.jsonl").read_bytes()
    require(digest(manifest) == context["manifest_sha256"], "Training manifest changed")
    rows = [json.loads(line) for line in manifest.splitlines()]
    expected_images, expected_targets = [], []
    for batch in inputs:
        require(batch["signature"] == digest(json.dumps(batch["images"], sort_keys=True).encode()), "Input signature changed")
        for entry in batch["images"]:
            row = rows[entry["image_id"]]
            require(row["split"] == "train" and row["image"] == entry["image"], "Input is not the frozen training row")
            require(row["image_sha256"] == entry["source_image_sha256"], "Source image hash record differs")
            require(entry["target_count"] == len(row["targets"]) and entry["canonical_labels"] ==
                    [t["class_index"] for t in row["targets"]], "Target counts/labels differ")
            expected_images.append({"id": entry["image_id"], "file_name": row["image"],
                                    "width": row["width"], "height": row["height"]})
            for target in row["targets"]:
                x1, y1, x2, y2 = target["box_xyxy"]
                expected_targets.append({"id": len(expected_targets)+1, "image_id": entry["image_id"],
                                         "category_id": target["class_index"]+1, "bbox": [x1, y1, x2-x1, y2-y1],
                                         "area": (x2-x1)*(y2-y1), "iscrowd": 0})
    require(gt["images"] == expected_images and gt["annotations"] == expected_targets, "Ground truth differs from manifest")
    require(gt["categories"] == [{"id": i+1, "name": name} for i, name in enumerate(audit.CLASSES)], "Ground truth classes differ")


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return math.isfinite(value) if isinstance(value, (float, int)) else True


def main():
    archive = ROOT / "detection_diagnostic_check_v1.tar.gz"
    files = audit.read_archive(archive, EXPECTED, "detection_diagnostic_check_v1")
    load = lambda name: json.loads(files[name])
    checksum_entries = verify_manifest(files)
    code_files = verify_code(files, ROOT / "detection_diagnostics_v1")
    protocol, run, gt, inputs = (load(n) for n in ("code/protocol.json", "run.json", "ground_truth.json", "inputs.json"))
    require(run["status"] == "complete" and run["protocol"] == protocol, "Incomplete or altered diagnostic")
    require(run["optimizer_steps"] == protocol["optimizer_steps"] == 0 and run["held_out_data_read"] is False, "Scope differs")
    require(run["versions"] == protocol["versions"] and protocol["modes"] == list(MODES), "Versions/modes differ")
    require(protocol["checkpoint_steps"] == [100, 200] and set(run["checkpoints"]) == {"100", "200"}, "Checkpoint coverage differs")
    require(protocol["expected_images"] == len(gt["images"]) == 15 and
            protocol["expected_targets"] == len(gt["annotations"]) == 16 and len(inputs) == 8, "Selection coverage differs")
    old_files = audit.read_archive(ROOT / "detection_learning_check_v1.tar.gz",
                                  protocol["previous_result_archive_sha256"], "detection_learning_check_v1")
    verify_manifest(old_files)
    old_code_files = verify_code(old_files, ROOT / "detection_learning_v1")
    require(set(protocol["previous_result_files_sha256"]) == set(old_files), "Prior result pins incomplete")
    for name, expected in protocol["previous_result_files_sha256"].items():
        require(digest(old_files[name]) == expected, f"Prior result differs: {name}")
    require(digest(old_files["code/SHA256SUMS"]) == protocol["learning_package_checksums_sha256"], "Dependency pin differs")
    old_run = json.loads(old_files["run.json"])
    require(run["source_context"] == old_run["context"], "Checkpoint source context differs")
    require(files["inputs.json"] == old_files["inputs.json"] and files["ground_truth.json"] == old_files["ground_truth.json"],
            "Input/ground-truth bytes differ from learning check")
    require(digest((ROOT / "detection_splits_v1/config.json").read_bytes()) == protocol["split_config_sha256"], "Split config changed")
    verify_ground_truth(gt, inputs, run["source_context"])
    images = {i["id"]: i for i in gt["images"]}
    positive_ids = {a["image_id"] for a in gt["annotations"]}
    summaries, metric_rows, target_rows, assignment_rows = {}, [], [], []
    maximum_metric_error, forward_records = 0., 0
    for source in protocol["checkpoints"]:
        step = source["completed_steps"]
        require(run["checkpoints"][str(step)]["source"] == source, "Checkpoint record differs")
        results, predictions_by_mode = {}, {}
        for mode in MODES:
            stem = f"step_{step:03d}_{mode}"
            d, predictions = load(stem + "_diagnostics.json"), load(stem + "_predictions.json")
            require(finite(d) and finite(predictions), "Nonfinite output")
            require(d["mode"] == mode and d["state_and_rng_unchanged"] is True and d["batchnorm_modules"] == 57,
                    "Missing state/RNG or normalization evidence")
            require({k: d[k] for k in ("metrics", "state_and_rng_unchanged", "batchnorm_modules")} ==
                    run["checkpoints"][str(step)]["modes"][mode], "Summary differs from diagnostic")
            counts = Counter(p["image_id"] for p in predictions)
            for p in predictions:
                require(p["image_id"] in images and p["category_id"] in (1, 2, 3, 4) and 0 <= p["score"] <= 1,
                        "Invalid prediction identity/score")
                x, y, w, h = p["bbox"]
                image = images[p["image_id"]]
                require(x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= image["width"]+1e-5 and
                        y+h <= image["height"]+1e-5 and counts[p["image_id"]] <= 100, "Invalid prediction geometry/count")
            recomputed = audit.coco_metrics(gt, predictions)
            for key in ("AP50", "AP50_95", "AP75", "AR100"):
                error = abs(recomputed[key]-d["metrics"][key])
                maximum_metric_error = max(maximum_metric_error, error)
                require(error <= 1e-12, "Scalar COCO metric differs")
            for name in audit.CLASSES:
                for key in ("AP50", "AP50_95"):
                    error = abs(recomputed["per_class"][name][key]-d["metrics"]["per_class"][name][key])
                    maximum_metric_error = max(maximum_metric_error, error)
                    require(error <= 1e-12, "Per-class COCO metric differs")
            negative_counts = {str(i): sum(p["image_id"] == i and p["score"] >= .25 for p in predictions)
                               for i in images if i not in positive_ids}
            require(d["metrics"]["no_target_predicted_boxes"] == negative_counts and
                    d["metrics"]["no_target_images_with_predictions"] == sum(v > 0 for v in negative_counts.values()),
                    "No-target metrics differ")
            require(d["metrics"]["prediction_boxes"] == len(predictions) and
                    d["metrics"]["evaluation_images"] == 15 and d["metrics"]["target_boxes"] == 16 and
                    d["metrics"]["no_target_images"] == 8, "Metric denominators differ")
            require(len(d["clipping"]) == 15, "Clipping records incomplete")
            require([c["image_id"] for c in d["clipping"]] == list(images), "Clipping record order differs")
            for c in d["clipping"]:
                require(c["after_clipping"] == counts[c["image_id"]] and
                        c["before_clipping"]-c["collapsed_after_clipping"] == c["after_clipping"], "Clipping count differs")
            require(len(d["batches"]) == len(inputs), "Batch records incomplete")
            mode_targets, mode_assignments = [], []
            for index, (b, original) in enumerate(zip(d["batches"], inputs)):
                require(b["batch_index"] == index and b["input_signature"] == original["signature"] and
                        b["target_counts"] == [i["target_count"] for i in original["images"]], "Diagnostic batch differs")
                require(len(b["images"]) == len(original["images"]) and b["classification_logit_min"] <= b["classification_logit_max"],
                        "Batch image/logit records differ")
                assignments = b["assignments"]
                require(all(len(values) == len(b["images"]) for values in assignments.values()), "Assignments incomplete")
                for j, (im, original_image) in enumerate(zip(b["images"], original["images"])):
                    require(im["image_id"] == original_image["image_id"] and im["raw_candidates"] == 8400 and
                            len(im["targets"]) == original_image["target_count"], "Raw candidate image/target coverage differs")
                    require(len(im["candidates_above_threshold_by_class"]) == len(im["maximum_probability_by_class"]) == 4 and
                            all(0 <= n <= 8400 for n in im["candidates_above_threshold_by_class"]) and
                            all(0 <= p <= 1 for p in im["maximum_probability_by_class"]), "Raw candidate scores/counts invalid")
                    fg, mass, indices = (assignments[k][j] for k in
                                         ("foreground_count_per_image", "target_score_sum_per_image", "assigned_gt_indices_per_image"))
                    require(0 <= fg <= 8400 and 0 <= mass <= fg+1e-4 and
                            len(indices) == len(set(indices)) and all(0 <= k < len(im["targets"]) for k in indices), "Invalid assignment")
                    require(bool(fg) == bool(indices) and (len(im["targets"]) > 0 or (fg == mass == 0)), "Empty image assigned foreground")
                    row = {"step": step, "mode": mode, "batch": index, "image_id": im["image_id"],
                           "targets": len(im["targets"]), "foreground_candidates": fg, "target_score_sum": mass,
                           "assigned_target_count": len(indices)}
                    mode_assignments.append(row)
                    assignment_rows.append(row)
                    for k, target in enumerate(im["targets"]):
                        require(target["target_index"] == k and target["class_index"] == original_image["canonical_labels"][k],
                                "Raw candidate target label/index differs")
                        raw, eligible, same = (target[n] for n in ("best_raw_iou", "best_any_class_eligible_iou", "best_same_class_eligible_iou"))
                        require(0 <= same <= eligible <= raw <= 1, "Raw candidate IoUs invalid")
                        row = {"step": step, "mode": mode, "image_id": im["image_id"], **target}
                        mode_targets.append(row)
                        target_rows.append(row)
            forward_records += len(d["batches"])
            retained = audit.prediction_diagnostics(gt, predictions)
            raw_coverage = {key: sum(t[key] >= .5 for t in mode_targets) for key in
                            ("best_raw_iou", "best_any_class_eligible_iou", "best_same_class_eligible_iou")}
            summary = {"metrics": d["metrics"], "retained_prediction_geometry": retained,
                       "raw_targets_covered_at_iou_0_5": raw_coverage,
                       "foreground_candidates": sum(x["foreground_candidates"] for x in mode_assignments),
                       "assigned_target_score_sum": sum(x["target_score_sum"] for x in mode_assignments),
                       "assigned_targets": sum(x["assigned_target_count"] for x in mode_assignments),
                       "classification_logit_min": min(b["classification_logit_min"] for b in d["batches"]),
                       "classification_logit_max": max(b["classification_logit_max"] for b in d["batches"]),
                       "state_and_rng_unchanged_reported": True}
            results[mode], predictions_by_mode[mode] = d, predictions
            summaries.setdefault(str(step), {})[mode] = summary
            metric_rows.append({"step": step, "mode": mode, **{k: recomputed[k] for k in ("AP50", "AP50_95", "AP75", "AR100")},
                                "predictions": len(predictions), **raw_coverage,
                                "retained_same_class_coverage_at_iou_0_5": retained["targets_with_same_class_iou_at_least_0_5"]})
        require(files[f"step_{step}_running_stats_predictions.json"] == old_files[f"yolov8n_step_{step}_predictions.json"] ==
                files[f"step_{step}_running_stats_after_predictions.json"], "Running-stat replay not byte-identical")
        before, after = results["running_stats"], results["running_stats_after"]
        for key in ("batches", "clipping", "metrics"):
            require(before[key] == after[key], f"Running-stat diagnostic repeat differs: {key}")
        for result, key in ((before, "matches_saved_predictions"), (after, "matches_initial_running_predictions")):
            require(result[key] == {"predictions_match_within_tolerance": True, "maximum_original_coordinate_error": 0.,
                                    "maximum_score_error": 0.}, "Replay error recorded")
    require(forward_records == protocol["model_forward_budget"] == 48, "Forward record budget differs")
    record = {"status": "verified_normalization_sensitivity_on_training_subset", "archive": archive.name,
              "archive_sha256": EXPECTED, "archive_bytes": archive.stat().st_size, "archive_files": len(files),
              "result_checksum_entries": checksum_entries, "code_files_matched": code_files,
              "learning_dependency_code_files_matched": old_code_files, "previous_record_hashes_matched": len(old_files),
              "diagnostic_forward_records_verified": forward_records, "optimizer_steps_reported": 0,
              "held_out_data_read_reported": False, "ground_truth_matches_frozen_train_manifest": True,
              "input_records_match_learning_run_exactly": True, "running_stat_predictions_replay_original_bytes": True,
              "running_stat_diagnostics_repeat_exactly": True, "maximum_recomputed_metric_error": maximum_metric_error,
              "local_pycocotools": version("pycocotools"), "remote_pycocotools": run["versions"]["pycocotools"],
              "audit_code_sha256": digest(Path(__file__).read_bytes()),
              "audit_helper_sha256": digest(Path(SPEC.origin).read_bytes()),
              "pbs_job": run["pbs_job"], "host": run["host"], "gpu": run["gpu_name"],
              "finished_utc_reported": run["finished_utc"], "checkpoint_results": summaries,
              "limits": ["No raw image decoding, model execution or full checkpoint reload was repeated locally.",
                         "Parameter, buffer, RNG and checkpoint preservation are inspected execution records, not an independent live check.",
                         "Raw candidates and assignment summaries were checked for consistency, not regenerated from tensors.",
                         "Candidate coverage is not one-to-one recall or AP; score sums are normalized soft assignment weights, not object counts.",
                         "Batch-stat results depend on the fixed image batches; no deployment or held-out claim.",
                         "This does not identify why normalization statistics differ or prove the cause of the earlier update spike."]}
    for name, data in files.items():
        save(OUT / "returned_records" / name, data)
    save(OUT / "ccast_verification.json", audit.json_bytes(record))
    for filename, rows in (("metrics.csv", metric_rows), ("raw_target_coverage.csv", target_rows), ("assignments.csv", assignment_rows)):
        buffer = io.StringIO(newline="")
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        save(OUT / filename, buffer.getvalue().encode())
    print(json.dumps({"status": record["status"], "files": len(files), "metric_error": maximum_metric_error,
                      "metrics": metric_rows, "assignments": {s: {m: {k: r[k] for k in
                        ("foreground_candidates", "assigned_target_score_sum", "assigned_targets", "classification_logit_min", "classification_logit_max")}
                        for m, r in modes.items()} for s, modes in summaries.items()}}, indent=2))


if __name__ == "__main__":
    main()
