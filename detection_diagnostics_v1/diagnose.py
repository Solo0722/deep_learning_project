"""Evaluation-only YOLO normalization diagnostic on existing training checkpoints."""
from contextlib import contextmanager
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import socket
import sys
import time
import traceback

import torch

# The previously verified release is a pinned sibling dependency, not copied or edited.
LEARNING = Path(__file__).resolve().parent.parent / "detection_learning_v1"
sys.path.insert(0, str(LEARNING))
import checkpoints
from integration import check_output, paired_batch, require, verified_package, write_json
from learning import archive_records, compare_predictions
import metrics
from models import build_model
from rdd_data import TrainingDataset, inside


def verify_files(root, expected):
    for name, digest in expected.items():
        require(checkpoints.file_hash(inside(root, name)) == digest, f"Changed input: {name}")


@contextmanager
def normalization_mode(model, mode):
    """Change normalization for forward passes; preserve parameters and BN buffers."""
    require(mode in ("running_stats", "batch_stats", "running_stats_after"), "Unknown diagnostic mode")
    modules = list(model.modules())
    flags = [(module, module.training) for module in modules]
    bn = [module for module in modules if isinstance(module, torch.nn.BatchNorm2d)]
    require(bn and all(m.track_running_stats and m.running_mean is not None and m.running_var is not None for m in bn),
            "Expected tracked BatchNorm2d modules")
    tracked = [(module, module.track_running_stats) for module in bn]
    state = checkpoints.cpu_tree(model.state_dict())
    random_state = checkpoints.rng_state()
    model.eval()
    if mode == "batch_stats":
        for module in bn:
            module.training = True
            # BatchNorm uses current-batch statistics without updating running buffers.
            module.track_running_stats = False
    try:
        with torch.inference_mode():
            yield len(bn)
    finally:
        for module, track in tracked:
            module.track_running_stats = track
        for module, flag in flags:
            module.training = flag
        checkpoints.assert_same(model.state_dict(), state, "diagnostic model state")
        checkpoints.assert_same(checkpoints.rng_state(), random_state, "diagnostic RNG state")


def raw_geometry(decoded, examples):
    """Report raw candidate coverage at the fixed threshold, before NMS/clipping."""
    from torchvision.ops import box_iou
    require(decoded.ndim == 3 and decoded.shape[:2] == (len(examples), 8), "Unexpected decoded shape")
    require(torch.isfinite(decoded).all().item(), "Nonfinite decoded predictions")
    results = []
    for prediction, example in zip(decoded, examples):
        centers, sizes = prediction[:2].T, prediction[2:4].T
        boxes = torch.cat((centers-sizes/2, centers+sizes/2), dim=1)
        scores = prediction[4:].T
        require(((scores >= 0) & (scores <= 1)).all().item(), "Scores outside probability range")
        gt = torch.as_tensor(example.boxes, device=boxes.device, dtype=boxes.dtype).reshape(-1, 4)
        overlap = box_iou(gt, boxes) if len(gt) else boxes.new_empty((0, len(boxes)))
        any_eligible = scores.max(dim=1).values >= .001
        targets = []
        for i, label in enumerate(example.labels):
            eligible = scores[:, int(label)] >= .001
            targets.append({"target_index": i, "class_index": int(label),
                            "best_raw_iou": float(overlap[i].max().item()),
                            "best_any_class_eligible_iou": float(overlap[i, any_eligible].max().item()) if any_eligible.any().item() else 0.,
                            "best_same_class_eligible_iou": float(overlap[i, eligible].max().item()) if eligible.any().item() else 0.})
        results.append({"image_id": example.image_id, "raw_candidates": len(boxes),
                        "candidates_above_threshold_by_class": (scores >= .001).sum(dim=0).cpu().tolist(),
                        "maximum_probability_by_class": scores.max(dim=0).values.cpu().tolist(), "targets": targets})
    return results


def capture_assignments(output):
    _, _, scores, foreground, target_index = output
    require(torch.isfinite(scores).all().item(), "Nonfinite assigned target scores")
    return {"foreground_count_per_image": foreground.sum(dim=1).cpu().tolist(),
            "target_score_sum_per_image": scores.sum(dim=(1, 2)).cpu().tolist(),
            "assigned_gt_indices_per_image": [torch.unique(target_index[i][foreground[i]]).cpu().tolist()
                                               for i in range(len(foreground))]}


def measure(model, examples, evaluation, device, mode):
    from ultralytics.utils.nms import non_max_suppression
    require(evaluation["score_threshold"] == .001, "Diagnostic threshold changed")
    batches = [examples[i:i+2] for i in range(0, len(examples), 2)]
    predictions, clipping, diagnostics = [], [], []
    criterion = model.init_criterion()
    with normalization_mode(model, mode) as bn_count:
        for index, batch_examples in enumerate(batches):
            _, _, batch, inputs = paired_batch(batch_examples)
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(batch["img"])
            require(isinstance(output, tuple) and len(output) == 2 and isinstance(output[1], dict),
                    "Expected inference tensor and raw training-output dictionary")
            decoded, raw = output
            geometry = raw_geometry(decoded, batch_examples)
            require(torch.isfinite(raw["scores"]).all().item(), "Nonfinite raw logits")
            assignments = []
            hook = criterion.assigner.register_forward_hook(lambda module, args, values: assignments.append(capture_assignments(values)))
            try:
                _, loss_vector, losses = criterion.get_assigned_targets_and_loss(raw, batch)
            finally:
                hook.remove()
            require(len(assignments) == 1 and torch.isfinite(loss_vector).all().item(), "Invalid assignment/loss diagnostics")
            nms = non_max_suppression(decoded.clone(), conf_thres=evaluation["score_threshold"],
                                      iou_thres=evaluation["nms_iou"], nc=4, multi_label=evaluation["yolo_multi_label"],
                                      max_det=evaluation["max_detections"], max_time_img=5.0)
            require(len(nms) == len(batch_examples), "Missing NMS image")
            for example, values in zip(batch_examples, nms):
                pred = {"boxes": values[:, :4].cpu(), "scores": values[:, 4].cpu(), "labels": values[:, 5].cpu()}
                converted, info = metrics.convert_prediction(example, pred)
                predictions.extend(converted)
                clipping.append({"image_id": example.image_id, **info})
            diagnostics.append({"batch_index": index, "input_signature": inputs["signature"],
                                "target_counts": [len(e.boxes) for e in batch_examples],
                                "loss_components_without_batch_multiplier": {k: float(v.item()) for k, v in losses.items()},
                                "classification_logit_min": float(raw["scores"].min().item()),
                                "classification_logit_max": float(raw["scores"].max().item()),
                                "assignments": assignments[0], "images": geometry})
    return {"mode": mode, "batchnorm_modules": bn_count, "state_and_rng_unchanged": True,
            "metrics": metrics.evaluate(metrics.ground_truth(examples), predictions, evaluation["negative_confidence"]),
            "clipping": clipping, "batches": diagnostics}, predictions


def load_existing(path, expected, context, model):
    require(path.stat().st_size == expected["bytes"] and checkpoints.file_hash(path) == expected["sha256"],
            "Checkpoint bytes/hash differ")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["format_version"] == 1 and payload["model_name"] == "yolov8n" and
            payload["completed_steps"] == expected["completed_steps"] and payload["context"] == context,
            "Checkpoint identity/context differs")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    checkpoints.assert_same(model.state_dict(), payload["model"], "loaded model state")


def diagnose_checkpoint(model, examples, evaluation, device, reference, modes):
    results, predictions = {}, {}
    baseline = None
    for mode in modes:
        result, current = measure(model, examples, evaluation, device, mode)
        if mode == "running_stats":
            result["matches_saved_predictions"] = compare_predictions(reference, current)
            baseline = current
        elif mode == "running_stats_after":
            require(baseline is not None, "Missing running-stat baseline")
            result["matches_initial_running_predictions"] = compare_predictions(baseline, current)
        results[mode], predictions[mode] = result, current
        print(f"EVAL {mode} training_AP50={result['metrics']['AP50']:.6f} "
              f"training_AP50:95={result['metrics']['AP50_95']:.6f}", flush=True)
    return results, predictions


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "splits", "previous", "out"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    sums = {}
    for line in (package/"SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        require(name not in sums, "Duplicate package checksum")
        sums[name] = digest
    require({"diagnose.py", "protocol.json", "README.md", "tests/test_diagnose.py"}.issubset(sums), "Missing package files")
    verify_files(package, sums)
    protocol = json.loads((package/"protocol.json").read_text())
    require(protocol["checkpoint_steps"] == [100,200] and
            protocol["modes"] == ["running_stats", "batch_stats", "running_stats_after"] and
            protocol["optimizer_steps"] == 0, "Bounded diagnostic protocol changed")
    require(checkpoints.file_hash(LEARNING/"SHA256SUMS") == protocol["learning_package_checksums_sha256"], "Learning release differs")
    verified_package(LEARNING)
    root, splits, previous, out = [getattr(args,n).resolve() for n in ("root", "splits", "previous", "out")]
    archive = check_output(out, (root, splits, previous, package, LEARNING))
    require(os.environ.get("PBS_JOBID") and not socket.gethostname().lower().startswith("login"), "Use an allocated GPU compute job")
    require(Path(sys.prefix).resolve() == Path(protocol["expected_environment"]).resolve(), "Activate rdd_detection_v1")
    versions = {name:version(name) for name in protocol["versions"]}
    require(versions == protocol["versions"], "Dependency versions differ")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one assigned CUDA GPU")
    verify_files(previous, protocol["previous_result_files_sha256"])
    old_run = json.loads((previous/"run.json").read_text())
    require(old_run["status"] == "complete", "Original run incomplete")
    require(checkpoints.file_hash(splits/"config.json") == protocol["split_config_sha256"], "Frozen split configuration differs")
    records = [r for r in json.loads((previous/"checkpoint_records.json").read_text()) if r["model_name"] == "yolov8n"]
    require(records == protocol["checkpoints"], "Checkpoint records differ")
    for record in records:
        path = inside(previous/"checkpoints", record["file"])
        require(path.stat().st_size == record["bytes"] and checkpoints.file_hash(path) == record["sha256"], "Checkpoint changed")
    selection = json.loads((LEARNING/"selection.json").read_text())
    dataset = TrainingDataset(root, splits/"train.jsonl", old_run["context"]["manifest_sha256"],
                              [x["image"] for x in selection["selected"]], old_run["protocol"]["image_size"])
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out.mkdir(exist_ok=False)
    for name in [*sums, "SHA256SUMS"]:
        destination = out/"code"/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(inside(package,name).read_bytes())
    report = {"status":"running", "started_utc":datetime.now(timezone.utc).isoformat(), "protocol":protocol,
              "source_context":old_run["context"], "pbs_job":os.environ["PBS_JOBID"], "host":socket.gethostname(),
              "gpu_name":torch.cuda.get_device_name(0), "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
              "python":platform.python_version(), "versions":versions, "command":sys.argv,
              "optimizer_steps":0, "held_out_data_read":False, "checkpoints":{},
              "scope":"Exploratory training-only normalization diagnostic; batch-stat scores are not a deployment evaluation"}
    started = time.perf_counter()
    try:
        examples = [dataset[i] for i in range(len(dataset))]
        require(metrics.ground_truth(examples) == json.loads((previous/"ground_truth.json").read_text()), "Ground truth differs")
        inputs = [paired_batch(examples[i:i+2])[3] for i in range(0,len(examples),2)]
        require(inputs == json.loads((previous/"inputs.json").read_text()), "Paired input records changed")
        write_json(out/"ground_truth.json", metrics.ground_truth(examples))
        write_json(out/"inputs.json", inputs)
        for record in records:
            step = record["completed_steps"]
            torch.manual_seed(42)
            model, _, effective = build_model("yolov8n", 640, "cuda", old_run["protocol"]["evaluation"], None)
            checkpoint = inside(previous/"checkpoints", record["file"])
            load_existing(checkpoint, record, old_run["context"], model)
            reference = json.loads((previous/f"yolov8n_step_{step:03d}_predictions.json").read_text())
            print(f"CHECKPOINT step={step}: verified; starting evaluation-only diagnostic", flush=True)
            results, predictions = diagnose_checkpoint(model, examples, old_run["protocol"]["evaluation"], "cuda",
                                                       reference, protocol["modes"])
            for mode in protocol["modes"]:
                write_json(out/f"step_{step:03d}_{mode}_diagnostics.json", results[mode])
                write_json(out/f"step_{step:03d}_{mode}_predictions.json", predictions[mode])
            report["checkpoints"][str(step)] = {"source":record,"effective_configuration":effective,
                                                "modes":{mode:{k:results[mode][k] for k in ("metrics","state_and_rng_unchanged","batchnorm_modules")}
                                                         for mode in protocol["modes"]}}
            require(checkpoints.file_hash(checkpoint) == record["sha256"], "Source checkpoint changed")
            write_json(out/"run.json", report)
            del model
            torch.cuda.empty_cache()
        verify_files(previous, protocol["previous_result_files_sha256"])
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.perf_counter()-started,3)
        write_json(out/"run.json",report)
    archive_records(out,archive)
    print("YOLO EVALUATION-ONLY DIAGNOSTIC: COMPLETE (0 optimizer steps)")
    print(f"Archive: {archive}")
    print(f"Archive SHA-256: {checkpoints.file_hash(archive)}")


if __name__ == "__main__":
    main()
