"""Bounded pretrained learning check on 15 training images; never opens held-out data."""
import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import distributions, version
import inspect
import json
import os
from pathlib import Path
import platform
import random
import shutil
import socket
import sys
import tarfile
import time
import traceback

import numpy as np
import torch

import checkpoints
from integration import (check_output, checked_update, paired_batch, require,
                         verified_package, write_json)
import metrics
from models import build_model, infer
from rdd_data import TrainingDataset, inside, read_manifest, sha256
from select_training_cases import make_selection


def verify_weights(directory, protocol):
    actual = {}
    for name, spec in protocol["pretrained_weights"].items():
        path = inside(directory, spec["filename"])
        require(path.stat().st_size == spec["bytes"], f"Weight size differs: {name}")
        digest = checkpoints.file_hash(path)
        require(digest == spec["sha256"], f"Weight checksum differs: {name}")
        actual[name] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    declared = {}
    for line in (directory / "SHA256SUMS").read_text().splitlines():
        digest, filename = line.split("  ", 1)
        require(filename not in declared, "Duplicate weight checksum entry")
        declared[filename] = digest
    require(declared == {s["filename"]: s["sha256"] for s in protocol["pretrained_weights"].values()},
            "Weight checksum manifest differs")
    return actual


def optimizer_for(model, protocol):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                           lr=protocol["learning_rate"], momentum=protocol["momentum"],
                           weight_decay=protocol["weight_decay"])


def train_step(model, head, name, examples, optimizer, device):
    images, targets, yolo, inputs = paired_batch(examples)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if name == "fasterrcnn_resnet50_fpn":
        losses = model([v.to(device) for v in images],
                       [{k: v.to(device) for k, v in target.items()} for target in targets])
        total = sum(losses.values())
    else:
        vector, losses = model({k: v.to(device) for k, v in yolo.items()})
        require(torch.isfinite(vector).all().item(), "Nonfinite YOLO loss vector")
        total = vector.sum()
    return {"inputs": inputs, "update": checked_update(model, optimizer, head, total, losses)}


def evaluate_model(model, name, batches, evaluation, device, gt, predictor=infer):
    predictions, clipping = [], []
    for examples in batches:
        images, _, _, _ = paired_batch(examples)
        batch_predictions = predictor(model, name, images, evaluation, device)
        require(len(batch_predictions) == len(examples), "Missing evaluation image")
        for example, prediction in zip(examples, batch_predictions):
            require(len(prediction["scores"]) <= evaluation["max_detections"], "Too many detections")
            records, info = metrics.convert_prediction(example, prediction)
            predictions.extend(records)
            clipping.append({"image_id": example.image_id, **info})
    values = metrics.evaluate(gt, predictions, evaluation["negative_confidence"])
    return {"metrics": values, "clipping": clipping}, predictions


def compare_predictions(before, after):
    require(len(before) == len(after), "Checkpoint reload changed prediction count")
    maximum_box_error, maximum_score_error = 0.0, 0.0
    for first, second in zip(before, after):
        require(first["image_id"] == second["image_id"]
                and first["category_id"] == second["category_id"],
                "Checkpoint reload changed ordered prediction IDs")
        a, b = np.asarray(first["bbox"]), np.asarray(second["bbox"])
        require(np.allclose(a, b, atol=0.001, rtol=1e-5), "Checkpoint reload changed prediction boxes")
        require(np.isclose(first["score"], second["score"], atol=1e-6, rtol=1e-5),
                "Checkpoint reload changed prediction scores")
        maximum_box_error = max(maximum_box_error, float(np.max(np.abs(a-b))))
        maximum_score_error = max(maximum_score_error, abs(first["score"]-second["score"]))
    return {"predictions_match_within_tolerance": True,
            "maximum_original_coordinate_error": maximum_box_error,
            "maximum_score_error": maximum_score_error}


def learning_review(initial, final, rule):
    require(initial["AP50"] is not None and final["AP50"] is not None, "AP50 undefined")
    gain = final["AP50"] - initial["AP50"]
    sufficient = (final["AP50"] >= rule["minimum_final_training_AP50"]
                  and gain >= rule["minimum_training_AP50_gain"])
    return {"initial_training_AP50": initial["AP50"], "final_training_AP50": final["AP50"],
            "training_AP50_gain": gain,
            "decision": "learning_signal_observed" if sufficient else "needs_review",
            "scope": "Engineering review rule on training examples; no generalization or model-ranking claim"}


def run_models(examples, protocol, context, out, report, weights, device="cuda",
               builder=build_model, trainer=train_step, predictor=infer):
    batches = [examples[i:i+protocol["batch_size"]] for i in range(0, len(examples), protocol["batch_size"])]
    signatures = [paired_batch(batch)[3] for batch in batches]
    write_json(out / "inputs.json", signatures)
    gt = metrics.ground_truth(examples)
    write_json(out / "ground_truth.json", gt)
    (out / "checkpoints").mkdir(exist_ok=False)
    checkpoint_records = []
    for name in protocol["models"]:
        random.seed(protocol["seed"])
        np.random.seed(protocol["seed"])
        torch.manual_seed(protocol["seed"])
        if device == "cuda":
            torch.cuda.manual_seed_all(protocol["seed"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        status = {"status": "running", "steps_completed": 0, "image_presentations": 0,
                  "evaluations": [], "checkpoint_reloads": []}
        report["models"][name] = status
        write_json(out / "run.json", report)
        model, head, effective = builder(name, protocol["image_size"], device,
                                         protocol["evaluation"], Path(weights[name]["path"]))
        status["effective_configuration"] = effective
        source = inspect.getsourcefile(type(model))
        status["model_class_source_sha256"] = checkpoints.file_hash(Path(source)) if source else None
        optimizer = optimizer_for(model, protocol)
        for step in range(protocol["total_steps"] + 1):
            if step:
                batch_index = (step - 1) % len(batches)
                result = trainer(model, head, name, batches[batch_index], optimizer, device)
                require(result["inputs"] == signatures[batch_index], "Training inputs changed")
                with (out / "steps.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps({"model": name, "step": step, "batch_index": batch_index,
                                             "input_signature": result["inputs"]["signature"],
                                             "update": result["update"]}, allow_nan=False) + "\n")
                status["steps_completed"] = step
                status["image_presentations"] += len(batches[batch_index])
            if step in protocol["evaluate_steps"]:
                result, predictions = evaluate_model(model, name, batches, protocol["evaluation"],
                                                      device, gt, predictor)
                result.update(step=step, model=name)
                stem = f"{name}_step_{step:03d}"
                write_json(out / f"{stem}_predictions.json", predictions)
                write_json(out / f"{stem}_metrics.json", result)
                status["evaluations"].append({"step": step, **result["metrics"]})
                print(f"EVAL {name} step={step}/{protocol['total_steps']} "
                      f"training_AP50={result['metrics']['AP50']:.6f} "
                      f"training_AP50:95={result['metrics']['AP50_95']:.6f}", flush=True)
            if step in protocol["checkpoint_steps"]:
                path = out / "checkpoints" / f"{name}_step_{step:03d}.pt"
                saved = checkpoints.save(path, model, optimizer, name, step, context)
                checkpoint_records.append(saved)
                write_json(out / "checkpoint_records.json", checkpoint_records)
                del optimizer, head, model
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
                model, head, _ = builder(name, protocol["image_size"], device, protocol["evaluation"], None)
                optimizer = optimizer_for(model, protocol)
                restored = checkpoints.restore(path, saved, model, optimizer, name, context)
                restored.update(completed_steps=step, checkpoint=saved["file"])
                if step == protocol["total_steps"]:
                    _, replay = evaluate_model(model, name, batches, protocol["evaluation"], device, gt, predictor)
                    restored.update(compare_predictions(predictions, replay))
                    write_json(out / f"{name}_reload_predictions.json", replay)
                status["checkpoint_reloads"].append(restored)
                print(f"CHECKPOINT {name} step={step}: exact model/optimizer/RNG restore", flush=True)
            if step % 10 == 0:
                write_json(out / "run.json", report)
        require(status["steps_completed"] == protocol["total_steps"], "Incomplete updates")
        status["learning_review"] = learning_review(status["evaluations"][0], status["evaluations"][-1],
                                                     protocol["decision_rule"])
        status["status"] = "complete"
        if device == "cuda":
            status["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            status["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(out / "run.json", report)
        del optimizer, head, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    report["paired_inputs_verified"] = True


def archive_records(out, archive):
    if archive.exists() or (out / "checksums.json").exists():
        raise FileExistsError("Existing result checksums/archive are preserved")
    files = [p for p in out.rglob("*") if p.is_file() and not p.is_relative_to(out / "checkpoints")]
    for path in files:
        require(not path.is_symlink(), "Unexpected result symlink")
    write_json(out / "checksums.json", {p.relative_to(out).as_posix(): checkpoints.file_hash(p) for p in files})
    files.append(out / "checksums.json")
    with tarfile.open(archive, "x:gz") as bundle:
        for path in sorted(files):
            bundle.add(path, arcname=f"{out.name}/{path.relative_to(out).as_posix()}", recursive=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "splits", "weights", "out"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    files = verified_package(package)
    require({"learning.py", "models.py", "metrics.py", "checkpoints.py"}.issubset(files),
            "New learning sources missing from checksums")
    protocol = json.loads((package / "protocol.json").read_text())
    reference = json.loads((package / "reference.json").read_text())
    selection = json.loads((package / "selection.json").read_text())
    require(protocol["total_steps"] == 200 and protocol["evaluate_steps"] == [0,50,100,200]
            and protocol["checkpoint_steps"] == [100,200], "Bounded schedule changed")
    require(protocol["models"] == ["fasterrcnn_resnet50_fpn", "yolov8n"], "Model list changed")
    for name, digest in protocol["loader_source_sha256"].items():
        require(checkpoints.file_hash(inside(package, name)) == digest, "Frozen loader changed")
    require(os.environ.get("PBS_JOBID") and not socket.gethostname().lower().startswith("login"),
            "Use an allocated GPU compute job")
    require(Path(sys.prefix).resolve() == Path(reference["expected_environment"]).resolve(),
            "Activate rdd_detection_v1")
    versions = {name: version(name) for name in protocol["versions"]}
    require(versions == protocol["versions"], f"Dependencies changed: {versions}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one assigned CUDA GPU")
    root, splits, weights_dir, out = (getattr(args, k).resolve() for k in ("root", "splits", "weights", "out"))
    archive = check_output(out, (root, splits, weights_dir, package))
    weights = verify_weights(weights_dir, protocol)
    require(checkpoints.file_hash(splits / "config.json") == reference["config_sha256"], "Split config changed")
    rows = read_manifest(splits / "train.jsonl", reference["manifest_sha256"])
    require(len(rows) == reference["expected_training_images"]
            and sum(len(r["targets"]) for r in rows) == reference["expected_training_targets"], "Training population changed")
    require(selection == make_selection(rows, reference["manifest_sha256"]), "Selection changed")
    require(protocol["image_size"] == reference["image_size"]
            and protocol["batch_size"] == reference["batch_size"], "Input geometry changed")
    names = [item["image"] for item in selection["selected"]]
    require(len(names) == protocol["expected_images"], "Selection count changed")
    dataset = TrainingDataset(root, splits / "train.jsonl", reference["manifest_sha256"], names, protocol["image_size"])
    context = {"protocol_sha256": checkpoints.file_hash(package / "protocol.json"),
               "manifest_sha256": reference["manifest_sha256"],
               "selection_sha256": checkpoints.file_hash(package / "selection.json"),
               "package_checksums_sha256": checkpoints.file_hash(package / "SHA256SUMS"),
               "weights_sha256": {k: v["sha256"] for k, v in weights.items()}}
    torch.set_num_threads(protocol["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out.mkdir(exist_ok=False)
    for name in files:
        destination = out / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(package, name), destination)
    report = {"status": "running", "protocol": protocol, "context": context, "weights": weights,
              "models": {}, "started_utc": datetime.now(timezone.utc).isoformat(),
              "command": sys.argv, "pbs_job": os.environ["PBS_JOBID"], "host": socket.gethostname(),
              "python": platform.python_version(), "executable": sys.executable, "versions": versions,
              "cuda_build": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_name": torch.cuda.get_device_name(0), "held_out_data_read": False,
              "model_choice": "unresolved", "paired_inputs_verified": False,
              "archive_scope": "Records and source only; full checkpoints retained on CCAST",
              "limitations": "Training subset only; no architecture ranking, generalization or augmentation efficacy claim"}
    write_json(out / "package_versions.json", {dist.metadata["Name"]: dist.version for dist in distributions()})
    write_json(out / "run.json", report)
    started = time.perf_counter()
    try:
        examples = [dataset[i] for i in range(len(dataset))]
        require(sum(len(e.boxes) for e in examples) == protocol["expected_targets"], "Target coverage changed")
        require(set(int(c) for e in examples for c in e.labels) == {0,1,2,3}, "A class is missing")
        run_models(examples, protocol, context, out, report, weights)
        report["status"] = "complete"
        report["learning_review"] = {k: v["learning_review"]["decision"] for k, v in report["models"].items()}
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        for state in report["models"].values():
            if state["status"] == "running":
                state["status"] = "failed"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(out / "run.json", report)
    archive_records(out, archive)
    print("TRAINING-ONLY LEARNING CHECK: COMPLETE (inspect learning_review before a larger run)")
    print(f"Archive: {archive}")
    print(f"Archive SHA-256: {checkpoints.file_hash(archive)}")


if __name__ == "__main__":
    main()
