"""Bounded real-training-image detector integration check; no accuracy evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from importlib.metadata import distributions, version
import inspect
import json
import math
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
from torch.utils.data import DataLoader

from rdd_data import TrainingDataset, inside, read_manifest, sha256, torchvision_collate, yolo_collate
from select_training_cases import make_selection


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
                          encoding="utf-8", newline="\n")


def tensor_hash(tensor):
    cpu = tensor.detach().cpu().contiguous()
    header = f"{cpu.dtype}:{tuple(cpu.shape)}:".encode("ascii")
    return sha256(header + cpu.numpy().tobytes())


def verified_package(package):
    names = []
    for line in (package / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in names, "Duplicate package checksum entry")
        require(sha256(inside(package, name).read_bytes()) == digest, f"Changed package file: {name}")
        names.append(name)
    required = {"integration.py", "protocol.json", "rdd_data.py", "reference.json",
                "selection.json", "select_training_cases.py"}
    require(required.issubset(names), "Required code/configuration is not covered by checksums")
    return names + ["SHA256SUMS"]


def check_output(out, protected):
    for path in protected:
        require(not (out == path or out.is_relative_to(path) or path.is_relative_to(out)),
                "Output must be separate from input and package directories")
    archive = out.with_name(out.name + ".tar.gz")
    if out.exists() or archive.exists():
        raise FileExistsError("Use a new output directory and archive; previous results are preserved")
    return archive


def paired_batch(examples):
    images, targets, metadata = torchvision_collate(examples)
    yolo, yolo_metadata = yolo_collate(examples)
    require(metadata == yolo_metadata, "Adapter metadata differs")
    require(torch.equal(torch.stack(images), yolo["img"]), "Adapters received different pixels")
    entries = []
    for index, (example, image, target) in enumerate(zip(examples, images, targets)):
        require(image.dtype == torch.float32 and torch.isfinite(image).all().item(), "Invalid pixels")
        require(image.min().item() >= 0 and image.max().item() <= 1, "Pixels outside [0, 1]")
        mask = yolo["batch_idx"] == index
        canonical = torch.tensor(example.labels, dtype=torch.int64)
        require(torch.equal(target["labels"].cpu() - 1, canonical), "Faster R-CNN labels differ")
        require(torch.equal(yolo["cls"][mask].flatten(), canonical.to(yolo["cls"].dtype)), "YOLO labels differ")
        require(len(target["boxes"]) == len(example.boxes) == int(mask.sum()), "Targets lost or added")
        require(not target["iscrowd"].any().item(), "A target was changed to crowd")
        xywh = yolo["bboxes"][mask] * example.geometry["size"]
        xyxy = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), dim=1)
        require(torch.allclose(xyxy, target["boxes"], atol=0.002, rtol=0), "Adapter boxes disagree")
        entries.append({"image": example.record["image"], "image_id": example.image_id,
                        "source_image_sha256": example.record["image_sha256"],
                        "prepared_tensor_sha256": tensor_hash(image),
                        "target_count": len(example.boxes), "canonical_labels": example.labels.tolist(),
                        "prepared_xyxy_sha256": tensor_hash(target["boxes"]), "geometry": example.geometry})
    signature = sha256(json.dumps(entries, sort_keys=True).encode("utf-8"))
    return images, targets, yolo, {"signature": signature, "images": entries}


def checked_update(model, optimizer, head, total, losses):
    require(isinstance(losses, dict) and bool(losses), "Expected named loss components")
    require(total.ndim == 0 and torch.isfinite(total).item(), "Nonfinite or nonscalar total loss")
    for name, loss in losses.items():
        require(torch.is_tensor(loss) and loss.numel() == 1 and torch.isfinite(loss).all().item(),
                f"Invalid loss: {name}")
    total.backward()
    gradients = 0
    for name, parameter in model.named_parameters():
        if parameter.grad is not None:
            require(torch.isfinite(parameter.grad).all().item(), f"Nonfinite gradient: {name}")
            gradients += 1
    require(head.grad is not None and head.grad.abs().sum().item() > 0, "Classifier received no gradient")
    before = head.detach().clone()
    optimizer.step()
    require(not torch.equal(before, head.detach()), "Classifier did not update")
    for name, parameter in model.named_parameters():
        require(torch.isfinite(parameter).all().item(), f"Nonfinite updated parameter: {name}")
    return {"total_loss": total.detach().item(),
            "loss_components": {name: loss.detach().item() for name, loss in losses.items()},
            "gradient_tensors_checked": gradients,
            "classifier_max_absolute_update": (head.detach() - before).abs().max().item()}


def prediction_summary(boxes, scores, labels, foreground_start, size=None):
    require(scores.ndim == 1 and boxes.shape == (len(scores), 4) and labels.shape == scores.shape,
            "Invalid prediction shapes")
    require(torch.isfinite(boxes).all().item() and torch.isfinite(scores).all().item(),
            "Nonfinite predictions")
    require(((scores >= 0) & (scores <= 1)).all().item(), "Invalid prediction score")
    require(((labels >= foreground_start) & (labels < foreground_start + 4)
             & (labels == labels.round())).all().item(), "Invalid foreground prediction ID")
    require((boxes[:, 2:] > boxes[:, :2]).all().item(), "Nonpositive prediction area")
    if size is not None:
        require(((boxes >= 0) & (boxes <= size)).all().item(), "Prediction outside prepared image")
    return {"detections": len(scores), "boxes_sha256": tensor_hash(boxes),
            "scores_sha256": tensor_hash(scores), "labels_sha256": tensor_hash(labels)}


def build_model(name, size, device):
    if name == "fasterrcnn_resnet50_fpn":
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=5,
                                       min_size=size, max_size=size).to(device)
        require(model.transform.min_size == (size,) and model.transform.max_size == size,
                "Unexpected Faster R-CNN resize settings")

        def check_transform(module, inputs, output):
            require(all(tuple(shape) == (size, size) for shape in output[0].image_sizes),
                    "Faster R-CNN resized the prepared image")
            require(tuple(output[0].tensors.shape[-2:]) == (size, size), "Unexpected image padding")
            if len(inputs) > 1 and inputs[1] is not None:
                require(len(inputs[1]) == len(output[1]), "Transform lost a target dictionary")
                for before, after in zip(inputs[1], output[1]):
                    require(torch.equal(before["boxes"], after["boxes"]), "Transform changed prepared boxes")

        model.transform.register_forward_hook(check_transform)
        return model, model.roi_heads.box_predictor.cls_score.weight, {
            "constructor": "fasterrcnn_resnet50_fpn", "weights": None, "weights_backbone": None,
            "num_classes": 5, "min_size": size, "max_size": size,
            "image_mean": model.transform.image_mean, "image_std": model.transform.image_std,
            "score_threshold": model.roi_heads.score_thresh, "nms_threshold": model.roi_heads.nms_thresh,
            "detections_per_image": model.roi_heads.detections_per_img}
    require(name == "yolov8n", "Unknown model")
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import DetectionModel
    model = DetectionModel("yolov8n.yaml", ch=3, nc=4, verbose=False).to(device)
    model.args = get_cfg()
    require(model.model[-1].nc == 4 and not model.end2end, "Unexpected YOLO head")
    return model, model.model[-1].cv3[0][-1].weight, {
        "constructor": "DetectionModel", "yaml": model.yaml, "weights": "random initialization",
        "loss_gains": {name: getattr(model.args, name) for name in ("box", "cls", "dfl")},
        "confidence_threshold": 0.0, "nms_iou_threshold": 0.5, "max_detections": 100,
        "confidence_note": "Zero threshold exercises NMS on random weights; not an evaluation setting"}


def model_step(model, head, name, examples, optimizer, device, size):
    images, targets, yolo, signature = paired_batch(examples)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    if name == "fasterrcnn_resnet50_fpn":
        inputs = [image.to(device) for image in images]
        target_inputs = [{key: value.to(device) for key, value in target.items()} for target in targets]
        losses = model(inputs, target_inputs)
        total = sum(losses.values())
    else:
        inputs = {key: value.to(device) for key, value in yolo.items()}
        vector, losses = model(inputs)
        require(torch.isfinite(vector).all().item(), "Nonfinite YOLO loss vector")
        total = vector.sum()
    update = checked_update(model, optimizer, head, total, losses)
    model.eval()
    with torch.inference_mode():
        if name == "fasterrcnn_resnet50_fpn":
            predictions = model(inputs)
            require(len(predictions) == len(examples), "Missing Faster R-CNN image output")
            summaries = [prediction_summary(p["boxes"], p["scores"], p["labels"], 1, size) for p in predictions]
        else:
            from ultralytics.utils.nms import non_max_suppression
            decoded = model(inputs["img"])[0]
            require(decoded.ndim == 3 and tuple(decoded.shape[:2]) == (len(examples), 8),
                    "Unexpected decoded YOLO shape")
            require(torch.isfinite(decoded).all().item() and (decoded[:, 2:4] > 0).all().item(),
                    "Invalid decoded YOLO boxes")
            require(((decoded[:, 4:] >= 0) & (decoded[:, 4:] <= 1)).all().item(), "Invalid decoded YOLO scores")
            predictions = non_max_suppression(decoded, conf_thres=0.0, iou_thres=0.5, nc=4,
                                             max_det=100, max_time_img=5.0)
            require(len(predictions) == len(examples), "Missing YOLO image output")
            summaries = []
            for prediction in predictions:
                require(prediction.ndim == 2 and prediction.shape[1] == 6 and 0 < len(prediction) <= 100,
                        "YOLO NMS returned an empty or invalid output at zero confidence")
                summaries.append(prediction_summary(prediction[:, :4], prediction[:, 4], prediction[:, 5], 0))
    return {"inputs": signature, "update": update, "inference": summaries,
            "target_counts": [len(example.boxes) for example in examples]}


def run_models(batches, protocol, out, report, device="cuda", builder=build_model, stepper=model_step):
    reference_signatures = []
    for name in protocol["models"]:
        random.seed(protocol["seed"])
        np.random.seed(protocol["seed"])
        torch.manual_seed(protocol["seed"])
        if device == "cuda":
            torch.cuda.manual_seed_all(protocol["seed"])
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        model_report = {"status": "running", "steps_completed": 0, "images_processed": 0}
        report["models"][name] = model_report
        write_json(out / "run.json", report)
        model, head, effective = builder(name, protocol["image_size"], device)
        model_report["effective_configuration"] = effective
        model_report["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        source = inspect.getsourcefile(type(model))
        model_report["model_class_source_sha256"] = sha256(Path(source).read_bytes()) if source else None
        optimizer = torch.optim.SGD(model.parameters(), lr=protocol["learning_rate"], momentum=0, weight_decay=0)
        signatures = []
        for index, examples in enumerate(batches):
            if device == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            result = stepper(model, head, name, examples, optimizer, device, protocol["image_size"])
            if device == "cuda":
                torch.cuda.synchronize()
            result.update(model=name, step=index + 1, diagnostic_wall_seconds=round(time.perf_counter() - started, 6))
            signatures.append(result["inputs"]["signature"])
            if reference_signatures:
                require(signatures[-1] == reference_signatures[index], "Models received different input batches")
            with (out / "steps.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
            model_report["steps_completed"] += 1
            model_report["images_processed"] += len(examples)
            write_json(out / "run.json", report)
            print(f"PASS {name} batch={index + 1}/{len(batches)} targets={result['target_counts']}", flush=True)
        require(model_report["images_processed"] == protocol["expected_images"], "Incomplete image coverage")
        require(model_report["steps_completed"] == math.ceil(protocol["expected_images"] / protocol["batch_size"]),
                "Incomplete batch coverage")
        if not reference_signatures:
            reference_signatures = signatures
        model_report["status"] = "complete"
        if device == "cuda":
            model_report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            model_report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        write_json(out / "run.json", report)
        # Drop the optimizer, classifier reference, and model before the next detector.
        del optimizer, head, model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    report["paired_input_signatures_match"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    files = verified_package(package)
    protocol = json.loads((package / "protocol.json").read_text(encoding="utf-8"))
    reference = json.loads((package / "reference.json").read_text(encoding="utf-8"))
    selection = json.loads((package / "selection.json").read_text(encoding="utf-8"))
    for name, digest in protocol["loader_source_sha256"].items():
        require(sha256(inside(package, name).read_bytes()) == digest, "Frozen loader source differs")
    require(os.environ.get("PBS_JOBID") and not socket.gethostname().lower().startswith("login"),
            "Run in an allocated GPU compute job")
    require(Path(sys.prefix).resolve() == Path(reference["expected_environment"]).resolve(),
            "Activate rdd_detection_v1")
    versions = {name: version(name) for name in protocol["versions"]}
    require(versions == protocol["versions"], f"Dependency versions changed: {versions}")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected one assigned CUDA GPU")
    root, splits, out = args.root.resolve(), args.splits.resolve(), args.out.resolve()
    archive = check_output(out, (root, splits, package))
    require(sha256((splits / "config.json").read_bytes()) == reference["config_sha256"], "Split config changed")
    rows = read_manifest(splits / "train.jsonl", reference["manifest_sha256"])
    require(len(rows) == reference["expected_training_images"]
            and sum(len(row["targets"]) for row in rows) == reference["expected_training_targets"],
            "Training population changed")
    require(selection == make_selection(rows, reference["manifest_sha256"]), "Selection changed")
    require(protocol["image_size"] == reference["image_size"]
            and protocol["batch_size"] == reference["batch_size"], "Prepared input convention changed")
    names = [entry["image"] for entry in selection["selected"]]
    require(len(names) == protocol["expected_images"], "Selected image count changed")
    dataset = TrainingDataset(root, splits / "train.jsonl", reference["manifest_sha256"], names, protocol["image_size"])
    torch.set_num_threads(protocol["cpu_threads"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    out.mkdir(parents=False, exist_ok=False)
    for name in files:
        destination = out / "code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inside(package, name), destination)
    report = {"status": "running", "purpose": protocol["purpose"], "protocol": protocol, "models": {},
              "started_utc": datetime.now(timezone.utc).isoformat(), "command": sys.argv,
              "pbs_job": os.environ["PBS_JOBID"], "host": socket.gethostname(),
              "python": platform.python_version(), "executable": sys.executable, "versions": versions,
              "cuda_build": torch.version.cuda, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "gpu_name": torch.cuda.get_device_name(0), "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "manifest_sha256": reference["manifest_sha256"], "selection_sha256": sha256((package / "selection.json").read_bytes()),
              "held_out_data_read": False, "image_previews_written": False, "checkpoints_saved": False,
              "accuracy_evaluated": False, "model_choice": "unresolved", "paired_input_signatures_match": False,
              "timing_note": "Diagnostic timings include validation and transfers; not a throughput comparison"}
    write_json(out / "run.json", report)
    write_json(out / "package_versions.json", {dist.metadata["Name"]: dist.version for dist in distributions()})
    started = time.perf_counter()
    try:
        batches = list(DataLoader(dataset, batch_size=protocol["batch_size"], shuffle=False,
                                  num_workers=0, collate_fn=list))
        require(sum(len(example.boxes) for batch in batches for example in batch) == protocol["expected_targets"],
                "Selected target count changed")
        require(any(all(len(example.boxes) == 0 for example in batch) for batch in batches),
                "No all-empty batch was exercised")
        report["all_empty_batches"] = [index + 1 for index, batch in enumerate(batches) if all(len(e.boxes) == 0 for e in batch)]
        run_models(batches, protocol, out, report)
        report["status"] = "complete"
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        for model_report in report["models"].values():
            if model_report["status"] == "running":
                model_report["status"] = "failed"
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(out / "run.json", report)
    with tarfile.open(archive, "x:gz") as bundle:
        bundle.add(out, arcname=out.name)
    print(f"REAL-IMAGE DETECTOR INTEGRATION: PASS ({len(protocol['models'])} models, "
          f"{protocol['expected_images']} training images each)")
    print(f"Archive: {archive}")
    print(f"Archive SHA-256: {sha256(archive.read_bytes())}")


if __name__ == "__main__":
    main()
