"""Bounded CPU check of real training images. No detector, previews, or metrics."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import sys
import tarfile
import time
import traceback

import numpy as np
import torch
from torch.utils.data import DataLoader

from rdd_data import (CLASSES, TrainingDataset, inside, read_manifest,
                      restore_boxes, sha256, torchvision_collate, yolo_collate)
from select_training_cases import make_selection


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def check_batch(examples):
    images, targets, tv_metadata = torchvision_collate(examples)
    yolo, yolo_metadata = yolo_collate(examples)
    if tv_metadata != yolo_metadata:
        raise ValueError("Adapter metadata disagrees")
    records = []
    for index, (example, image, target) in enumerate(zip(examples, images, targets)):
        if not torch.equal(image, yolo["img"][index]):
            raise ValueError("Adapters receive different image pixels")
        if not (torch.isfinite(image).all().item() and image.min().item() >= 0 and image.max().item() <= 1):
            raise ValueError("Invalid image tensor range")
        labels = example.labels
        if not np.array_equal(target["labels"].numpy(), labels + 1):
            raise ValueError("Faster R-CNN foreground mapping changed")
        if target["iscrowd"].any().item():
            raise ValueError("A target was silently marked crowd/ignored")
        mask = yolo["batch_idx"] == index
        if not np.array_equal(yolo["cls"][mask].numpy().reshape(-1), labels):
            raise ValueError("YOLO foreground mapping changed")
        original = np.array([t["box_xyxy"] for t in example.record["targets"]], dtype=np.float64).reshape(-1, 4)
        tv_original = restore_boxes(target["boxes"].numpy(), example.geometry)
        xywh = yolo["bboxes"][mask].numpy().astype(np.float64) * example.geometry["size"]
        xyxy = np.concatenate((xywh[:, :2] - xywh[:, 2:] / 2,
                               xywh[:, :2] + xywh[:, 2:] / 2), axis=1)
        yolo_original = restore_boxes(xyxy, example.geometry)
        if original.shape != tv_original.shape or original.shape != yolo_original.shape:
            raise ValueError("A target was lost or added")
        for recovered in (tv_original, yolo_original):
            if not np.allclose(recovered, original, rtol=0, atol=0.002):
                raise ValueError(f"Box round-trip error exceeds 0.002 original pixels: {example.record['image']}")
        records.append({
            "image": example.record["image"], "source": example.record["source"],
            "image_id": example.image_id,
            "image_sha256": example.record["image_sha256"],
            "xml_sha256": example.record["xml_sha256"],
            "rgb_sha256": example.record["rgb_sha256"],
            "annotation_category": example.record["annotation_category"],
            "exif_orientation": example.record["exif_orientation"],
            "targets": len(original), "labels": labels.tolist(),
            "difficult_targets": sum(str(t["difficult"]) == "1" for t in example.record["targets"]),
            "geometry": example.geometry,
            "torchvision_max_error_pixels": float(np.max(np.abs(tv_original - original))) if len(original) else 0.0,
            "yolo_max_error_pixels": float(np.max(np.abs(yolo_original - original))) if len(original) else 0.0,
            "checks": "source hashes, stored pixels, tensor equality, all targets, class mapping, inverse geometry passed",
        })
    return records


def verify_package(package):
    contents = (package / "SHA256SUMS").read_text(encoding="utf-8")
    for line in contents.splitlines():
        expected, name = line.split("  ", 1)
        if sha256(inside(package, name).read_bytes()) != expected:
            raise ValueError(f"Package checksum mismatch: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    verify_package(package)
    reference = json.loads((package / "reference.json").read_text(encoding="utf-8"))
    selection = json.loads((package / "selection.json").read_text(encoding="utf-8"))
    if not os.environ.get("PBS_JOBID") or socket.gethostname().lower().startswith("login"):
        raise SystemExit("STOP: run this image-decoding check in a CPU compute allocation.")
    if Path(sys.prefix).resolve() != Path(reference["expected_environment"]).resolve():
        raise SystemExit("STOP: activate the project rdd_detection_v1 environment.")
    versions = {name: version(name) for name in reference["versions"]}
    if versions != reference["versions"]:
        raise SystemExit(f"STOP: dependency versions differ from reference: {versions}")
    root, splits, out = args.root.resolve(), args.splits.resolve(), args.out.resolve()
    for protected in (root, splits, package):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError("Output must not contain, replace, or sit inside input directories")
    archive = out.with_name(out.name + ".tar.gz")
    if out.exists() or archive.exists():
        raise FileExistsError("Use a new output directory and archive name; existing results are preserved")
    if sha256((splits / "config.json").read_bytes()) != reference["config_sha256"]:
        raise ValueError("Split configuration checksum mismatch")
    manifest = splits / reference["manifest_name"]
    rows = read_manifest(manifest, reference["manifest_sha256"])
    if len(rows) != reference["expected_training_images"] or sum(len(r["targets"]) for r in rows) != reference["expected_training_targets"]:
        raise ValueError("Training manifest coverage differs from reference")
    if selection != make_selection(rows, reference["manifest_sha256"]):
        raise ValueError("Saved selection differs from deterministic training-only selection")
    names = [entry["image"] for entry in selection["selected"]]
    dataset = TrainingDataset(root, manifest, reference["manifest_sha256"], names, reference["image_size"])
    torch.set_num_threads(1)
    out.mkdir(parents=False, exist_ok=False)
    code = out / "code"
    code.mkdir()
    for path in package.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
            destination = code / path.relative_to(package)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    started = time.perf_counter()
    report = {
        "status": "running", "purpose": "CPU data-loader software verification only",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "pbs_job": os.environ["PBS_JOBID"], "host": socket.gethostname(),
        "python": platform.python_version(), "executable": sys.executable,
        "versions": versions, "command": sys.argv,
        "raw_root": str(root), "manifest_sha256": reference["manifest_sha256"],
        "selection_sha256": sha256((package / "selection.json").read_bytes()),
        "images_expected": len(dataset), "images_checked": 0,
        "image_size": reference["image_size"], "batch_size": reference["batch_size"],
        "num_workers": 0, "required_features": selection["required_features"],
        "held_out_data_read": False, "detector_execution": False,
        "image_previews_written": False, "model_choice": "unresolved",
    }
    write_json(out / "run.json", report)
    checked = []
    try:
        loader = DataLoader(dataset, batch_size=reference["batch_size"], shuffle=False,
                            num_workers=0, collate_fn=list)
        with (out / "samples.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
            for examples in loader:
                batch_records = check_batch(examples)
                for record in batch_records:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
                    print(f"PASS {record['image']} targets={record['targets']}", flush=True)
                handle.flush()
                checked.extend(batch_records)
                report["images_checked"] = len(checked)
        if len(checked) != len(dataset):
            raise ValueError("Not all selected images were checked")
        report.update(
            status="complete", targets_checked=sum(r["targets"] for r in checked),
            sources=dict(Counter(r["source"] for r in checked)),
            categories=dict(Counter(r["annotation_category"] for r in checked)),
            classes=dict(Counter(CLASSES[label] for r in checked for label in r["labels"])),
            max_original_pixel_round_trip_error=max(max(r["torchvision_max_error_pixels"], r["yolo_max_error_pixels"]) for r in checked),
        )
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["elapsed_seconds"] = round(time.perf_counter() - started, 3)
        write_json(out / "run.json", report)
    with tarfile.open(archive, "x:gz") as bundle:
        bundle.add(out, arcname=out.name)
    print(f"TRAINING DATA LOADER CHECK: PASS ({len(checked)} images)")
    print(f"Archive: {archive}")
    print(f"Archive SHA-256: {sha256(archive.read_bytes())}")


if __name__ == "__main__":
    main()
