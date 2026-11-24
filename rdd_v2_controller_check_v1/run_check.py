"""Bounded GPU diagnostic; no training, held-out evaluation or automatic follow-up."""
import argparse
import copy
import json
import os
from pathlib import Path
import platform
import sys
import tarfile
import time
from importlib.metadata import version

import numpy as np
from PIL import Image, ImageDraw

import controller
import diagnostic as diag
import geometry
import rdd_data


def output_paths(out, protected):
    out = out.resolve()
    archive = out.with_name(out.name+".tar.gz")
    for path in protected:
        path = path.resolve()
        diag.require(out != path and not out.is_relative_to(path) and not path.is_relative_to(out), "Output overlaps inputs")
    diag.require(not out.exists() and not archive.exists(), "Output/archive already exists")
    diag.require(out.parent.is_dir(), "Output parent missing")
    return out, archive


def archive_results(out, archive):
    diag.require(not archive.exists(), "Archive already exists")
    lines = [f"{diag.file_hash(p)}  {p.relative_to(out).as_posix()}"
             for p in sorted(out.rglob("*")) if p.is_file()]
    with (out/"SHA256SUMS").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines)+"\n")
    temporary = archive.with_suffix(archive.suffix+".partial")
    diag.require(not temporary.exists(), "Partial archive already exists")
    with tarfile.open(temporary, "x:gz") as tar:
        tar.add(out, arcname=out.name)
    # The parent job is exclusive and output path is new; never replace an archive.
    temporary.rename(archive)
    return diag.file_hash(archive)


def overlay(rgb, annotations, title):
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    for a in annotations:
        box = a["box_xyxy"]
        draw.rectangle(box, outline="black", width=3)
        draw.rectangle(box, outline="white", width=1)
        original = a["original"]
        label = f'{original.get("label", "other")} #{a["id"]}'
        x, y = box[:2]
        draw.text((x+2, y+2), label, fill="white", stroke_width=1, stroke_fill="black")
    draw.rectangle((0, 0, image.width, 19), fill="white")
    draw.text((4, 3), title, fill="black")
    return image


def save_sheet(out, selection, images, annotations):
    keys = ("reference", "brightness_down/3/0", "gaussian_blur/3/0", "gaussian_noise/3/0")
    sheet = Image.new("RGB", (1280, 1280), "white")
    for j, key in enumerate(keys):
        tile = overlay(images[key], annotations, key)
        sheet.paste(tile, ((j % 2)*640, (j//2)*640))
    filename = f'panel_{selection["panel"]}_anchor_{selection["anchor_class"]}.png'
    sheet.save(out/"previews"/filename)
    return {"file": "previews/"+filename, "image": selection["image"],
            "anchor_class": selection["anchor_class"], "object_indices": selection["object_indices"]}


def thin_case(root, rows, cfg, out):
    name = cfg["thin_case"]["image"]
    row = next(r for r in rows if r["image"] == name)
    original = rdd_data.load_example(root, row)
    prepared = rdd_data.prepare_example(original, cfg["geometry"]["size"])
    annotations = geometry.prepared_annotations(row, 640)
    candidate = next(c for c in geometry.bank(cfg["geometry"]) if not c["flip"]
                     and c["resized_size"] == 608 and c["tx"] == c["ty"] == 0)
    transformed, updated, verdict = geometry.apply(prepared.rgb, annotations, candidate)
    diag.require(verdict["accepted"], "Thin-case geometry unexpectedly rejected")
    folder = out/"thin_case"
    folder.mkdir()
    for label, rgb, boxes in (("prepared", prepared.rgb, annotations), ("scale095", transformed, updated)):
        Image.fromarray(rgb).save(folder/(label+"_rgb.png"))
        overlay(rgb, boxes, label).save(folder/(label+"_boxes.png"))
    target = next(t for t in row["targets"] if t["object_index"] == cfg["thin_case"]["object_index"])
    x1, y1, x2, y2 = target["box_xyxy"]
    left, top = max(0, int(np.floor(x1))-64), max(0, int(np.floor(y1))-64)
    right, bottom = min(row["width"], int(np.ceil(x2))+64), min(row["height"], int(np.ceil(y2))+64)
    Image.fromarray(original.rgb[top:bottom, left:right]).save(folder/"raw_crop_rgb.png")
    record = {"image": name, "target": target, "raw_crop_xyxy": [left, top, right, bottom],
              "prepared_annotations": annotations, "transformed_annotations": updated,
              "candidate": candidate, "outside_panel_statistics": True,
              "limits": "Pixel/annotation QA only; raw crop contains stored RGB pixels, not an interpolated zoom"}
    diag.write_json(folder/"record.json", record)
    return record


def collect(detector, root, rows, plan, cfg, out, deadline):
    lookup = {r["image"]: r for r in rows}
    records = {"A": [], "B": []}
    previewed, preview_index = set(), []
    parity = None
    for panel in ("A", "B"):
        for number, selection in enumerate(plan["panels"][panel], 1):
            diag.require(time.monotonic() < deadline, "Administrative time budget exhausted; results incomplete")
            row = lookup[selection["image"]]
            example = rdd_data.prepare_example(rdd_data.load_example(root, row), cfg["geometry"]["size"])
            p = {"image": row["image"], "source_cluster_id": row["source_cluster_id"],
                 "classes": [t["class_index"] for t in row["targets"]],
                 "object_indices": [t["object_index"] for t in row["targets"]],
                 "target_positions": selection["target_positions"], "selection": selection,
                 "views": {}}
            view_list = list(diag.action_views(example, selection, cfg))
            diag.require(len(view_list) == cfg["views_per_image"], "View budget mismatch")
            if parity is None:
                rgb = view_list[1][1]
                before = detector.predict([rgb, rgb])
                after = detector.predict([rgb, rgb])
                parity = {"max_abs_error": diag.check_prediction_parity(before, after, cfg["parity_tolerance"]),
                          "extra_detector_images": 4, "tolerance": cfg["parity_tolerance"]}
            saved_images = {}
            qa = (panel, selection["anchor_class"]) not in previewed
            for start in range(0, len(view_list), cfg["batch_size"]):
                diag.require(time.monotonic() < deadline, "Administrative time budget exhausted; results incomplete")
                batch = view_list[start:start+cfg["batch_size"]]
                outputs = detector.predict([pixels for key, pixels, meta in batch])
                diag.require(len(outputs) == len(batch), "Detector batch length mismatch")
                for (key, pixels, meta), prediction in zip(batch, outputs):
                    observation = diag.match_prediction(prediction, meta["boxes"], meta["labels"], cfg["controller"])
                    observation.update(prediction=prediction, pixel_sha256=rdd_data.sha256(pixels.tobytes()), render=meta)
                    if key in ("clean", "reference"):
                        p[key] = observation
                        if key == "reference":
                            p["geometry"] = {k: meta[k] for k in ("candidate", "accepted", "reason", "cropped_ids")}
                    else:
                        p["views"][key] = observation
                    if qa and key in ("reference", "brightness_down/3/0", "gaussian_blur/3/0", "gaussian_noise/3/0"):
                        saved_images[key] = pixels
            if qa:
                preview_index.append(save_sheet(out, selection, saved_images, p["reference"]["render"]["annotations"]))
                previewed.add((panel, selection["anchor_class"]))
            diag.write_json(out/"observations"/f"{panel}_{number:03d}.json", p)
            records[panel].append(p)
            if number == 1 or number % 16 == 0:
                print(f"PANEL {panel} images={number}/{len(plan['panels'][panel])} detector_images={detector.images_forwarded}", flush=True)
    return records, preview_index, parity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    package = Path(__file__).resolve().parent
    hashes = diag.verify_package(package)
    cfg = json.loads((package/"protocol.json").read_text())
    out, archive = output_paths(args.out, [args.root, args.splits, args.checkpoint.parent, package])
    diag.require(os.environ.get("PBS_JOBID"), "Run within a PBS allocation")
    diag.require(Path(sys.prefix).resolve() == Path(cfg["expected_environment"]).resolve(), "Wrong Python environment")
    versions = {k: version(k) for k in cfg["versions"]}
    diag.require(versions == cfg["versions"], "Pinned dependency versions differ")
    import torch
    diag.require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected exactly one allocated CUDA GPU")
    torch.set_num_threads(cfg["cpu_threads"])
    rows = rdd_data.read_manifest(args.splits/"train.jsonl", cfg["train_manifest_sha256"])
    diag.require(len(rows) == cfg["expected_train_images"], "Training size differs")
    plan = diag.plan_panels(rows, cfg)
    diag.require(plan == json.loads((package/"panel_plan.json").read_text()), "Panel plan changed")
    from detector import FrozenDetector
    detector = FrozenDetector(args.checkpoint, cfg)
    out.mkdir()
    try:
        (out/"observations").mkdir()
        (out/"previews").mkdir()
        diag.write_json(out/"protocol.json", cfg)
        diag.write_json(out/"panel_plan.json", plan)
        diag.write_json(out/"context.json", {"package_hashes": hashes, "versions": versions,
            "python": sys.version, "host": platform.node(), "pbs_job": os.environ["PBS_JOBID"],
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "gpu": torch.cuda.get_device_name(0),
            "cuda_build": torch.version.cuda, "cudnn_version": torch.backends.cudnn.version(),
            "checkpoint": cfg["checkpoint"], "checkpoint_model_sha256": detector.checkpoint_model_sha256,
            "cudnn_benchmark": False, "cudnn_deterministic": True, "amp": False, "tf32": False})
        thin_case(args.root, rows, cfg, out)
        with detector.frozen():
            records, previews, parity = collect(detector, args.root, rows, plan, cfg, out,
                                                started+cfg["administrative_seconds"])
        diag.require(detector.images_forwarded == cfg["max_detector_images"], "Incomplete detector forward count")
        analysis = diag.analyze(records, rows, cfg)
        analysis.update(detector_audit=detector.audit, parity=parity,
                        detector_images=detector.images_forwarded, forward_calls=detector.forward_calls,
                        forward_seconds=detector.forward_seconds, elapsed_seconds=time.monotonic()-started,
                        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                        thin_case_is_additional_qa=True)
        diag.write_json(out/"preview_index.json", previews)
        diag.write_json(out/"result.json", analysis)
        checksum = archive_results(out, archive)
    except BaseException as exc:
        diag.write_json(out/"failure.json", {"status": "incomplete", "error": repr(exc),
                                          "detector_images": detector.images_forwarded})
        raise
    for name, result in analysis["panels"].items():
        for c, cell in result["policy"]["pooled"].items():
            print(f"SUPPORT panel={name} class={c} detected={cell['detected_objects']} clusters={cell['detected_clusters']} supported={cell['supported']} allowed={sum(cell['allowed'])}")
    print("FROZEN CONTROLLER CHECK: COMPLETE; inspect support, stability and previews before training.")
    print("Archive:", archive)
    print("Archive SHA-256:", checksum)


if __name__ == "__main__":
    main()
