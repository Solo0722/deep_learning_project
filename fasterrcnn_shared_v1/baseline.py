"""Matched fixed shared-augmentation Faster R-CNN baseline. No test data access."""
import argparse
from contextlib import contextmanager
import copy
import csv
from datetime import datetime, timezone
import gzip
import io
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import signal
import socket
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

import checkpoints
from integration import require
import metrics
import shared_augmentation as aug
from models import build_model, infer
from rdd_data import inside, load_example, prepare_example, read_manifest, sha256, torchvision_collate

PACKAGE = Path(__file__).resolve().parent
MODEL = "fasterrcnn_resnet50_fpn"


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def atomic_bytes(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_json(path, value):
    atomic_bytes(path, json_bytes(value))


def verify_package():
    sums = {}
    for line in (PACKAGE / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        require(name not in sums and checkpoints.file_hash(inside(PACKAGE, name)) == expected,
                f"Package file changed: {name}")
        sums[name] = expected
    require({"baseline.py", "shared_augmentation.py", "identity_protocol.json", "protocol.json", "smoke_selection.json", "rdd_data.py", "metrics.py", "models.py", "checkpoints.py"} <= sums.keys(),
            "Incomplete package manifest")
    return {"SHA256SUMS": checkpoints.file_hash(PACKAGE / "SHA256SUMS"), **sums}


class Dataset:
    def __init__(self, root, rows, split, size, selection=None):
        require(split in ("train", "val"), "Only train and validation are permitted")
        self.root, self.split, self.size = Path(root), split, size
        self.policy, self.seed, self.epoch = None, 0, 0
        require(all(r["split"] == split for r in rows), "Wrong partition in dataset")
        selected = {r["image"] for r in rows} if selection is None else set(selection)
        require(selection is None or len(selected) == len(selection), "Duplicate selected image")
        self.rows = [(i, r) for i, r in enumerate(rows) if r["image"] in selected]
        require(len(self.rows) == len(selected) and self.rows, "Missing or empty selection")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image_id, row = self.rows[index]
        example = prepare_example(load_example(self.root, row, image_id, self.split), self.size)
        return aug.apply_example(example, aug.decision(row, self.policy, self.seed, self.epoch))

    def ground_truth(self):
        gt = metrics.ground_truth([SimpleNamespace(image_id=i, record=r) for i, r in self.rows])
        gt["info"]["description"] = f"Frozen RDD detection {self.split} partition"
        return gt


def verify_partitions(train, val):
    for key in ("image", "source_cluster_id", "rgb_sha256"):
        require(not {r[key] for r in train} & {r[key] for r in val}, f"Train/validation overlap: {key}")


def epoch_batches(length, batch_size, seed, epoch, start=0):
    generator = torch.Generator().manual_seed(seed + 1000003 * epoch)
    order = torch.randperm(length, generator=generator).tolist()
    batches = [order[i:i+batch_size] for i in range(0, length, batch_size)]
    require(0 <= start <= len(batches), "Invalid resume batch")
    return batches[start:]


def collate(examples):
    images, targets, _ = torchvision_collate(examples)
    return images, targets, examples


def worker_init(worker_id):
    torch.set_num_threads(1)
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def loader(dataset, batches, cfg, epoch, validation=False):
    dataset.policy = None if validation else cfg.get("augmentation_policy")
    dataset.seed, dataset.epoch = cfg["seed"], epoch
    # Worker creation consumes only this generator, never the model RNG stream.
    generator = torch.Generator().manual_seed(cfg["seed"] + 2000003 * epoch + int(validation))
    options = {"prefetch_factor": 1} if cfg["workers"] else {}
    return DataLoader(dataset, batch_sampler=batches, num_workers=cfg["workers"],
                      collate_fn=collate, worker_init_fn=worker_init, generator=generator, **options)


def learning_rate(cfg, step, epoch):
    factor = cfg["lr_gamma"] ** sum(epoch >= e for e in cfg["lr_decay_epochs"])
    if cfg["warmup_steps"] and step < cfg["warmup_steps"]:
        factor *= cfg["warmup_start_factor"] + (1-cfg["warmup_start_factor"]) * step/cfg["warmup_steps"]
    return cfg["learning_rate"] * factor


def update(model, optimizer, images, targets, device):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = model([im.to(device) for im in images],
                   [{k: v.to(device) for k, v in t.items()} for t in targets])
    require(losses and all(v.numel() == 1 and torch.isfinite(v).item() for v in losses.values()), "Nonfinite/malformed training loss")
    total = sum(losses.values())
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    require(grads, "No model gradients")
    # Finite checks do not clip or rescale the update.
    grad_norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.detach()) for g in grads]))
    require(torch.isfinite(grad_norm).item(), "Nonfinite training gradient")
    optimizer.step()
    return {k: float(v.detach().item()) for k, v in losses.items()}, float(grad_norm.item())


def validate(model, dataset, cfg, device, should_stop=lambda: False):
    gt, predictions, clipping = dataset.ground_truth(), [], []
    batches = [list(range(i, min(i+cfg["validation_batch_size"], len(dataset))))
               for i in range(0, len(dataset), cfg["validation_batch_size"])]
    visited = []
    for images, _, examples in loader(dataset, batches, cfg, 0, validation=True):
        if should_stop():
            return None
        outputs = infer(model, MODEL, images, cfg["evaluation"], device)
        require(len(outputs) == len(examples), "Validation image missing")
        for example, prediction in zip(examples, outputs):
            require(len(prediction["scores"]) <= cfg["evaluation"]["max_detections"], "Too many validation predictions")
            records, info = metrics.convert_prediction(example, prediction)
            predictions.extend(records)
            clipping.append({"image_id": example.image_id, **info})
            visited.append(example.image_id)
    require(visited == [i for i, _ in dataset.rows], "Validation coverage/order differs")
    values = metrics.evaluate(gt, predictions, cfg["evaluation"]["negative_confidence"])
    values["scope"] = "Frozen validation partition; unaugmented stored images; test data not read"
    require(values["AP50_95"] is not None, "Validation AP is undefined")
    return {"metrics": values, "predictions": predictions, "clipping": clipping}


def initial_progress():
    return {"epoch": 0, "next_batch": 0, "step": 0, "images_seen": 0,
            "epoch_images": 0, "loss_sums": {}, "history": [], "best": None,
            "augmentation_trace": aug.empty_trace()}


def write_history(out, history):
    write_json(out / "history.json", history)
    if not history:
        return
    table = []
    for item in history:
        row = {"epoch": item["epoch"], "step": item["step"], "images_seen": item["images_seen"],
               "train_total_loss": sum(item["training_loss"].values()),
               **{f"val_{key}": item["validation"][key] for key in ("AP50_95", "AP50", "AP75", "AR100")}}
        for name, values in item["validation"]["per_class"].items():
            row[f"val_{name}_AP50_95"] = values["AP50_95"]
            row[f"val_{name}_AP50"] = values["AP50"]
        table.append(row)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(table[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(table)
    atomic_bytes(out / "history.csv", buffer.getvalue().encode())


def commit(out, model, optimizer, progress, context, label):
    require(all(torch.isfinite(p).all().item() for p in model.parameters()), "Nonfinite checkpoint parameters")
    name = f"{label}_step_{progress['step']:08d}_{uuid.uuid4().hex[:12]}.pt"
    if label.startswith("epoch_"):
        progress["history"][-1]["checkpoint_file"] = name
        if progress["best"]["epoch"] == progress["epoch"]:
            progress["best"]["checkpoint_file"] = name
    payload = {"format": 1, "context": context, "progress": copy.deepcopy(progress),
               "model": checkpoints.cpu_tree(model.state_dict()),
               "optimizer": checkpoints.cpu_tree(optimizer.state_dict()), "rng": checkpoints.rng_state()}
    path = out / "checkpoints" / name
    temporary = path.with_suffix(".tmp")
    with temporary.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    record = {"file": name, "sha256": checkpoints.file_hash(path), "bytes": path.stat().st_size,
              "step": progress["step"], "completed_epochs": progress["epoch"], "label": label}
    write_json(path.with_suffix(".json"), record)
    previous = json.loads((out / "last.json").read_text()) if (out / "last.json").exists() else None
    write_json(out / "last.json", record)  # Commit pointer only after durable checkpoint and digest.
    write_history(out, progress["history"])
    if progress["best"] is not None:
        best = progress["best"]
        best_record = json.loads((out / "checkpoints" / best["checkpoint_file"]).with_suffix(".json").read_text())
        write_json(out / "best.json", {**best, **best_record})
    # Only this run's replaceable intermediate checkpoint is retired after commit.
    if previous and not previous["label"].startswith("epoch_"):
        old = inside(out / "checkpoints", previous["file"])
        require(old.parent == (out / "checkpoints").resolve() and old.suffix == ".pt", "Unsafe checkpoint retention path")
        old.unlink()
        old.with_suffix(".json").unlink()
    return record


def restore(out, model, optimizer, context):
    record = json.loads((out / "last.json").read_text())
    path = inside(out / "checkpoints", record["file"])
    require(path.stat().st_size == record["bytes"] and checkpoints.file_hash(path) == record["sha256"], "Resume checkpoint changed")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(payload["format"] == 1 and payload["context"] == context, "Resume data/code/configuration/environment changed")
    require(payload["progress"]["step"] == record["step"], "Resume step differs")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    checkpoints.assert_same(model.state_dict(), payload["model"], "resumed model")
    checkpoints.assert_same(optimizer.state_dict(), payload["optimizer"], "resumed optimizer")
    checkpoints.restore_rng(payload["rng"])
    checkpoints.assert_same(checkpoints.rng_state(), payload["rng"], "resumed RNG")
    return payload["progress"]


def fit(model, optimizer, train, val, cfg, device, out, context, progress, should_stop=lambda: False):
    while progress["epoch"] < cfg["epochs"]:
        epoch = progress["epoch"]
        start = progress["next_batch"]
        batches = epoch_batches(len(train), cfg["batch_size"], cfg["seed"], epoch, start)
        for batch_index, (images, targets, examples) in enumerate(loader(train, batches, cfg, epoch), start):
            if should_stop():
                commit(out, model, optimizer, progress, context, "intermediate")
                return "budget_stopped"
            lr = learning_rate(cfg, progress["step"], epoch)
            for group in optimizer.param_groups:
                group["lr"] = lr
            losses, grad_norm = update(model, optimizer, images, targets, device)
            for example in examples:
                aug.record_decision(progress["augmentation_trace"], example.augmentation_decision, example.augmentation_changed)
            progress["step"] += 1
            progress["next_batch"] = batch_index + 1
            progress["epoch_images"] += len(images)
            progress["images_seen"] += len(images)
            for name, value in losses.items():
                progress["loss_sums"][name] = progress["loss_sums"].get(name, 0.) + value * len(images)
            if progress["step"] % cfg["log_every_steps"] == 0:
                print(f"TRAIN epoch={epoch+1}/{cfg['epochs']} batch={batch_index+1} step={progress['step']} "
                      f"loss={sum(losses.values()):.5f} lr={lr:.7g} grad_norm={grad_norm:.4f}", flush=True)
            if progress["step"] % cfg["checkpoint_every_steps"] == 0:
                commit(out, model, optimizer, progress, context, "intermediate")
        require(progress["epoch_images"] == len(train), "Epoch omitted or duplicated training images")
        expected_trace = aug.empty_trace()
        for batch in epoch_batches(len(train), cfg["batch_size"], cfg["seed"], epoch):
            for index in batch:
                aug.record_decision(expected_trace, aug.decision(train.rows[index][1], cfg.get("augmentation_policy"), cfg["seed"], epoch))
        actual_trace = progress["augmentation_trace"]
        require(all(actual_trace[k] == expected_trace[k] for k in ("counts", "decision_sha256")),
                "Augmentation exposure/order differs from the fixed plan")
        # Resume from here reruns validation, without repeating the completed training epoch.
        commit(out, model, optimizer, progress, context, "prevalidation")
        result = validate(model, val, cfg, device, should_stop)
        if result is None:
            return "budget_stopped"
        completed = epoch + 1
        prediction_bytes = json.dumps(result.pop("predictions"), separators=(",", ":"), allow_nan=False).encode()
        atomic_bytes(out / f"epoch_{completed:03d}_val_predictions.json.gz", gzip.compress(prediction_bytes, mtime=0))
        write_json(out / f"epoch_{completed:03d}_validation.json", result)
        row = {"epoch": completed, "step": progress["step"], "images_seen": progress["images_seen"],
               "train_images": len(train), "val_images": len(val),
               "training_loss": {k: v/len(train) for k, v in progress["loss_sums"].items()},
               "validation": result["metrics"], "augmentation": copy.deepcopy(actual_trace)}
        progress["history"].append(row)
        score = result["metrics"]["AP50_95"]
        if progress["best"] is None or score > progress["best"]["validation_AP50_95"]:
            progress["best"] = {"epoch": completed, "validation_AP50_95": score}
        progress.update(epoch=completed, next_batch=0, epoch_images=0, loss_sums={}, augmentation_trace=aug.empty_trace())
        commit(out, model, optimizer, progress, context, f"epoch_{completed:03d}")
        print(f"VALIDATION epoch={completed}/{cfg['epochs']} AP50:95={score:.6f} "
              f"AP50={result['metrics']['AP50']:.6f} best_epoch={progress['best']['epoch']}", flush=True)
        print(f"AUGMENTATION epoch={completed} applied={actual_trace['counts'].get('applied_images', 0)} "
              f"changed={actual_trace['changed_images']} decision_sha256={actual_trace['decision_sha256']}", flush=True)
    require(progress["images_seen"] == cfg["epochs"] * len(train) and
            progress["step"] == cfg["epochs"] * math.ceil(len(train) / cfg["batch_size"]), "Completed training budget differs")
    return "complete"


@contextmanager
def exclusive_run(out):
    # Kernel lock disappears on process exit, including an abrupt scheduler kill.
    import fcntl
    with (out / ".lock").open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def archive_records(out, attempt):
    path = out.parent / f"{out.name}_records_{attempt}.tar.gz"
    files = [p for p in out.rglob("*") if p.is_file() and p.suffix not in (".pt", ".tmp") and p.name != ".lock"]
    sums = {p.relative_to(out).as_posix(): checkpoints.file_hash(p) for p in files if p.name != "checksums.json"}
    write_json(out / "checksums.json", sums)
    files = [p for p in files if p.name != "checksums.json"] + [out / "checksums.json"]
    with tarfile.open(path, "x:gz") as archive:
        for file in sorted(files):
            archive.add(file, arcname=f"{out.name}/{file.relative_to(out).as_posix()}", recursive=False)
    print(f"Records archive: {path}\nArchive SHA-256: {checkpoints.file_hash(path)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "splits", "weights", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-seconds", type=int, default=23*3600)
    args = parser.parse_args()
    require(args.max_seconds > 0, "Invalid job time budget")
    started = time.monotonic()
    package_hashes = verify_package()
    cfg = json.loads((PACKAGE / "protocol.json").read_text())
    require(checkpoints.file_hash(PACKAGE / "identity_protocol.json") == cfg["comparator"]["identity_protocol_sha256"],
            "Identity comparator protocol changed")
    aug.verify_pairing(cfg, json.loads((PACKAGE / "identity_protocol.json").read_text()))
    versions = {name: version(name) for name in cfg["versions"]}
    require(versions == cfg["versions"], "Activate the pinned detection environment")
    require(Path(sys.prefix).resolve() == Path(cfg["expected_environment"]).resolve(), "Wrong Python environment")
    require(os.environ.get("PBS_JOBID") and not socket.gethostname().lower().startswith("login"), "Use a PBS compute job")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "Expected exactly one allocated GPU")
    root, splits, weights, out = [getattr(args, n).resolve() for n in ("root", "splits", "weights", "out")]
    require(all(out != p and not out.is_relative_to(p) and not p.is_relative_to(out)
                for p in (root, splits, weights, PACKAGE)), "Output overlaps an input path")
    require(checkpoints.file_hash(splits / "config.json") == cfg["split_config_sha256"], "Split configuration changed")
    rows = {s: read_manifest(splits / f"{s}.jsonl", cfg["manifests"][s], s) for s in ("train", "val")}
    require({s: len(v) for s, v in rows.items()} == cfg["expected_rows"], "Unexpected dataset sizes")
    verify_partitions(rows["train"], rows["val"])
    source_weight = inside(weights, cfg["weight"]["filename"])
    require(source_weight.stat().st_size == cfg["weight"]["bytes"] and
            checkpoints.file_hash(source_weight) == cfg["weight"]["sha256"], "Pretrained weights changed")
    selected = json.loads((PACKAGE / "smoke_selection.json").read_text()) if args.smoke else None
    if args.smoke:
        cfg.update(epochs=1, warmup_steps=0, lr_decay_epochs=[])
        cfg["augmentation_policy"]["warmup_epochs"] = 0
    datasets = {s: Dataset(root, rows[s], s, cfg["image_size"], selected[s] if selected else None) for s in rows}
    context = {"configuration": cfg, "package_hashes": package_hashes, "versions": versions,
               "profile": "smoke" if args.smoke else "full", "raw_root": str(root), "split_root": str(splits)}
    if args.resume:
        require(out.is_dir() and (out / "last.json").is_file(), "No committed checkpoint to resume")
        require(json.loads((out / "context.json").read_text()) == context, "Run context changed")
        require(json.loads((out / "last.json").read_text())["completed_epochs"] < cfg["epochs"], "Run is already complete")
    else:
        out.mkdir(parents=True, exist_ok=False)
    with exclusive_run(out):
        if not args.resume:
            (out / "checkpoints").mkdir()
            (out / "attempts").mkdir()
            write_json(out / "context.json", context)
            write_json(out / "validation_ground_truth.json", datasets["val"].ground_truth())
            for name in package_hashes:
                destination = inside(out / "code", name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(inside(PACKAGE, name).read_bytes())
        attempt = uuid.uuid4().hex[:12]
        report = {"status": "running", "profile": context["profile"], "started_utc": datetime.now(timezone.utc).isoformat(),
                  "pbs_job": os.environ["PBS_JOBID"], "host": socket.gethostname(), "gpu": torch.cuda.get_device_name(0),
                  "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "command": sys.argv,
                  "test_data_read": False, "resume": args.resume}
        write_json(out / "run.json", report)
        stop = {"requested": False}
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.update(requested=True))
        torch.set_num_threads(cfg["cpu_threads"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        torch.cuda.manual_seed_all(cfg["seed"])
        try:
            if args.smoke:
                from smoke_augmentation import check_operations
                report["augmentation_operation_checks"] = check_operations(datasets["train"], cfg, out)
            model, _, effective = build_model(MODEL, cfg["image_size"], "cuda", cfg["evaluation"], source_weight)
            report["effective_model"] = effective
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                        lr=cfg["learning_rate"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
            progress = restore(out, model, optimizer, context) if args.resume else initial_progress()
            require(progress["epoch"] < cfg["epochs"], "Run is already complete")
            if not args.resume:
                commit(out, model, optimizer, progress, context, "initial")
            report["status"] = fit(model, optimizer, datasets["train"], datasets["val"], cfg, "cuda", out, context,
                                   progress, lambda: stop["requested"] or time.monotonic()-started >= args.max_seconds)
            report.update(completed_epochs=progress["epoch"], optimizer_steps=progress["step"],
                          image_presentations=progress["images_seen"], best=progress["best"])
            report["augmentation_applied_images"] = sum(r["augmentation"]["counts"].get("applied_images", 0) for r in progress["history"])
            if args.smoke and report["status"] == "complete":
                require(report["augmentation_applied_images"] > 0, "Smoke did not exercise augmentation during training")
                images, _, _ = next(iter(loader(datasets["val"], [list(range(4))], cfg, 0, validation=True)))
                expected = infer(model, MODEL, images, cfg["evaluation"], "cuda")
                restored = restore(out, model, optimizer, context)
                actual = infer(model, MODEL, images, cfg["evaluation"], "cuda")
                require(restored == progress, "Smoke checkpoint progress changed")
                for first, second in zip(expected, actual):
                    for key in ("boxes", "scores", "labels"):
                        require(torch.equal(first[key], second[key]), "Smoke checkpoint predictions changed")
                report["checkpoint_prediction_replay"] = "exact"
        except BaseException as error:
            report.update(status="failed", error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            report.update(finished_utc=datetime.now(timezone.utc).isoformat(), elapsed_seconds=time.monotonic()-started)
            report["peak_cuda_memory_allocated_bytes"] = torch.cuda.max_memory_allocated()
            write_json(out / "attempts" / f"{attempt}.json", report)
            write_json(out / "run.json", report)
        archive_records(out, attempt)
        print(f"FASTER R-CNN {context['profile'].upper()}: {report['status'].upper()}", flush=True)


if __name__ == "__main__":
    main()
