"""Scratch CNN baseline for the frozen RDD2022 crop manifests.

Python 3.9+, PyTorch 2.3+, NumPy, Pillow. No torchvision or pretrained weights.
Modes: geometry (metadata), sanity (tiny training subset), train (train/val).
There is deliberately no held-out-test evaluation mode.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import socket
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

LABELS = ["D00", "D10", "D20", "D40"]
MODEL_VERSION = "compact_cnn_v1"
SHARED_POLICY = "content-only hflip p=.5; brightness and contrast each p=.5, factors U(.9,1.1)"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def torch_write(path, value):
    temporary = path.with_name(path.name + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def safe_path(root, relative):
    path = (root / relative).resolve()
    path.relative_to(root.resolve())
    return path


def read_manifest(root):
    content = (root / "crops.csv").read_bytes()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if sha256(content) != config["manifest_sha256"]:
        raise ValueError("Crop manifest does not match its recorded hash")
    if config["labels"] != {str(i): label for i, label in enumerate(LABELS)}:
        raise ValueError("Unexpected class mapping")
    if config["crop_size"] != 224:
        raise ValueError("This CNN expects the versioned 224-pixel crops")
    rows = list(csv.DictReader(io.StringIO(content.decode("utf-8"))))
    seen_ids, seen_paths = set(), set()
    memberships = {key: {} for key in ("image", "group_id", "file_sha256")}
    for row in rows:
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Unexpected split")
        label_id = int(row["label_id"])
        if not 0 <= label_id < len(LABELS) or LABELS[label_id] != row["label"]:
            raise ValueError("Class mapping mismatch")
        if row["sample_id"] in seen_ids or row["crop_path"] in seen_paths:
            raise ValueError("Duplicate crop record")
        seen_ids.add(row["sample_id"])
        seen_paths.add(row["crop_path"])
        expected = Path(row["split"]) / row["label"] / (row["sample_id"] + ".png")
        if Path(row["crop_path"]) != expected:
            raise ValueError("Unexpected crop path")
        safe_path(root, row["crop_path"])
        width, height = int(row["right"])-int(row["left"]), int(row["bottom"])-int(row["top"])
        if width <= 0 or height <= 0:
            raise ValueError("Nonpositive crop dimensions")
        for key, membership in memberships.items():
            previous = membership.setdefault(row[key], row["split"])
            if previous != row["split"]:
                raise ValueError("Cross-split " + key + ": " + row[key])
    splits = {s: [r for r in rows if r["split"] == s] for s in ("train", "val")}
    for split, records in splits.items():
        if set(r["label"] for r in records) != set(LABELS):
            raise ValueError("Missing class in " + split)
    # Test metadata were checked for partition integrity; test pixels are never opened.
    return splits, config


def read_checksums(root, splits):
    run = json.loads((root / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete":
        raise ValueError("Crop generation is incomplete")
    if run["manifest_sha256"] != sha256((root / "crops.csv").read_bytes()):
        raise ValueError("Crop generation used a different manifest")
    with (root / "crop_checksums.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    checksums = {r["crop_path"]: r["sha256"] for r in records}
    if len(checksums) != len(records):
        raise ValueError("Duplicate crop checksum records")
    for rows in splits.values():
        for row in rows:
            if row["crop_path"] not in checksums:
                raise ValueError("Missing crop checksum")
    return checksums


def augmentation_rng(seed, epoch, sample_id):
    key = "%d:%d:%s" % (seed, epoch, sample_id)
    return random.Random(int(sha256(key.encode("utf-8"))[:16], 16))


def augment(image, row, rng):
    """Shared mild policy, applied only inside the letterbox content rectangle."""
    width, height = int(row["right"])-int(row["left"]), int(row["bottom"])-int(row["top"])
    scale = min(224 / width, 224 / height)
    w, h = max(1, min(224, round(width*scale))), max(1, min(224, round(height*scale)))
    left, top = (224-w)//2, (224-h)//2
    content = image.crop((left, top, left+w, top+h))
    if rng.random() < 0.5:
        content = ImageOps.mirror(content)
    if rng.random() < 0.5:
        content = ImageEnhance.Brightness(content).enhance(rng.uniform(0.9, 1.1))
    if rng.random() < 0.5:
        content = ImageEnhance.Contrast(content).enhance(rng.uniform(0.9, 1.1))
    result = image.copy()
    result.paste(content, (left, top))
    return result


class CropDataset(Dataset):
    def __init__(self, root, rows, checksums, policy="none", seed=42):
        if policy not in ("none", "shared"):
            raise ValueError("Unknown augmentation policy")
        if any(r["split"] == "test" for r in rows):
            raise ValueError("Test images are disabled during development")
        if policy != "none" and any(r["split"] != "train" for r in rows):
            raise ValueError("Augmentation is restricted to training images")
        self.root, self.rows, self.checksums = root, rows, checksums
        self.policy, self.seed, self.epoch = policy, seed, 0

    def __len__(self):
        return len(self.rows)

    def image(self, index, augmented=True):
        row = self.rows[index]
        data = safe_path(self.root, row["crop_path"]).read_bytes()
        if sha256(data) != self.checksums[row["crop_path"]]:
            raise ValueError("Crop changed since generation: " + row["crop_path"])
        with Image.open(io.BytesIO(data)) as im:
            if im.size != (224, 224) or im.mode != "RGB":
                raise ValueError("Expected 224 x 224 RGB crop")
            picture = im.copy()
        if augmented and self.policy == "shared":
            picture = augment(picture, row, augmentation_rng(self.seed, self.epoch, row["sample_id"]))
        return picture

    def __getitem__(self, index):
        picture = self.image(index)
        array = np.array(picture, dtype=np.float32, copy=True)
        # Fixed scaling, without ImageNet statistics or fitted validation statistics.
        tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().div_(127.5).sub_(1.0)
        return tensor, int(self.rows[index]["label_id"]), index


class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers, incoming = [], 3
        for outgoing in (16, 32, 64, 128):
            for _ in range(2):
                layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1, bias=False),
                               nn.BatchNorm2d(outgoing), nn.ReLU(inplace=True)])
                incoming = outgoing
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers, nn.AvgPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128*7*7, 64),
                                        nn.ReLU(inplace=True), nn.Dropout(0.25), nn.Linear(64, 4))

    def forward(self, x):
        return self.classifier(self.features(x))


def confusion_matrix(truth, prediction):
    truth, prediction = np.asarray(truth, dtype=np.int64), np.asarray(prediction, dtype=np.int64)
    return np.bincount(4*truth+prediction, minlength=16).reshape(4, 4)


def metrics(cm):
    cm = np.asarray(cm, dtype=np.int64)
    tp, support, predicted = cm.diagonal(), cm.sum(axis=1), cm.sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(4), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(4), where=support != 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(4), where=support+predicted != 0)
    return {"accuracy": float(tp.sum()/max(1, cm.sum())), "macro_f1": float(f1.mean()),
            "confusion_matrix": cm.tolist(), "confusion_order": LABELS,
            "confusion_axes": "rows=true, columns=predicted",
            "per_class": {label: {"support": int(support[i]), "precision": float(precision[i]),
                                  "recall": float(recall[i]), "f1": float(f1[i])}
                          for i, label in enumerate(LABELS)}}


def geometry(splits, out):
    def ratios(rows):
        return np.array([math.log((int(r["right"])-int(r["left"]))/
                                 (int(r["bottom"])-int(r["top"]))) for r in rows])
    train, val = splits["train"], splits["val"]
    # Twenty equal-frequency bins defined using TRAINING geometry only.
    boundaries = np.unique(np.quantile(ratios(train), np.arange(1, 20)/20))
    counts = np.ones((len(boundaries)+1, 4), dtype=np.float64)
    for index, row in zip(np.searchsorted(boundaries, ratios(train)), train):
        counts[index, int(row["label_id"])] += 1
    predictions = counts[np.searchsorted(boundaries, ratios(val))].argmax(axis=1)
    truth = [int(r["label_id"]) for r in val]
    majority = Counter(int(r["label_id"]) for r in train).most_common(1)[0][0]
    result = {"method": "Training-fitted 20-quantile-bin classifier of log(width/height); add-one counts",
              "evaluation_split": "val", "test_evaluated": False,
              "boundaries_log_ratio": boundaries.tolist(), "smoothed_class_counts": counts.tolist(),
              "geometry": metrics(confusion_matrix(truth, predictions)),
              "training_majority_class": LABELS[majority],
              "majority": metrics(confusion_matrix(truth, [majority]*len(truth)))}
    json_write(out / "geometry_metrics.json", result)
    print("GEOMETRY COMPLETE | val accuracy %.4f | macro-F1 %.4f" %
          (result["geometry"]["accuracy"], result["geometry"]["macro_f1"]), flush=True)
    return result


def seed_all(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def worker_init(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.set_num_threads(1)


def loader(dataset, args, shuffle):
    generator = torch.Generator().manual_seed(args.seed + dataset.epoch)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, pin_memory=args.device == "cuda",
                      worker_init_fn=worker_init, generator=generator, drop_last=False)


def epoch_pass(model, batches, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total, cm, indices, predictions = 0.0, 0, np.zeros((4, 4), dtype=np.int64), [], []
    with torch.set_grad_enabled(optimizer is not None):
        for step, (x, y, index) in enumerate(batches, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, y)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            pred = logits.detach().argmax(dim=1).cpu().numpy()
            cm += confusion_matrix(y.cpu().numpy(), pred)
            total_loss += loss.item()*len(y)
            total += len(y)
            indices.extend(index.tolist())
            predictions.extend(pred.tolist())
            if optimizer is not None and step % 100 == 0:
                print("  batch %d/%d | mean loss %.4f" % (step, len(batches), total_loss/total), flush=True)
    result = metrics(cm)
    result["loss"] = total_loss/total
    return result, indices, predictions


def tiny_rows(rows, seed, per_class=8):
    chosen = []
    for label in LABELS:
        candidates = sorted((r for r in rows if r["label"] == label), key=lambda r:
                            sha256((str(seed)+":"+r["sample_id"]).encode()))
        if len(candidates) < per_class:
            raise ValueError("Insufficient training examples for sanity check")
        chosen.extend(candidates[:per_class])
    return chosen


def augmentation_preview(dataset, out):
    canvas = Image.new("RGB", (4*224, 4*248+24), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 4), "TRAIN ONLY: original / policy=" + dataset.policy + " (two samples per class)", fill="black")
    for i, label in enumerate(LABELS):
        indices = [j for j, r in enumerate(dataset.rows) if r["label"] == label][:2]
        for k, index in enumerate(indices):
            y = 24+i*248
            canvas.paste(dataset.image(index, augmented=False), (k*448, y+20))
            canvas.paste(dataset.image(index), (k*448+224, y+20))
            draw.text((k*448+4, y), label+" original / policy="+dataset.policy, fill="black")
    canvas.save(out / "augmentation_preview.jpg", quality=94)


def sanity(args, splits, checksums, out):
    rows = tiny_rows(splits["train"], args.seed)
    dataset = CropDataset(args.data, rows, checksums)
    tensors = [dataset[i] for i in range(len(dataset))]
    x = torch.stack([r[0] for r in tensors]).to(args.device)
    y = torch.tensor([r[1] for r in tensors], device=args.device)
    model = CompactCNN().to(args.device)
    # Disable dropout for this debugging exercise, not for full training.
    for layer in model.modules():
        if isinstance(layer, nn.Dropout):
            layer.p = 0.0
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    model.eval()
    with torch.no_grad():
        initial_loss = nn.functional.cross_entropy(model(x), y).item()
    for step in range(1, args.steps+1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x), y)
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite sanity-check loss")
        loss.backward()
        optimizer.step()
        if step % 25 == 0 or step == args.steps:
            print("Sanity step %d/%d | training loss %.4f" % (step, args.steps, loss.item()), flush=True)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        result = metrics(confusion_matrix(y.cpu().numpy(), logits.argmax(1).cpu().numpy()))
        result["final_loss"] = nn.functional.cross_entropy(logits, y).item()
    result.update(initial_loss=initial_loss, steps=args.steps, training_samples=len(rows),
                  test_evaluated=False, validation_evaluated=False,
                  purpose="Memorization check on training data; not a generalization result",
                  sample_ids=[r["sample_id"] for r in rows])
    result["passed"] = result["accuracy"] >= 0.95 and result["final_loss"] < initial_loss*0.5
    json_write(out / "sanity_metrics.json", result)
    dataset.policy = args.augmentation
    augmentation_preview(dataset, out)
    print("SANITY %s | accuracy %.4f | loss %.4f -> %.4f" %
          ("PASSED" if result["passed"] else "NEEDS REVIEW", result["accuracy"],
           initial_loss, result["final_loss"]), flush=True)
    return result["passed"]


def cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def write_history(out, history):
    json_write(out / "history.json", history)
    temporary = out / "history.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "lr", "seconds", "train_loss", "train_accuracy", "train_macro_f1",
                         "val_loss", "val_accuracy", "val_macro_f1"])
        for record in history:
            writer.writerow([record["epoch"], record["lr"], record["seconds"]] +
                            [record[split][key] for split in ("train", "val")
                             for key in ("loss", "accuracy", "macro_f1")])
    temporary.replace(out / "history.csv")


def save_best(out, best, config, val_rows):
    torch_write(out / "best.pt", {"model": best["model"], "epoch": best["epoch"], "config": config})
    json_write(out / "best_validation.json", {"epoch": best["epoch"], "metrics": best["metrics"],
                                              "test_evaluated": False})
    with (out / "validation_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["sample_id", "source", "image", "group_id", "true_class", "predicted_class"])
        for i, pred in zip(best["indices"], best["predictions"]):
            row = val_rows[i]
            writer.writerow([row["sample_id"], row["source"], row["image"], row["group_id"],
                             row["label"], LABELS[pred]])


def train(args, splits, checksums, out, config):
    train_data = CropDataset(args.data, splits["train"], checksums, args.augmentation, args.seed)
    val_data = CropDataset(args.data, splits["val"], checksums)
    model = CompactCNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    history, best, first_epoch = [], None, 1
    if args.resume:
        # Only resume this script's locally generated checkpoint in the run directory.
        saved = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
        if saved["config"] != config:
            raise ValueError("Resume config differs; use the same arguments and script")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        torch.set_rng_state(saved["torch_rng"])
        if args.device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        history, best, first_epoch = saved["history"], saved["best"], saved["epoch"]+1
        write_history(out, history)
        save_best(out, best, config, splits["val"])
    else:
        geometry(splits, out)
        preview = CropDataset(args.data, tiny_rows(splits["train"], args.seed, per_class=2),
                              checksums, args.augmentation, args.seed)
        augmentation_preview(preview, out)
    started = time.monotonic()
    print("Parameters:", sum(p.numel() for p in model.parameters()), flush=True)
    completed_epoch = first_epoch-1
    for epoch in range(first_epoch, args.epochs+1):
        begin, lr = time.monotonic(), optimizer.param_groups[0]["lr"]
        train_data.epoch = epoch
        train_result, _, _ = epoch_pass(model, loader(train_data, args, True), args.device, optimizer)
        val_result, indices, predictions = epoch_pass(model, loader(val_data, args, False), args.device)
        scheduler.step()
        history.append({"epoch": epoch, "lr": lr, "seconds": time.monotonic()-begin,
                        "train": train_result, "val": val_result})
        improved = best is None or val_result["macro_f1"] > best["metrics"]["macro_f1"]
        if improved:
            best = {"epoch": epoch, "metrics": val_result, "model": cpu_state(model),
                    "indices": indices, "predictions": predictions}
        saved = {"config": config, "epoch": epoch, "model": cpu_state(model),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "torch_rng": torch.get_rng_state(),
                 "cuda_rng": torch.cuda.get_rng_state_all() if args.device == "cuda" else [],
                 "history": history, "best": best}
        torch_write(out / "last.pt", saved)
        write_history(out, history)
        if improved:
            save_best(out, best, config, splits["val"])
        completed_epoch = epoch
        print("Epoch %d/%d | %.1fs | train loss %.4f | val loss %.4f | val acc %.4f | val macro-F1 %.4f" %
              (epoch, args.epochs, history[-1]["seconds"], train_result["loss"], val_result["loss"],
               val_result["accuracy"], val_result["macro_f1"]), flush=True)
        if args.stop_after_epoch and epoch >= args.stop_after_epoch:
            break
        if args.max_minutes and (time.monotonic()-started)/60 >= args.max_minutes:
            break
    complete = completed_epoch == args.epochs
    json_write(out / "run_status.json", {"status": "complete" if complete else "paused_after_epoch",
                                         "completed_epochs": completed_epoch, "planned_epochs": args.epochs,
                                         "best_epoch": best["epoch"], "test_evaluated": False})
    print("TRAINING %s | best epoch %d | val macro-F1 %.4f" %
          ("COMPLETE" if complete else "PAUSED (resume with --resume)", best["epoch"], best["metrics"]["macro_f1"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("geometry", "sanity", "train"))
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--augmentation", choices=("shared", "none"), default="shared")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-minutes", type=float, default=75,
                        help="Pause at an epoch boundary after this many minutes; 0 disables")
    parser.add_argument("--stop-after-epoch", type=int, default=0,
                        help="Optional planned interruption, mainly for resume verification")
    args = parser.parse_args()
    if min(args.epochs, args.steps, args.batch_size, args.threads) <= 0 or args.workers < 0:
        parser.error("Counts must be positive; workers may be zero")
    if args.lr <= 0 or args.weight_decay < 0 or args.max_minutes < 0 or args.stop_after_epoch < 0:
        parser.error("Invalid optimizer or stopping option")
    if args.resume and args.mode != "train":
        parser.error("Only training supports resume")
    args.data, args.out = args.data.resolve(), args.out.resolve()
    if args.out == args.data or args.data in args.out.parents:
        parser.error("Output must be outside the prepared data directory")
    splits, data_config = read_manifest(args.data)
    if args.mode != "geometry" and args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; request a GPU job and activate PyTorch")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible allocated GPU; check the scheduler assignment")
    seed_all(args.seed, args.threads)
    config = {"script_sha256": sha256(Path(__file__).read_bytes()),
              "manifest_sha256": data_config["manifest_sha256"], "mode": args.mode,
              "model": MODEL_VERSION, "pretrained": False, "seed": args.seed,
              "epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr,
              "weight_decay": args.weight_decay if args.mode == "train" else 0.0,
              "augmentation": args.augmentation if args.mode == "train" else "none",
              "preview_policy": args.augmentation if args.mode != "geometry" else None,
              "device": args.device, "workers": args.workers, "threads": args.threads,
              "steps": args.steps, "data": str(args.data),
              "torch": str(torch.__version__), "pillow": PIL.__version__, "numpy": np.__version__,
              "loss": "unweighted cross entropy", "sampling": "shuffle, no oversampling",
              "selection": "maximum validation macro-F1; first epoch wins ties",
              "precision": "float32, TF32 disabled", "test_evaluated": False,
              "policy": SHARED_POLICY if args.mode == "train" and args.augmentation == "shared" else "none",
              "sanity_protocol": "32 training samples; no augmentation, dropout, weight decay or scheduler" if args.mode == "sanity" else None,
              "normalization": "RGB / 127.5 - 1", "scheduler": "cosine to 1e-5; fixed planned epoch budget"}
    if args.resume:
        if json.loads((args.out / "config.json").read_text(encoding="utf-8")) != config:
            raise ValueError("Resume configuration mismatch")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        json_write(args.out / "config.json", config)
    json_write(args.out / ("environment_resume.json" if args.resume else "environment.json"),
               {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "executable": sys.executable, "host": socket.gethostname(), "platform": platform.platform(),
                "torch": str(torch.__version__), "pillow": PIL.__version__, "numpy": np.__version__,
                "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "gpu": torch.cuda.get_device_name(0) if args.mode != "geometry" and args.device == "cuda" else None,
                "PBS_JOBID": os.environ.get("PBS_JOBID"),
                "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "argv": sys.argv})
    print("Data:", {s: len(rows) for s, rows in splits.items()}, "| mode:", args.mode, flush=True)
    if args.mode == "geometry":
        geometry(splits, args.out)
    else:
        checksums = read_checksums(args.data, splits)
        if args.mode == "sanity":
            if not sanity(args, splits, checksums, args.out):
                return 2
        else:
            train(args, splits, checksums, args.out, config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
