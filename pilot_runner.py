"""Paired operation/magnitude diagnostics, not a learned augmentation controller.

Python 3.9+, PyTorch 2.3+, NumPy and Pillow; imports the unchanged baseline.
No test pixels, pretrained weights, torchvision, or additional dependencies.
See PILOT_RUNBOOK.md for estimands, limitations and CCAST commands.
"""
import argparse
import copy
import csv
import io
import json
import math
import platform
import random
import socket
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import train_baseline as base  # Sets deterministic CUDA environment before torch import.
import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import torch
from torch import nn

VERSION = "augmentation_utility_pilot_v1"
LABELS = base.LABELS
OPERATIONS = ("brightness", "contrast", "blur", "translate_x", "translate_y")
DEFAULT_BANK = [{"id": "identity", "operation": "identity", "magnitude": 0.0}] + [
    {"id": op + "_" + level, "operation": op, "magnitude": magnitude}
    for op, levels in (("brightness", (0.10, 0.25)), ("contrast", (0.10, 0.25)),
                       ("blur", (0.5, 1.0)), ("translate_x", (0.03, 0.08)),
                       ("translate_y", (0.03, 0.08)))
    for level, magnitude in zip(("low", "high"), levels)]


def digest(value):
    return base.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_hashes():
    return {p.name: base.sha256(p.read_bytes()) for p in (Path(__file__), Path(base.__file__))}


def keyed_rng(*parts):
    return random.Random(int(base.sha256(":".join(map(str, parts)).encode())[:16], 16))


def validate_bank(bank):
    if not bank or len({a["id"] for a in bank}) != len(bank):
        raise ValueError("Empty bank or duplicate candidate IDs")
    for a in bank:
        if not a["id"].replace("_", "").isalnum():
            raise ValueError("Candidate IDs must be alphanumeric with optional underscores")
        op, m = a["operation"], a["magnitude"]
        if op not in ("identity",) + OPERATIONS or not isinstance(m, (float, int)) or not math.isfinite(m):
            raise ValueError("Unknown operation or invalid magnitude")
        bound = 0 if op == "identity" else (3 if op == "blur" else 0.5)
        if m < 0 or m > bound:
            raise ValueError("Magnitude outside implemented diagnostic bounds")
    if sum(a["operation"] == "identity" for a in bank) != 1:
        raise ValueError("Exactly one identity control is required")


def content_box(row):
    w, h = int(row["right"]) - int(row["left"]), int(row["bottom"]) - int(row["top"])
    scale = min(224 / w, 224 / h)
    w, h = max(1, min(224, round(w * scale))), max(1, min(224, round(h * scale)))
    x, y = (224 - w) // 2, (224 - h) // 2
    return x, y, x + w, y + h


def transform(image, row, action, sign):
    """One extra operation on content only. Magnitude zero is exact identity.

    Brightness/contrast factor = 1 +/- magnitude. Blur radius is in resized
    pixels. Translation is a rounded fraction of content width/height, with
    gray fill and no wraparound. Tiny content may yield a zero-pixel shift.
    """
    op, m = action["operation"], action["magnitude"]
    if op == "identity" or m == 0:
        return image.copy()
    box = content_box(row)
    content = image.crop(box)
    if op in ("brightness", "contrast"):
        enhancer = ImageEnhance.Brightness if op == "brightness" else ImageEnhance.Contrast
        content = enhancer(content).enhance(1 + sign * m)
    elif op == "blur":
        content = content.filter(ImageFilter.GaussianBlur(radius=m))
    else:
        shifted = Image.new("RGB", content.size, (127, 127, 127))
        dx = sign * round(m * content.width) if op == "translate_x" else 0
        dy = sign * round(m * content.height) if op == "translate_y" else 0
        shifted.paste(content, (dx, dy))
        content = shifted
    result = image.copy()
    result.paste(content, box[:2])
    return result


def grouped_roles(rows, seed):
    """Union ALL known image/group/file relationships before splitting validation."""
    parent = list(range(len(rows)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        for field in ("image", "group_id", "file_sha256"):
            key = field, row[field]
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    cells = sorted({(r["source"], r["label"]) for r in rows})
    total = Counter((r["source"], r["label"]) for r in rows)
    allocated = [Counter(), Counter()]
    result = {}
    ordered = sorted(groups.values(), key=lambda group: (
        -len(group), base.sha256((str(seed) + min(r["sample_id"] for r in group)).encode())))
    for group in ordered:
        counts = Counter((r["source"], r["label"]) for r in group)
        scores = []
        for side in (0, 1):
            scores.append(sum(((allocated[s][cell] + (counts[cell] if side == s else 0)
                                - total[cell] / 2) ** 2) / max(total[cell], 1)
                              for s in (0, 1) for cell in cells))
        side = 0 if scores[0] < scores[1] else 1
        allocated[side].update(counts)
        for row in group:
            result[row["sample_id"]] = ("reference", "audit")[side]
    for role in ("reference", "audit"):
        if {r["label"] for r in rows if result[r["sample_id"]] == role} != set(LABELS):
            raise ValueError("Too few independent groups to give both development roles all classes")
    return result


def select_panel(rows, per_class, seed):
    """Prefer distinct groups; round-robin sources. Metadata-only selection."""
    selected = []
    for label in LABELS:
        pools = defaultdict(list)
        for row in rows:
            if row["label"] == label:
                pools[row["source"]].append(row)
        for source, pool in pools.items():
            pool.sort(key=lambda r: base.sha256((str(seed) + r["sample_id"]).encode()))
        seen, chosen, deferred = set(), [], []
        while any(pools.values()):
            for source in sorted(pools):
                if pools[source]:
                    row = pools[source].pop()
                    if row["group_id"] in seen:
                        deferred.append(row)
                    else:
                        chosen.append(row)
                        seen.add(row["group_id"])
        selected.extend((chosen + deferred)[:per_class])
    return selected


def prepare(args):
    splits, data_config = base.read_manifest(args.manifest.parent)
    bank = read_json(args.bank) if args.bank else DEFAULT_BANK
    validate_bank(bank)
    roles = grouped_roles(splits["val"], args.seed)
    panels = {role: select_panel([r for r in splits["val"] if roles[r["sample_id"]] == role],
                                 args.eval_per_class, args.seed)
              for role in ("reference", "audit")}
    payload = {"version": VERSION, "manifest_sha256": data_config["manifest_sha256"],
               "seed": args.seed, "bank": bank, "roles": roles,
               "panels": {k: [r["sample_id"] for r in v] for k, v in panels.items()},
               "requested_eval_per_class": args.eval_per_class,
               "panel_counts": {k: dict(Counter(r["label"] for r in v)) for k, v in panels.items()},
               "development_only": True, "test_evaluated": False,
               "prior_access": "All original validation data previously used for baseline selection",
               "estimand": "Extra operation on target class after shared conventional augmentation;"
                           " balanced diagnostic training batches; fixed checkpoint learning rate",
               "strength_units": "brightness/contrast: factor 1 +/- m; blur: pixels; translation:"
                                 " fraction of resized content dimension; all ranges provisional"}
    payload["design_sha256"] = digest(payload)
    args.out.mkdir(parents=True, exist_ok=False)
    base.json_write(args.out / "design.json", payload)
    with (args.out / "development_membership.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sample_id", "source", "label", "image", "group_id", "role", "in_panel"])
        selected = set(sum(payload["panels"].values(), []))
        for row in splits["val"]:
            writer.writerow([row[k] for k in ("sample_id", "source", "label", "image", "group_id")]
                            + [roles[row["sample_id"]], row["sample_id"] in selected])
    print("DESIGN COMPLETE", payload["panel_counts"], flush=True)


def load_design(path, splits, manifest_hash):
    design = read_json(path / "design.json")
    claimed = design.pop("design_sha256")
    if digest(design) != claimed or design["manifest_sha256"] != manifest_hash:
        raise ValueError("Design integrity or source manifest mismatch")
    design["design_sha256"] = claimed
    validate_bank(design["bank"])
    val = {r["sample_id"]: r for r in splits["val"]}
    if set(design["roles"]) != set(val) or set(design["roles"].values()) != {"reference", "audit"}:
        raise ValueError("Development role coverage mismatch")
    memberships = {k: {} for k in ("image", "group_id", "file_sha256")}
    for sid, row in val.items():
        for field, membership in memberships.items():
            if membership.setdefault(row[field], design["roles"][sid]) != design["roles"][sid]:
                raise ValueError("Cross-role " + field)
    panels = {}
    for role in ("reference", "audit"):
        ids = design["panels"][role]
        if len(set(ids)) != len(ids) or any(i not in val or design["roles"][i] != role for i in ids):
            raise ValueError("Invalid development panel")
        panels[role] = [val[i] for i in ids]
        if {r["label"] for r in panels[role]} != set(LABELS):
            raise ValueError("Missing class in panel")
    return design, panels


def preview(args):
    manifest_root = args.data if args.data else args.manifest.parent
    splits, config = base.read_manifest(manifest_root)
    design, _ = load_design(args.design, splits, config["manifest_sha256"])
    rows = splits["train"]
    proxies = {}
    provenance = {"test_evaluated": False, "design_sha256": design["design_sha256"],
                  "bank": design["bank"], "images": []}
    if args.qa_dir:
        # These are tiles of our recorded contact sheets, NOT recovered original PNGs.
        qa_rows = list(csv.DictReader((args.qa_dir / "selected_samples.csv").open(encoding="utf-8")))
        canonical = {r["sample_id"]: r for r in rows}
        pools = defaultdict(list)
        for r in qa_rows:
            if r["sample_id"] not in canonical or r != canonical[r["sample_id"]]:
                raise ValueError("QA selection does not match frozen training manifest")
            pools[r["source"], r["label"]].append(r)
        for (source, label), members in sorted(pools.items()):
            sheet_path = args.qa_dir / (source + ".jpg")
            with Image.open(sheet_path) as sheet:
                if sheet.size != (960, 1080):
                    raise ValueError("Unexpected contact-sheet layout")
                for column, r in enumerate(members[:args.preview_per_cell]):
                    x, y = column * 240 + 8, 47 + LABELS.index(label) * 260
                    proxies[r["sample_id"]] = sheet.crop((x, y, x + 224, y + 224)).convert("RGB")
            provenance.setdefault("source_sheet_hashes", {})[source] = base.sha256(sheet_path.read_bytes())
        selected = [r for r in qa_rows if r["sample_id"] in proxies]
        provenance["source_kind"] = "LOSSY JPEG CONTACT-SHEET PROXIES; visual diagnostics only"
        get_image = lambda r: proxies[r["sample_id"]].copy()
    else:
        import materialize_crops
        cells = materialize_crops.select_previews(rows)
        selected = [r for source in sorted(cells) for label in LABELS
                    for r in cells[source].get(label, [])[:args.preview_per_cell]]
        checksums = base.read_checksums(args.data, splits)
        dataset = base.CropDataset(args.data, selected, checksums)
        indices = {r["sample_id"]: i for i, r in enumerate(selected)}
        get_image = lambda r: dataset.image(indices[r["sample_id"]], augmented=False)
        provenance["source_kind"] = "ORIGINAL CHECKSUM-VERIFIED TRAINING PNGS"
    args.out.mkdir(parents=True, exist_ok=False)
    for source in sorted({r["source"] for r in selected}):
        members = [r for r in selected if r["source"] == source]
        for op in OPERATIONS:
            actions = [a for a in design["bank"] if a["operation"] == op]
            if not actions:
                continue
            variants = [(a, s) for a in actions for s in ((1,) if op == "blur" else (-1, 1))]
            cols = 2 + len(variants)
            canvas = Image.new("RGB", (cols * 232, 66 + len(members) * 256), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 5), source + " | " + op + " | " + provenance["source_kind"], fill="black")
            captions = ["Original", "Shared reference"] + [a["id"] + (" -" if s < 0 else " +") for a, s in variants]
            for j, caption in enumerate(captions):
                draw.text((8 + j * 232, 34), caption, fill="black")
            for i, row in enumerate(members):
                original = get_image(row)
                shared = base.augment(original, row, keyed_rng(design["seed"], "preview", row["sample_id"]))
                pictures = [original, shared] + [transform(shared, row, a, s) for a, s in variants]
                for j, picture in enumerate(pictures):
                    canvas.paste(picture, (j * 232 + 4, 66 + i * 256))
                draw.text((8, 66 + i * 256 + 227), row["label"] + " | " + row["sample_id"], fill="black")
            filename = source + "_" + op + ".png"
            canvas.save(args.out / filename)
            provenance["images"].append({"file": filename, "sample_ids": [r["sample_id"] for r in members]})
    base.json_write(args.out / "preview_manifest.json", provenance)
    print("PREVIEWS COMPLETE |", provenance["source_kind"], "| sheets:", len(provenance["images"]), flush=True)


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])


def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def snapshot(model, optimizer, scheduler, epoch, config):
    return {"kind": VERSION, "model": cpu_copy(model.state_dict()),
            "optimizer": cpu_copy(optimizer.state_dict()), "scheduler": scheduler.state_dict(),
            "rng": cpu_copy(capture_rng()), "epoch": epoch, "config": config}


def restore_snapshot(saved, device):
    model = base.CompactCNN().to(device)
    model.load_state_dict(saved["model"])
    optimizer = torch.optim.AdamW(model.parameters())
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=saved["config"]["epochs"], eta_min=1e-5)
    scheduler.load_state_dict(copy.deepcopy(saved["scheduler"]))
    # Constructor can reset initial LR; restore the complete optimizer state last.
    optimizer.load_state_dict(copy.deepcopy(saved["optimizer"]))
    restore_rng(saved["rng"])
    return model, optimizer, scheduler


def checked_snapshot(path, design, data_config, device):
    # Only load trusted checkpoints produced by this runner, never downloaded weights.
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved.get("config", {})
    if saved.get("kind") != VERSION or config.get("code_hashes") != code_hashes():
        raise ValueError("Checkpoint is not from this version of the pilot runner")
    if config.get("manifest_sha256") != data_config["manifest_sha256"] or config.get("design_sha256") != design["design_sha256"]:
        raise ValueError("Checkpoint data/design mismatch")
    if device == "cpu" and saved["rng"]["cuda"]:
        # CPU diagnostics are allowed, but are explicitly a different execution environment.
        saved["rng"]["cuda"] = []
    return saved


def tensor(image):
    return torch.from_numpy(np.array(image, dtype=np.float32, copy=True)).permute(2, 0, 1).contiguous().div_(127.5).sub_(1)


def evaluate(model, dataset, device, batch_size, predictions=False):
    model.eval()
    cm, sums, counts, records = np.zeros((4, 4), dtype=np.int64), np.zeros(4), np.zeros(4, dtype=int), []
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            examples = [dataset[i] for i in range(start, min(len(dataset), start + batch_size))]
            x = torch.stack([e[0] for e in examples]).to(device)
            y = torch.tensor([e[1] for e in examples], device=device)
            logits = model(x)
            losses = nn.functional.cross_entropy(logits, y, reduction="none")
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite evaluation loss")
            truth, pred, loss = y.cpu().numpy(), logits.argmax(1).cpu().numpy(), losses.cpu().numpy()
            cm += base.confusion_matrix(truth, pred)
            sums += np.bincount(truth, weights=loss, minlength=4)
            counts += np.bincount(truth, minlength=4)
            if predictions:
                records.extend({"sample_id": dataset.rows[e[2]]["sample_id"], "label": LABELS[e[1]],
                                "prediction": LABELS[int(p)], "loss": float(l)}
                               for e, p, l in zip(examples, pred, loss))
    if (counts == 0).any():
        raise ValueError("Evaluation requires every class")
    result = base.metrics(cm)
    result.update(loss=float(sums.sum() / counts.sum()), balanced_loss=float((sums / counts).mean()),
                  class_loss={label: float(sums[i] / counts[i]) for i, label in enumerate(LABELS)})
    if predictions:
        result["predictions"] = records
    return result


def reference_gradient(model, dataset, device, batch_size):
    """Gradient of mean per-class reference CE, in eval mode (fixed BN buffers)."""
    model.eval()
    model.zero_grad(set_to_none=True)
    counts = Counter(int(r["label_id"]) for r in dataset.rows)
    for start in range(0, len(dataset), batch_size):
        examples = [dataset[i] for i in range(start, min(len(dataset), start + batch_size))]
        x = torch.stack([e[0] for e in examples]).to(device)
        y = torch.tensor([e[1] for e in examples], device=device)
        weights = torch.tensor([1 / (4 * counts[e[1]]) for e in examples], device=device)
        (nn.functional.cross_entropy(model(x), y, reduction="none") * weights).sum().backward()
    return torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()])


def balanced_plan(rows, steps, batch_size, seed):
    if batch_size % 4 or batch_size < 4:
        raise ValueError("Diagnostic batch size must be a positive multiple of four")
    pools = [[i for i, r in enumerate(rows) if r["label"] == label] for label in LABELS]
    if any(not p for p in pools):
        raise ValueError("Missing training class")
    rng, queues, plan = random.Random(seed), [[], [], [], []], []
    for _ in range(steps):
        batch = []
        for c in range(4):
            for _ in range(batch_size // 4):
                if not queues[c]:
                    queues[c] = pools[c].copy()
                    rng.shuffle(queues[c])
                batch.append(queues[c].pop())
        rng.shuffle(batch)
        plan.append(batch)
    return plan


def training_batch(dataset, indices, target, action, seed, step):
    images, labels, changed = [], [], 0
    for index in indices:
        row = dataset.rows[index]
        original = dataset.image(index, augmented=False)
        shared = base.augment(original, row, keyed_rng(seed, step, row["sample_id"], "shared"))
        if row["label"] == target:
            sign = keyed_rng(seed, step, row["sample_id"], "sign").choice((-1, 1))
            picture = transform(shared, row, action, sign)
            changed += not np.array_equal(np.array(shared), np.array(picture))
        else:
            picture = shared
        images.append(tensor(picture))
        labels.append(int(row["label_id"]))
    return torch.stack(images), torch.tensor(labels), changed


def difficulty_signals(saved, dataset, plan, target, action, seed, device):
    model, _, _ = restore_snapshot(saved, device)
    model.eval()
    indices = [i for i in plan[0] if dataset.rows[i]["label"] == target]
    identity = {"operation": "identity", "magnitude": 0}
    x0, y, _ = training_batch(dataset, indices, target, identity, seed, 1)
    xa, _, changed = training_batch(dataset, indices, target, action, seed, 1)
    with torch.no_grad():
        l0, la = model(x0.to(device)), model(xa.to(device))
        y = y.to(device)
        correct0, correcta = l0.argmax(1) == y, la.argmax(1) == y
        n = int(correct0.sum())
        return {"reference_correctness": float(correct0.float().mean()),
                "candidate_correctness": float(correcta.float().mean()),
                "conditional_retention": float(correcta[correct0].float().mean()) if n else None,
                "retention_denominator": n,
                "excess_loss": float(nn.functional.cross_entropy(la, y) - nn.functional.cross_entropy(l0, y)),
                "target_examples_first_batch": len(indices), "changed_target_examples_first_batch": changed}


def run_branch(saved, train_data, panels, plan, target, action, seed, horizons, device, eval_batch_size):
    model, optimizer, scheduler = restore_snapshot(saved, device)
    # Common random stream for this episode, identical across all candidates/reference.
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    initial = torch.cat([p.detach().cpu().flatten() for p in model.parameters()])
    results, first_gradient, first_delta, changed_total = {}, None, None, 0
    begin = time.monotonic()
    for step, indices in enumerate(plan, 1):
        x, y, changed = training_batch(train_data, indices, target, action, seed, step)
        changed_total += changed
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x.to(device)), y.to(device))
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite training loss")
        loss.backward()
        if step == 1:
            first_gradient = torch.cat([p.grad.detach().cpu().flatten() for p in model.parameters()])
        optimizer.step()
        if step == 1:
            first_delta = torch.cat([p.detach().cpu().flatten() for p in model.parameters()]) - initial
        # LOCAL interventions: hold snapshot LR constant; do not advance epoch scheduler.
        if step in horizons:
            before_eval = capture_rng()
            results[str(step)] = {role: evaluate(model, ds, device, eval_batch_size, predictions=True)
                                  for role, ds in panels.items()}
            restore_rng(before_eval)
    state = snapshot(model, optimizer, scheduler, saved["epoch"], saved["config"])
    return {"metrics": results, "first_gradient": first_gradient, "first_delta": first_delta,
            "state": state, "changed_target_examples": changed_total,
            "seconds": time.monotonic() - begin}


def equal_state(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return isinstance(b, np.ndarray) and np.array_equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(equal_state(x, y) for x, y in zip(a, b))
    return a == b


def identity_check(saved, train_data, panels, steps, batch_size, seed, device, eval_batch_size):
    plan = balanced_plan(train_data.rows, steps, batch_size, seed)
    action = {"operation": "identity", "magnitude": 0.0}
    reference = run_branch(saved, train_data, panels, plan, "D00", action, seed, [steps], device, eval_batch_size)
    # Deliberately perturb global RNG to test branch isolation/restoration.
    torch.rand(19)
    random.random()
    np.random.random(13)
    candidate = run_branch(saved, train_data, panels, plan, "D00", action, seed, [steps], device, eval_batch_size)
    checks = {key: equal_state(reference["state"][key], candidate["state"][key])
              for key in ("model", "optimizer", "scheduler", "rng")}
    checks["metrics_and_predictions"] = equal_state(reference["metrics"], candidate["metrics"])
    checks["first_gradient"] = equal_state(reference["first_gradient"], candidate["first_gradient"])
    checks["first_update"] = equal_state(reference["first_delta"], candidate["first_delta"])
    checks["no_changed_images"] = reference["changed_target_examples"] == candidate["changed_target_examples"] == 0
    result = {"passed": all(checks.values()), "checks": checks, "steps": steps, "batch_size": batch_size,
              "max_model_absolute_difference": max(float((reference["state"]["model"][k].double()
                   - candidate["state"]["model"][k].double()).abs().max()) for k in saved["model"]),
              "device": device, "tolerance": "exact equality in this execution environment",
              "test_evaluated": False, "seconds": reference["seconds"] + candidate["seconds"]}
    return result


def environment(device):
    return {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "torch": str(torch.__version__), "numpy": np.__version__, "pillow": PIL.__version__,
            "platform": platform.platform(), "host": socket.gethostname(), "device": device,
            "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "cuda_build": torch.version.cuda, "argv": sys.argv}


def setup_run(args, design, data_config, extra):
    config = {"version": VERSION, "mode": args.mode, "code_hashes": code_hashes(),
              "manifest_sha256": data_config["manifest_sha256"],
              "design_sha256": design["design_sha256"], "test_evaluated": False,
              "device": args.device, "seed": args.seed, "threads": args.threads,
              "batch_size": args.batch_size, "eval_batch_size": args.eval_batch_size,
              "libraries": {"torch": str(torch.__version__), "numpy": np.__version__, "pillow": PIL.__version__},
              **extra}
    if args.resume:
        if read_json(args.out / "config.json") != config:
            raise ValueError("Resume configuration/code/environment mismatch")
    else:
        args.out.mkdir(parents=True, exist_ok=False)
        base.json_write(args.out / "config.json", config)
    base.json_write(args.out / ("environment_resume.json" if args.resume else "environment.json"), environment(args.device))
    base.json_write(args.out / "run_status.json", {"status": "running", "test_evaluated": False})
    return config


def train_trajectory(args, splits, checksums, panels, design, data_config):
    if args.epochs < 2 or any(e < 1 or e > args.epochs for e in args.snapshots):
        raise ValueError("Snapshots must be within a planned trajectory of at least two epochs")
    config = setup_run(args, design, data_config,
                       {"epochs": args.epochs, "snapshots": args.snapshots,
                        "lr": args.lr, "weight_decay": args.weight_decay, "workers": args.workers,
                        "policy": base.SHARED_POLICY, "sampling": "ordinary shuffled training data",
                        "checkpoint_selection": "predetermined epochs; no best-validation selection"})
    training = base.CropDataset(args.data, splits["train"], checksums, "shared", args.seed)
    reference = base.CropDataset(args.data, panels["reference"], checksums)
    first, history = 1, []
    if args.resume:
        saved = checked_snapshot(args.out / "last.pt", design, data_config, args.device)
        model, optimizer, scheduler = restore_snapshot(saved, args.device)
        first, history = saved["epoch"] + 1, saved["history"]
        # Repair a snapshot interrupted after writing last.pt but before copying it.
        if saved["epoch"] in args.snapshots and not (args.out / ("epoch_%03d.pt" % saved["epoch"])).exists():
            base.torch_write(args.out / ("epoch_%03d.pt" % saved["epoch"]), saved)
    else:
        model = base.CompactCNN().to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    begin, completed = time.monotonic(), first - 1
    for epoch in range(first, args.epochs + 1):
        training.epoch = epoch
        t0, lr = time.monotonic(), optimizer.param_groups[0]["lr"]
        metrics, _, _ = base.epoch_pass(model, base.loader(training, args, True), args.device, optimizer)
        scheduler.step()
        row = {"epoch": epoch, "lr": lr, "train": metrics}
        if epoch in args.snapshots:
            row["reference"] = evaluate(model, reference, args.device, args.eval_batch_size)
        row["seconds"] = time.monotonic() - t0
        history.append(row)
        saved = snapshot(model, optimizer, scheduler, epoch, config)
        saved["history"] = history
        base.torch_write(args.out / "last.pt", saved)
        if epoch in args.snapshots:
            base.torch_write(args.out / ("epoch_%03d.pt" % epoch), saved)
        base.json_write(args.out / "history.json", history)
        completed = epoch
        print("TRAJECTORY epoch %d/%d | %.1fs | train loss %.5f" %
              (epoch, args.epochs, row["seconds"], metrics["loss"]), flush=True)
        if (args.stop_after_epoch and epoch >= args.stop_after_epoch) or (
                args.max_minutes and time.monotonic() - begin >= 60 * args.max_minutes):
            break
    complete = completed == args.epochs
    base.json_write(args.out / "run_status.json", {"status": "complete" if complete else "paused",
                     "completed_epochs": completed, "planned_epochs": args.epochs, "test_evaluated": False})
    print("TRAJECTORY " + ("COMPLETE" if complete else "PAUSED; repeat with --resume"), flush=True)


def cosine(a, b):
    denominator = float(a.norm() * b.norm())
    return float(torch.dot(a, b) / denominator) if denominator else None


def gains(reference, candidate):
    return {"balanced_loss_gain": reference["balanced_loss"] - candidate["balanced_loss"],
            "macro_f1_gain": candidate["macro_f1"] - reference["macro_f1"],
            "accuracy_gain": candidate["accuracy"] - reference["accuracy"],
            "class_loss_gain": {k: reference["class_loss"][k] - candidate["class_loss"][k] for k in LABELS},
            "class_recall_gain": {k: candidate["per_class"][k]["recall"] - reference["per_class"][k]["recall"]
                                  for k in LABELS}}


def export_results(out):
    """Export measured outcomes without choosing a winning algorithm on audit data."""
    records = [read_json(p) for p in sorted((out / "trials").glob("*.json"))]
    rows = []
    for record in records:
        for horizon, outcome in record["outcomes"].items():
            row = {"class": record["target_class"], "episode": record["episode"],
                   "candidate": record["action"]["id"], "operation": record["action"]["operation"],
                   "magnitude": record["action"]["magnitude"], "horizon": int(horizon),
                   **record["signals"]}
            for role in ("reference", "audit"):
                effect = outcome[role]
                row[role + "_balanced_loss_gain"] = effect["balanced_loss_gain"]
                row[role + "_macro_f1_gain"] = effect["macro_f1_gain"]
                for label in LABELS:
                    row[role + "_" + label + "_loss_gain"] = effect["class_loss_gain"][label]
                    row[role + "_" + label + "_recall_gain"] = effect["class_recall_gain"][label]
            rows.append(row)
    if rows:
        path = out / "results.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)


def probe(args, saved, splits, checksums, panels, design, data_config):
    invocation_started = time.monotonic()
    ids = {a["id"] for a in design["bank"]}
    if args.candidates and not set(args.candidates).issubset(ids):
        raise ValueError("Unknown candidate ID")
    actions = [a for a in design["bank"] if not args.candidates or a["id"] in args.candidates]
    identity = next(a for a in design["bank"] if a["operation"] == "identity")
    config = setup_run(args, design, data_config, {"checkpoint_sha256": base.sha256(args.checkpoint.read_bytes()),
                       "horizons": args.horizons, "classes": args.classes, "episodes": args.episodes,
                       "candidates": [a["id"] for a in actions], "checkpoint_epoch": saved["epoch"],
                       "sampling": "equal class counts per diagnostic batch; cycle shuffled pools",
                       "scheduler": "held fixed during local interventions", "identity_required": True})
    trials = args.out / "trials"
    trials.mkdir(exist_ok=True)
    for path in trials.glob("*.json"):
        record = read_json(path)
        expected_name = "%s_e%03d_%s.json" % (record["target_class"], record["episode"], record["action"]["id"])
        if (record.get("config_sha256") != digest(config) or path.name != expected_name
                or record["action"]["id"] not in config["candidates"]
                or record["target_class"] not in args.classes or not 0 <= record["episode"] < args.episodes):
            raise ValueError("Existing trial does not match the resumed experiment")
    datasets = {k: base.CropDataset(args.data, v, checksums) for k, v in panels.items()}
    training = base.CropDataset(args.data, splits["train"], checksums)
    check = identity_check(saved, training, datasets, 2, args.batch_size, args.seed,
                           args.device, args.eval_batch_size)
    base.json_write(args.out / "identity_control.json", check)
    if not check["passed"]:
        raise RuntimeError("Identity control failed; candidate trials are disabled")
    print("IDENTITY CONTROL PASSED", flush=True)
    model, _, _ = restore_snapshot(saved, args.device)
    g_ref = reference_gradient(model, datasets["reference"], args.device, args.eval_batch_size)
    del model
    start, paused = time.monotonic(), False
    for label in args.classes:
        for episode in range(args.episodes):
            seed = args.seed + 1009 * episode
            pending = [a for a in actions if not (trials / ("%s_e%03d_%s.json" % (label, episode, a["id"]))).exists()]
            if not pending:
                continue
            plan = balanced_plan(training.rows, max(args.horizons), args.batch_size, seed)
            reference = run_branch(saved, training, datasets, plan, label, identity, seed,
                                   args.horizons, args.device, args.eval_batch_size)
            plan_ids = [[training.rows[i]["sample_id"] for i in batch] for batch in plan]
            plan_hash = digest(plan_ids)
            base.json_write(args.out / ("batch_plan_e%03d.json" % episode),
                            {"sample_ids": plan_ids, "sha256": plan_hash, "seed": seed})
            for action in pending:
                signals = difficulty_signals(saved, training, plan, label, action, seed, args.device)
                candidate = run_branch(saved, training, datasets, plan, label, action, seed,
                                       args.horizons, args.device, args.eval_batch_size)
                signals["gradient_cosine"] = cosine(g_ref, candidate["first_gradient"])
                signals["reference_gradient_cosine"] = cosine(g_ref, reference["first_gradient"])
                signals["incremental_gradient_dot"] = float(torch.dot(g_ref, candidate["first_gradient"] - reference["first_gradient"]))
                # Actual AdamW displacement includes moment/preconditioning effects.
                # BN buffer changes and higher-order effects are NOT modeled by this linear score.
                signals["adamw_linear_gain"] = float(-torch.dot(g_ref, candidate["first_delta"] - reference["first_delta"]))
                signals["reference_one_step_gain"] = (reference["metrics"]["1"]["reference"]["balanced_loss"]
                    - candidate["metrics"]["1"]["reference"]["balanced_loss"])
                outcomes = {str(h): {role: gains(reference["metrics"][str(h)][role], candidate["metrics"][str(h)][role])
                                     for role in datasets} for h in args.horizons}
                result = {"target_class": label, "episode": episode, "action": action,
                          "signals": signals, "outcomes": outcomes, "batch_plan_sha256": plan_hash,
                          "reference_metrics": reference["metrics"], "candidate_metrics": candidate["metrics"],
                          "changed_target_examples": candidate["changed_target_examples"],
                          "reference_seconds": reference["seconds"], "candidate_seconds": candidate["seconds"],
                          "config_sha256": digest(config), "test_evaluated": False}
                base.json_write(trials / ("%s_e%03d_%s.json" % (label, episode, action["id"])), result)
                print("TRIAL COMPLETE | %s episode %d %s | audit gain %.6f" %
                      (label, episode, action["id"], outcomes[str(max(args.horizons))]["audit"]["balanced_loss_gain"]), flush=True)
                if args.max_minutes and time.monotonic() - start >= 60 * args.max_minutes:
                    paused = True
                    break
            if paused:
                break
        if paused:
            break
    export_results(args.out)
    completed = len(list(trials.glob("*.json")))
    expected = len(args.classes) * args.episodes * len(actions)
    base.json_write(args.out / "run_status.json", {"status": "complete" if completed == expected else "paused",
                     "completed_trials": completed, "expected_trials": expected,
                     "test_evaluated": False, "seconds_this_invocation_after_identity": time.monotonic() - start,
                     "seconds_this_probe_invocation": time.monotonic() - invocation_started})
    print("PROBE " + ("COMPLETE" if completed == expected else "PAUSED; repeat with --resume"), flush=True)


def integers(text):
    values = sorted(set(int(x) for x in text.split(",")))
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Expected positive comma-separated integers")
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preview", "train", "identity", "probe"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("dataset_splits_v1/crops.csv"))
    parser.add_argument("--design", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--bank", type=Path, help="Optional JSON candidate list for a NEW design")
    parser.add_argument("--qa-dir", type=Path, help="Local JPEG contact sheets; preview only")
    parser.add_argument("--preview-per-cell", type=int, default=2)
    parser.add_argument("--eval-per-class", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--snapshots", type=integers, default=[2, 10, 18])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--steps", type=int, default=3, help="Identity-control steps")
    parser.add_argument("--horizons", type=integers, default=[1, 10, 50])
    parser.add_argument("--classes", type=lambda s: s.split(","), default=LABELS.copy())
    parser.add_argument("--candidates", type=lambda s: s.split(","))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-minutes", type=float, default=75)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, default=0)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.eval_batch_size, args.eval_per_class, args.threads, args.steps,
           args.episodes, args.preview_per_cell) < 1 or args.preview_per_cell > 4 or args.workers < 0:
        parser.error("Invalid positive count, worker count, or preview count (maximum four)")
    if args.max_minutes < 0 or args.lr <= 0 or args.weight_decay < 0:
        parser.error("Invalid time/optimizer setting")
    if len(set(args.classes)) != len(args.classes) or not set(args.classes).issubset(LABELS) or not args.classes:
        parser.error("Invalid class list")
    if args.resume and args.mode not in ("train", "probe"):
        parser.error("Resume is supported for train/probe only")
    if args.qa_dir and (args.mode != "preview" or args.data):
        parser.error("QA proxies are for preview only and cannot be used with --data")
    if args.mode != "prepare" and not args.design:
        parser.error("--design is required")
    if args.mode in ("train", "identity", "probe") and not args.data:
        parser.error("--data is required")
    if args.mode == "preview" and not (args.data or args.qa_dir):
        parser.error("Preview needs --data or --qa-dir")
    if args.mode in ("identity", "probe") and not args.checkpoint:
        parser.error("--checkpoint is required")
    if args.mode in ("identity", "probe") and args.batch_size % 4:
        parser.error("Diagnostic --batch-size must be a multiple of four")
    if args.mode == "probe" and 1 not in args.horizons:
        parser.error("--horizons must include 1 for the one-step feedback signal")
    args.out = args.out.resolve()
    for root in (args.data, args.design, args.qa_dir, args.manifest.parent):
        if root and (args.out == root.resolve() or root.resolve() in args.out.parents):
            parser.error("Output must be separate from data/design/manifest/QA input directories")
    if args.mode == "prepare":
        prepare(args)
        return 0
    if args.mode == "preview":
        preview(args)
        return 0
    if args.device == "cuda" and (not torch.cuda.is_available() or torch.cuda.device_count() != 1):
        raise RuntimeError("Expected exactly one allocated visible CUDA GPU")
    base.seed_all(args.seed, args.threads)
    splits, data_config = base.read_manifest(args.data)
    checksums = base.read_checksums(args.data, splits)
    design, panels = load_design(args.design, splits, data_config["manifest_sha256"])
    try:
        if args.mode == "train":
            train_trajectory(args, splits, checksums, panels, design, data_config)
        else:
            saved = checked_snapshot(args.checkpoint, design, data_config, args.device)
            if args.mode == "probe":
                probe(args, saved, splits, checksums, panels, design, data_config)
            else:
                setup_run(args, design, data_config, {"checkpoint_sha256": base.sha256(args.checkpoint.read_bytes()),
                                                    "steps": args.steps})
                datasets = {k: base.CropDataset(args.data, v, checksums) for k, v in panels.items()}
                training = base.CropDataset(args.data, splits["train"], checksums)
                result = identity_check(saved, training, datasets, args.steps, args.batch_size,
                                        args.seed, args.device, args.eval_batch_size)
                base.json_write(args.out / "identity_control.json", result)
                if not result["passed"]:
                    raise RuntimeError("Identity control failed")
                base.json_write(args.out / "run_status.json", {"status": "complete", "test_evaluated": False})
                print("IDENTITY CONTROL PASSED | exact states, predictions and metrics", flush=True)
    except Exception as exc:
        # Preserve earlier evidence when startup/resume configuration is rejected.
        status = args.out / "run_status.json"
        if status.exists() and read_json(status).get("status") == "running":
            base.json_write(status, {"status": "failed", "error": str(exc), "test_evaluated": False})
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
