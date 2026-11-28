"""Detector trainer with exact resume and SLURM deadline stop.

Run layout (`runs_root/<run_id>/`): config.json, history.jsonl, last.pt (full resume state), best.pt
(model weights plus metadata only), val_predictions_last.json, val_predictions_best.json (compact JSON,
boxes rounded to 2 decimals, scores to 4), COMPLETE.json. Resume reads only last.pt. Every write is atomic
(tmp file + os.replace). Per-epoch randomness (shuffle order, torch global RNG used by the
detectors' samplers, keyed augmentation) is derived from (seed, epoch), so a resumed run follows
the same trajectory as an uninterrupted one.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cua.augment.local import SCOPES as EDIT_SCOPES
from cua.augment.menu import PLANT, PLANT_KINDS, PLANT_MOSAIC, ArmTransform, planted_menu, valid_menu
from cua.augment.pixel import PIXEL_EDITS, SCALABLE_EDITS, STRENGTH_LEVELS
from cua.augment.paste import MODES as PASTE_MODES, CopyPaste, adaptive_probs
from cua.datasets.dacl10k_masks import MASKS_FILE, load_masks
from cua.augment.rng import keyed_rng
from cua.data import DetectionDataset, collate
from cua.estimator.lookahead import LookaheadEstimator
from cua.estimator.utility import Estimator
from cua.evaluate import coco_metrics, predict
from cua.models import build_detector
from cua.policy import (IdentityPolicy, TablePolicy, UniformPolicy, UtilityPolicy, UtilityTable,
                        combine_with_prior, shared_prior)
from cua.policy_sampler import PolicyBatchSampler
from cua.records import read_classes, read_records

STOPPED_AT_DEADLINE = 3
_PRE_CHECKPOINT_ARTEFACTS = {"config.json", "best.pt", "val_predictions_last.json", "val_predictions_best.json"}
_INIT_MUST_MATCH = ("dataset", "detector", "size", "pretrained", "seed")
_PENDING = ".pending.e"  # best.pt.pending.e<epoch>, val_predictions_best.json.pending.e<epoch>
_BEST_FILES = ("best.pt", "val_predictions_best.json")


@dataclass(frozen=True)
class TrainConfig:
    run_id: str
    dataset: str
    detector: str
    seed: int
    epoch_lr: tuple[float, ...]
    edit: str | None
    edit_p: float = 0.5
    init_run: str | None = None
    first_epoch: int = 1
    lr_warmup_iters: int = 1000
    base_lr: float = 0.0025
    batch_size: int = 4
    amp: bool = True
    workers: int = 4
    pretrained: bool = True
    size: int = 640
    augment: bool = True
    # Plan 2 (Part C policies). Left out of the digest while at their defaults, so the digests of
    # runs created before these fields existed do not change.
    policy: str = "arm"  # arm (Part A edit/edit_p) | base | uniform | measured | shared | cua
    est_every: int = 200
    est_batch: int = 8
    ref_batch: int = 64
    theta: str = "head_fpn"  # head | head_fpn
    epsilon: float = 0.1
    beta: float = 0.9
    est_start_epoch: int = 2
    weights: str = "equal"  # equal | inv_ap
    harm_block: bool = True
    allow_identity: bool = True
    # CUA v2 (fixes after the Part C diagnosis); also out of the digest at their defaults.
    estimator: str = "grad"  # grad (v1 one-step alignment) | lookahead (v2)
    lookahead_steps: int = 4
    lookahead_lr_scale: float = 10.0
    lookahead_clip: float = 5.0  # global grad-norm clip per lookahead step (0 = off)
    instance_norm: bool = True  # lookahead only
    min_t: float = 0.0  # policy ignores cells with |mean| < min_t * se
    prior: str = "none"  # none | probe (<data>/<dataset>/probe_<detector>.json, combined on the z scale)
    # Per-class PBT (cua.pbt): fixed logit table read from table_path; out of the digest at defaults.
    table_path: str | None = None
    table_eps: float = 0.1
    # Class-adaptive copy-paste (cua.augment.paste); out of the digest at defaults.
    cp_mode: str = "none"  # none | random | balanced | adaptive
    cp_p: float = 0.5
    cp_max: int = 3
    cp_eps: float = 0.2
    cp_region: str = "full"  # full | lower (paste only into the lower half of the image)
    cp_every: int = 1  # adaptive: refresh class probabilities from reference AP50 every N epochs
    cp_mask: bool = False  # paste object polygons (<data>/<ds>/train_masks.json) instead of box rectangles
    # Positive control (cua.augment.menu.PLANT): add a planted edit that deletes one class's annotations.
    # none | frequent (the class with most training boxes) | <class name>; out of the digest at the default.
    plant: str = "none"
    # Kind of planted edit: drop (PLANT, appended to the menu) | mosaic (PLANT_MOSAIC in mosaic's slot;
    # reports/cua_control2/PROTOCOL.md); out of the digest at the default.
    plant_kind: str = "drop"
    # Object-local edits (cua.augment.local): image | object. "object" applies the arm's pixel edit only inside
    # the annotated polygons (<data>/<ds>/train_masks.json); out of the digest at the default.
    edit_scope: str = "image"
    # Strength level of a scalable pixel edit (cua.augment.pixel.STRENGTH_LEVELS; reports/cua_strength/
    # PROTOCOL.md). 1 is the menu strength and is out of the digest, so every existing run keeps its digest.
    edit_strength: int = 1

    def digest(self) -> str:
        """SHA-256 of the config as sorted JSON (tuples become lists); holds no paths.
        Plan-2 fields are omitted while they equal their defaults."""
        d = asdict(self)
        for f in fields(self):
            if f.name in _PLAN2_FIELDS and d[f.name] == f.default:
                del d[f.name]
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode("utf-8")).hexdigest()


_PLAN2_FIELDS = frozenset({"policy", "est_every", "est_batch", "ref_batch", "theta", "epsilon", "beta",
                           "est_start_epoch", "weights", "harm_block", "allow_identity", "estimator",
                           "lookahead_steps", "lookahead_lr_scale", "lookahead_clip", "instance_norm", "min_t",
                           "prior", "table_path", "table_eps", "cp_mode", "cp_p", "cp_max", "cp_eps",
                           "cp_every", "cp_region", "cp_mask", "plant", "plant_kind", "edit_scope",
                           "edit_strength"})
POLICIES = ("arm", "base", "uniform", "measured", "shared", "cua", "table")


def full_schedule(epochs: int) -> tuple[float, ...]:
    """Step schedule: 1.0, then 0.1 from epoch round(0.6E)+1, then 0.01 from round(0.8E)+1."""
    a, b = round(0.6 * epochs) + 1, round(0.8 * epochs) + 1
    return tuple(1.0 if e < a else (0.1 if e < b else 0.01) for e in range(1, epochs + 1))


def read_history(run_dir: Path) -> list[dict]:
    path = Path(run_dir) / "history.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---- atomic I/O ----

def _replace_from_tmp(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    _replace_from_tmp(path, lambda p: p.write_text(text, encoding="utf-8"))


def _write_json(path: Path, obj, compact: bool = False) -> None:
    _write_text(path, json.dumps(obj, separators=(",", ":")) if compact else json.dumps(obj))


def _write_history(path: Path, rows: list[dict]) -> None:
    _write_text(path, "".join(json.dumps(r) + "\n" for r in rows))


def _save(path: Path, obj) -> None:
    _replace_from_tmp(path, lambda p: torch.save(obj, p))


def _load(path: Path, device) -> dict:
    return torch.load(path, map_location=device, weights_only=True)


# ---- helpers ----

def _derived_seed(seed: int, epoch: int, stream: str) -> int:
    return int(keyed_rng(seed, epoch, 0, stream).integers(0, 2**63 - 1))


def _validate(cfg: TrainConfig) -> None:
    if not cfg.epoch_lr:
        raise ValueError("epoch_lr must not be empty")
    if cfg.augment and cfg.size != 640:
        raise ValueError(f"augment=True requires size=640 (mosaic canvas); got size={cfg.size}")
    if not cfg.augment and cfg.edit is not None:
        raise ValueError(f"edit={cfg.edit!r} requires augment=True")
    if cfg.first_epoch < 1:
        raise ValueError("first_epoch must be >= 1")
    if cfg.policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {cfg.policy!r}")
    if cfg.policy != "arm" and cfg.edit is not None:
        raise ValueError(f"policy={cfg.policy!r} chooses edits itself; edit must be None, got {cfg.edit!r}")
    if not 0.0 <= cfg.epsilon <= 1.0:
        raise ValueError(f"epsilon must be in [0, 1], got {cfg.epsilon}")
    if not 0.0 <= cfg.beta < 1.0:
        raise ValueError(f"beta must be in [0, 1), got {cfg.beta}")
    if cfg.theta not in ("head", "head_fpn"):
        raise ValueError(f"theta must be 'head' or 'head_fpn', got {cfg.theta!r}")
    if cfg.weights not in ("equal", "inv_ap"):
        raise ValueError(f"weights must be 'equal' or 'inv_ap', got {cfg.weights!r}")
    if cfg.policy != "arm" and not cfg.augment:
        raise ValueError(f"policy={cfg.policy!r} needs augment=True")
    if cfg.estimator not in ("grad", "lookahead"):
        raise ValueError(f"estimator must be 'grad' or 'lookahead', got {cfg.estimator!r}")
    if cfg.prior not in ("none", "probe"):
        raise ValueError(f"prior must be 'none' or 'probe', got {cfg.prior!r}")
    if cfg.prior != "none" and cfg.policy not in ("shared", "cua"):
        raise ValueError(f"prior={cfg.prior!r} applies to the shared/cua policies only")
    if cfg.min_t < 0:
        raise ValueError(f"min_t must be >= 0, got {cfg.min_t}")
    if (cfg.policy == "table") != (cfg.table_path is not None):
        raise ValueError("policy='table' needs table_path, and table_path needs policy='table'")
    if not 0.0 <= cfg.table_eps <= 1.0:
        raise ValueError(f"table_eps must be in [0, 1], got {cfg.table_eps}")
    if cfg.cp_mode not in ("none", *PASTE_MODES):
        raise ValueError(f"cp_mode must be 'none' or one of {PASTE_MODES}, got {cfg.cp_mode!r}")
    if cfg.cp_mode != "none" and not cfg.augment:
        raise ValueError("copy-paste needs augment=True")
    if cfg.cp_mask and cfg.cp_mode == "none":
        raise ValueError("cp_mask needs a copy-paste mode")
    if cfg.cp_region not in ("full", "lower"):
        raise ValueError(f"cp_region must be 'full' or 'lower', got {cfg.cp_region!r}")
    if not 0.0 <= cfg.cp_p <= 1.0 or not 0.0 <= cfg.cp_eps <= 1.0:
        raise ValueError("cp_p and cp_eps must be in [0, 1]")
    if cfg.plant != "none" and cfg.policy not in ("uniform", "shared", "cua"):
        raise ValueError(f"plant={cfg.plant!r} needs policy uniform, shared or cua, got {cfg.policy!r}")
    if cfg.plant_kind not in PLANT_KINDS:
        raise ValueError(f"plant_kind must be one of {PLANT_KINDS}, got {cfg.plant_kind!r}")
    if cfg.plant_kind != "drop":
        if cfg.plant == "none":
            raise ValueError(f"plant_kind={cfg.plant_kind!r} needs a planted class (plant != 'none')")
        planted_menu(cfg.dataset, cfg.plant_kind)  # raises if the kind does not fit the dataset's menu
    if cfg.edit_scope not in EDIT_SCOPES:
        raise ValueError(f"edit_scope must be one of {EDIT_SCOPES}, got {cfg.edit_scope!r}")
    if cfg.edit_scope == "object" and (cfg.policy != "arm" or cfg.edit not in PIXEL_EDITS):
        raise ValueError(f"edit_scope='object' needs policy='arm' and a pixel edit, got {cfg.policy!r}/{cfg.edit!r}")
    if cfg.edit_strength not in STRENGTH_LEVELS:
        raise ValueError(f"edit_strength must be one of {STRENGTH_LEVELS}, got {cfg.edit_strength!r}")
    if cfg.edit_strength != 1 and (cfg.policy != "arm" or cfg.edit not in SCALABLE_EDITS):
        raise ValueError(f"edit_strength={cfg.edit_strength} needs policy='arm' and one of {SCALABLE_EDITS}, "
                         f"got {cfg.policy!r}/{cfg.edit!r}")
    for name in ("est_every", "est_batch", "ref_batch", "est_start_epoch", "lookahead_steps", "cp_max", "cp_every"):
        if getattr(cfg, name) < 1:
            raise ValueError(f"{name} must be >= 1, got {getattr(cfg, name)}")


def _check_init_run(cfg: TrainConfig, runs_root: Path, max_train_images: int | None) -> Path:
    """The warm run must be complete, share architecture, dataset and seed, use the same training
    subset, and end exactly at first_epoch - 1 (so keyed draws continue it)."""
    init_dir = runs_root / cfg.init_run
    if not (init_dir / "COMPLETE.json").exists():
        raise RuntimeError(f"init_run {cfg.init_run!r} is not complete: {init_dir / 'COMPLETE.json'} missing")
    stored = json.loads((init_dir / "config.json").read_text(encoding="utf-8"))
    init_cfg = stored["config"]
    for key in _INIT_MUST_MATCH:
        if init_cfg.get(key) != getattr(cfg, key):
            raise ValueError(f"init_run {cfg.init_run!r} has {key}={init_cfg.get(key)!r}, "
                             f"this run has {getattr(cfg, key)!r}")
    if stored.get("max_train_images") != max_train_images:
        raise ValueError(f"init_run {cfg.init_run!r} used max_train_images={stored.get('max_train_images')}, "
                         f"this run uses {max_train_images}")
    final = json.loads((init_dir / "COMPLETE.json").read_text(encoding="utf-8")).get("final_epoch")
    if final is None or cfg.first_epoch != int(final) + 1:
        raise ValueError(f"first_epoch={cfg.first_epoch} must be init_run final_epoch + 1 (final_epoch={final})")
    return init_dir / "last.pt"


def _pending_path(out: Path, name: str, epoch: int) -> Path:
    return out / f"{name}{_PENDING}{epoch}"


def _promote_pending(out: Path, epoch: int) -> None:
    """Move epoch's pending best files into place (each os.replace is atomic; idempotent per file)."""
    for name in _BEST_FILES:
        pending = _pending_path(out, name, epoch)
        if pending.exists():
            os.replace(pending, out / name)


def _settle_pending(out: Path, ckpt_epoch: int, best_epoch: int | None) -> None:
    """On resume: promote pending best files of the checkpointed epoch if it was the best epoch;
    delete every other pending file (from an epoch that last.pt does not contain)."""
    if best_epoch is not None and best_epoch == ckpt_epoch:
        _promote_pending(out, ckpt_epoch)
    for f in list(out.iterdir()):
        if _PENDING in f.name:
            f.unlink()


def _prepare_fresh_dir(out: Path, digest: str) -> bool:
    """For resume without last.pt: True (and clear stale pre-checkpoint files) when the directory is
    absent, empty, or holds only this config's artefacts written before the first last.pt."""
    if not out.exists():
        return True
    files = list(out.iterdir())
    if not files:
        return True
    names = {f.name for f in files}
    if any(f.is_dir() for f in files) or "config.json" not in names:
        return False
    if any(n not in _PRE_CHECKPOINT_ARTEFACTS and not n.endswith(".tmp") and _PENDING not in n
           for n in names):
        return False
    if json.loads((out / "config.json").read_text(encoding="utf-8")).get("digest") != digest:
        return False
    for f in files:
        if f.name != "config.json":
            f.unlink()
    return True


def _lr_factor(global_iter: int, warmup_iters: int, warm_up: bool) -> float:
    if not warm_up or warmup_iters <= 0 or global_iter >= warmup_iters:
        return 1.0
    return 0.001 + 0.999 * global_iter / warmup_iters


def _train_one_epoch(model, optimizer, scaler, loader, device, use_amp: bool, cfg: TrainConfig,
                     epoch: int, lr_mult: float, global_iter: int, warm_up: bool,
                     after_step=None) -> tuple[dict, int]:
    """One pass over `loader`; returns (mean losses per key plus 'total', updated global_iter)."""
    model.train()
    sums: dict[str, float] = {}
    n = 0
    for it, (images, targets) in enumerate(loader):
        lr = cfg.base_lr * lr_mult * _lr_factor(global_iter, cfg.lr_warmup_iters, warm_up)
        for group in optimizer.param_groups:
            group["lr"] = lr
        images = [im.to(device, non_blocking=True) for im in images]
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
        with torch.autocast(device_type=device.type, enabled=use_amp):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        values = {k: float(v.detach()) for k, v in loss_dict.items()}
        values["total"] = float(loss.detach())
        if not all(math.isfinite(v) for v in values.values()):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}, iteration {it} "
                               f"(global iteration {global_iter}): {values}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        for k, v in values.items():
            sums[k] = sums.get(k, 0.0) + v
        n += 1
        global_iter += 1
        if after_step is not None:
            after_step(global_iter, lr)
    if n == 0:
        raise RuntimeError(f"Epoch {epoch} saw no training batches")
    return {k: v / n for k, v in sums.items()}, global_iter


def _predictions_json(preds: list[dict]) -> list[dict]:
    """Compact form: boxes rounded to 2 decimals (pixels), scores to 4."""
    return [{"image_index": int(p["image_index"]),
             "boxes": np.round(np.asarray(p["boxes"], np.float64), 2).tolist(),
             "scores": np.round(np.asarray(p["scores"], np.float64), 4).tolist(),
             "labels": np.asarray(p["labels"]).tolist()} for p in preds]


def resolve_plant(plant: str, classes: tuple[str, ...], records) -> int | None:
    """Target class of the planted edit: None (off), the class with most training boxes ("frequent") or the
    second most ("second"; ties: lowest index), or the named class."""
    if plant == "none":
        return None
    if plant in ("frequent", "second"):
        counts = np.zeros(len(classes), dtype=np.int64)
        for r in records:
            np.add.at(counts, np.asarray(r.labels, dtype=np.int64)[~np.asarray(r.ignore, dtype=bool)], 1)
        order = sorted(range(len(classes)), key=lambda c: (-counts[c], c))
        return int(order[0 if plant == "frequent" else 1])
    if plant not in classes:
        raise ValueError(f"plant must be 'none', 'frequent', 'second' or a class name in {list(classes)}, "
                         f"got {plant!r}")
    return classes.index(plant)


def _load_measured(ds_dir: Path, detector: str, classes: tuple[str, ...],
                   menu: tuple[str, ...], name: str = "measured") -> UtilityTable:
    """Part A measured-effect table (written by cua.measured): mean dAP50 and its seed s.e.
    name="probe": the first-arm-epoch table (CUA v2 prior)."""
    path = ds_dir / f"{name}_{detector}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run python -m cua.measured first")
    d = json.loads(path.read_text(encoding="utf-8"))
    if tuple(d["classes"]) != tuple(classes):
        raise ValueError(f"{path}: classes {d['classes']} differ from the dataset's {list(classes)}")
    if tuple(d["edits"]) != tuple(menu):
        raise ValueError(f"{path}: edits {d['edits']} differ from the valid menu {list(menu)}")
    mean = np.asarray(d["mean"], dtype=np.float64).reshape(len(menu), len(classes))
    se = np.asarray(d["se"], dtype=np.float64).reshape(mean.shape)
    return UtilityTable(mean=mean, se=se, count=np.full(mean.shape, int(d["n_seeds"]), dtype=np.int64))


def load_table(path: Path, classes: tuple[str, ...], menu: tuple[str, ...], epsilon: float) -> TablePolicy:
    """PBT logit table JSON: {"menu": [...], "classes": [...], "shared": bool, "logits": [[...]]}."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if tuple(d["menu"]) != tuple(menu):
        raise ValueError(f"{path}: menu {d['menu']} differs from the valid menu {list(menu)}")
    if tuple(d["classes"]) != tuple(classes):
        raise ValueError(f"{path}: classes {d['classes']} differ from the dataset's {list(classes)}")
    return TablePolicy(menu, np.asarray(d["logits"], dtype=np.float64), shared=bool(d["shared"]), epsilon=epsilon)


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def _init_from_warm(ic: dict, model, optimizer, scaler, use_amp: bool) -> None:
    """Arm start: warm run's model, optimizer and (with AMP) GradScaler state from its last.pt."""
    model.load_state_dict(ic["model"])
    optimizer.load_state_dict(ic["optimizer"])
    if use_amp and ic.get("scaler"):
        scaler.load_state_dict(ic["scaler"])


# ---- entry point ----

def train(cfg: TrainConfig, data_root: Path, runs_root: Path, resume: bool = False,
          deadline: float | None = None, max_train_images: int | None = None) -> int:
    """Train `cfg`; returns 0 when complete or 3 when stopped before an epoch at `deadline`
    (an absolute time.monotonic() value)."""
    _validate(cfg)
    data_root, runs_root = Path(data_root), Path(runs_root)
    out = runs_root / cfg.run_id
    digest = cfg.digest()
    init_ckpt = _check_init_run(cfg, runs_root, max_train_images) if cfg.init_run is not None else None

    # Refusal / resume decision (before anything is written).
    ckpt_path = out / "last.pt"
    resume_ckpt = False
    if resume:
        if (out / "COMPLETE.json").exists():
            raise RuntimeError(f"Run {out} is already complete")
        if ckpt_path.exists():
            stored = json.loads((out / "config.json").read_text(encoding="utf-8"))
            if stored.get("digest") != digest:
                raise RuntimeError(f"Config digest differs from {out / 'config.json'}; refusing to resume")
            if stored.get("max_train_images") != max_train_images:
                raise RuntimeError(f"max_train_images={max_train_images} differs from the stored "
                                   f"{stored.get('max_train_images')}; refusing to resume")
            resume_ckpt = True
        elif not _prepare_fresh_dir(out, digest):
            raise RuntimeError(f"Cannot resume {out}: no last.pt and the directory is not empty")
    elif out.exists():
        raise RuntimeError(f"Run directory {out} exists; pass resume=True to continue it")

    # Data: train.jsonl and val.jsonl only (reference/test are never read).
    ds_dir = data_root / cfg.dataset
    classes = read_classes(ds_dir / "classes.json")
    image_root = Path(json.loads((ds_dir / "source.json").read_text(encoding="utf-8"))["image_root"])
    if not image_root.is_absolute():
        image_root = ds_dir / image_root
    train_recs = read_records(ds_dir / "train.jsonl")
    if max_train_images is not None:
        train_recs = train_recs[:max_train_images]
    val_recs = read_records(ds_dir / "val.jsonl")
    plant_class = resolve_plant(cfg.plant, classes, train_recs)
    polygons = load_masks(ds_dir / MASKS_FILE, train_recs) if cfg.edit_scope == "object" else None
    transform = (ArmTransform(cfg.dataset, cfg.edit, cfg.seed, cfg.edit_p, plant_class=plant_class,
                              scope=cfg.edit_scope, polygons=polygons, records=train_recs,
                              strength=cfg.edit_strength, plant_kind=cfg.plant_kind)
                 if cfg.augment else None)
    train_ds = DetectionDataset(train_recs, image_root, transform, cfg.size)
    val_ds = DetectionDataset(val_recs, image_root, None, cfg.size)
    menu = valid_menu(cfg.dataset) if cfg.policy != "arm" else ()
    if plant_class is not None:
        menu = planted_menu(cfg.dataset, cfg.plant_kind)
    measured = _load_measured(ds_dir, cfg.detector, classes, menu) if cfg.policy == "measured" else None
    estimating = cfg.policy in ("shared", "cua")
    adaptive_paste = cfg.cp_mode == "adaptive"
    needs_ref = estimating or adaptive_paste
    ref_recs = read_records(ds_dir / "reference.jsonl") if needs_ref else None
    ref_ds = DetectionDataset(ref_recs, image_root, None, cfg.size) if needs_ref else None
    paste = None
    if cfg.cp_mode != "none":
        masks = load_masks(ds_dir / MASKS_FILE, train_recs) if cfg.cp_mask else None
        paste = CopyPaste(train_recs, len(classes), cfg.cp_mode, cfg.seed, p=cfg.cp_p, max_objects=cfg.cp_max,
                          epsilon=cfg.cp_eps, size=cfg.size, region=cfg.cp_region, masks=masks)
        train_ds.paste = paste

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(cfg.seed)
    model = build_detector(cfg.detector, len(classes), cfg.size, cfg.pretrained).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=cfg.base_lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if not estimating:
        estimator = None
    elif cfg.estimator == "lookahead":
        estimator = LookaheadEstimator(model, cfg.detector, cfg.theta, menu, len(classes), est_batch=cfg.est_batch,
                                       steps=cfg.lookahead_steps, ref_batch=cfg.ref_batch,
                                       lr_scale=cfg.lookahead_lr_scale, instance_norm=cfg.instance_norm,
                                       clip=cfg.lookahead_clip,
                                       beta=cfg.beta, device=device, workers=max(1, cfg.workers))
    else:
        estimator = Estimator(model, cfg.detector, cfg.theta, menu, len(classes), est_batch=cfg.est_batch,
                              ref_batch=cfg.ref_batch, beta=cfg.beta, amp=use_amp, device=device,
                              workers=max(1, cfg.workers))
    prior = None
    if cfg.prior == "probe":
        # plant_mosaic takes mosaic's prior row (it is mosaic on every image without c*; PROTOCOL of control 2)
        real = tuple("mosaic" if e == PLANT_MOSAIC else e for e in menu if e != PLANT)
        prior = _load_measured(ds_dir, cfg.detector, classes, real, "probe")
        if plant_class is not None and PLANT in menu:  # no prior information about PLANT: its row is online-only
            z = np.zeros((1, len(classes)))
            prior = UtilityTable(mean=np.vstack([prior.mean, z]), se=np.vstack([prior.se, z]),
                                 count=np.vstack([prior.count, z.astype(np.int64)]))
    weights = np.ones(len(classes))
    weights_fn = (lambda: weights) if cfg.weights == "inv_ap" else None
    knobs = dict(epsilon=cfg.epsilon, harm_block=cfg.harm_block, allow_identity=cfg.allow_identity)
    if cfg.min_t > 0:  # omitted at the default so v1 policies are constructed exactly as before
        knobs["min_t"] = cfg.min_t
    if cfg.policy == "base":
        main_policy = IdentityPolicy()
    elif cfg.policy == "uniform":
        main_policy = UniformPolicy(menu)
    elif cfg.policy == "measured":
        main_policy = UtilityPolicy(menu, lambda: measured, shared=False, weights_fn=weights_fn,
                                    min_estimates=1, **knobs)
    elif cfg.policy == "shared":
        shared_table = (estimator.shared.table if prior is None else
                        (lambda sp=shared_prior(prior): combine_with_prior(estimator.shared.table(), sp)))
        main_policy = UtilityPolicy(menu, shared_table, shared=True, **knobs)
    elif cfg.policy == "table":
        main_policy = load_table(Path(cfg.table_path), classes, menu, cfg.table_eps)
    elif cfg.policy == "cua":
        cua_table = (estimator.cua.table if prior is None else
                     (lambda: combine_with_prior(estimator.cua.table(), prior)))
        main_policy = UtilityPolicy(menu, cua_table, shared=False, weights_fn=weights_fn, **knobs)
    else:
        main_policy = None
    uniform_policy = UniformPolicy(menu)
    utility_rows: list[dict] = []

    out.mkdir(parents=True, exist_ok=True)
    warm_up = cfg.init_run is None
    global_iter, best, best_epoch = 0, -math.inf, None
    start = cfg.first_epoch
    history: list[dict] = []
    if resume_ckpt:
        ck = _load(ckpt_path, device)
        if ck["cfg_digest"] != digest:
            raise RuntimeError(f"Checkpoint {ckpt_path} digest differs; refusing to resume")
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        global_iter, best, start = int(ck["global_iter"]), float(ck["best_macro"]), int(ck["epoch"]) + 1
        best_epoch = None if ck["best_epoch"] is None else int(ck["best_epoch"])
        _settle_pending(out, int(ck["epoch"]), best_epoch)
        if paste is not None and adaptive_paste:
            paste.set_probs(np.asarray(ck["paste_state"]["probs"], dtype=np.float64))
        if main_policy is not None:
            ps = ck["policy_state"]
            weights[:] = np.asarray(ps["weights"], dtype=np.float64)
            if estimator is not None:
                estimator.cua.load_state_dict(ps["cua"])
                estimator.shared.load_state_dict(ps["shared"])
                utility_rows = ([r for r in _read_jsonl(out / "utility.jsonl") if r["epoch"] < int(ck["epoch"])]
                                + list(ps["utility_rows"]))
                _write_history(out / "utility.jsonl", utility_rows)
        # last.pt is authoritative: drop history beyond it, restore its row if the kill came first.
        history = [h for h in read_history(out) if h["epoch"] < int(ck["epoch"])] + [ck["history_row"]]
        _write_history(out / "history.jsonl", history)
    else:
        _write_json(out / "config.json", {"config": asdict(cfg), "digest": digest,
                                          "max_train_images": max_train_images,
                                          **({"plant_class": classes[plant_class]} if plant_class is not None else {})})
        if init_ckpt is not None:
            _init_from_warm(_load(init_ckpt, device), model, optimizer, scaler, use_amp)

    last_epoch = cfg.first_epoch + len(cfg.epoch_lr) - 1
    prev_seconds = history[-1]["seconds"] if history else None
    for epoch in range(start, last_epoch + 1):
        if (deadline is not None and prev_seconds is not None
                and deadline - time.monotonic() < 1.2 * prev_seconds):
            return STOPPED_AT_DEADLINE
        t0 = time.monotonic()
        lr_mult = float(cfg.epoch_lr[epoch - cfg.first_epoch])
        torch.manual_seed(_derived_seed(cfg.seed, epoch, "torch"))
        train_ds.set_epoch(epoch)
        order = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(
            _derived_seed(cfg.seed, epoch, "shuffle"))).tolist()
        epoch_rows: list[dict] = []
        fallbacks0 = getattr(main_policy, "fallbacks", 0)  # the policy counts cumulatively; log per epoch
        est_seconds = [0.0]
        after_step = None
        sampler = None
        if main_policy is None:
            loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=order, collate_fn=collate,
                                num_workers=cfg.workers, persistent_workers=False,
                                pin_memory=device.type == "cuda")
        else:
            def policy_fn(epoch=epoch):
                return uniform_policy if estimator is not None and epoch < cfg.est_start_epoch else main_policy

            sampler = PolicyBatchSampler(train_recs, policy_fn, cfg.batch_size, order, cfg.seed, epoch, menu,
                                         classes)
            loader = DataLoader(train_ds, batch_sampler=sampler, collate_fn=collate, num_workers=cfg.workers,
                                persistent_workers=False, pin_memory=device.type == "cuda")
            if estimator is not None and epoch >= cfg.est_start_epoch:
                def after_step(gi, lr, epoch=epoch):
                    if gi % cfg.est_every:
                        return
                    res = estimator.estimate(lambda i, e: train_ds[(i, e)], lambda i: ref_ds[i], len(train_ds),
                                             ref_recs, keyed_rng(cfg.seed, epoch, gi, "estimate"), lr, epoch, gi)
                    tab, sh = estimator.cua.table(), estimator.shared.table()
                    epoch_rows.append({
                        "epoch": epoch, "global_iter": gi, "lr": lr, "u": res["u"].tolist(),
                        "mean": tab.mean.tolist(), "se": tab.se.tolist(), "count": tab.count.tolist(),
                        "Lc": res["Lc"].tolist(), "valid": res["valid"].tolist(),
                        "shared_mean": sh.mean[:, 0].tolist(), "shared_se": sh.se[:, 0].tolist(),
                        "seconds": res["seconds"], **({"skipped": True} if res.get("skipped") else {})})
                    est_seconds[0] += res["seconds"]
        paste_row = None
        if paste is not None:
            planned = np.zeros(len(classes), dtype=np.int64)
            for i in range(len(train_ds)):
                for c, _, _ in paste.plan(epoch, i):
                    planned[c] += 1
            paste_row = {"probs": paste.probs.tolist(), "planned": dict(zip(classes, planned.tolist()))}
        losses, global_iter = _train_one_epoch(model, optimizer, scaler, loader, device, use_amp, cfg,
                                               epoch, lr_mult, global_iter, warm_up, after_step)

        preds = predict(model, val_ds, device, batch_size=2 * cfg.batch_size, num_workers=cfg.workers)
        metrics = coco_metrics(val_recs, preds, classes)
        score = -math.inf if metrics["macro_ap50"] is None else float(metrics["macro_ap50"])
        preds_json = _predictions_json(preds)
        _write_json(out / "val_predictions_last.json", preds_json, compact=True)
        improved = score > best or best_epoch is None
        if improved:
            best, best_epoch = score, epoch
        if cfg.weights == "inv_ap":
            per = metrics.get("per_class") or {}
            weights[:] = [1.0 / max(float((per.get(c) or {}).get("ap50") or 1.0), 1.0) for c in classes]
        if adaptive_paste and (epoch - cfg.first_epoch + 1) % cfg.cp_every == 0:
            ref_preds = predict(model, ref_ds, device, batch_size=2 * cfg.batch_size, num_workers=cfg.workers)
            ref_per = coco_metrics(ref_recs, ref_preds, classes).get("per_class") or {}
            ref_ap = [(ref_per.get(c) or {}).get("ap50") for c in classes]
            paste.set_probs(adaptive_probs(ref_ap, paste.has_objects, cfg.cp_eps))
            paste_row["ref_ap50"] = ref_ap
            paste_row["next_probs"] = paste.probs.tolist()
        row = {"epoch": epoch, "lr_mult": lr_mult, "losses": losses, "metrics": metrics,
               "seconds": time.monotonic() - t0}
        if paste_row is not None:
            row["paste"] = paste_row
        if sampler is not None:
            row["policy"] = {**sampler.stats(), "estimator_seconds": est_seconds[0], "estimates": len(epoch_rows),
                             "fallbacks": getattr(main_policy, "fallbacks", 0) - fallbacks0}
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scaler": scaler.state_dict(), "epoch": epoch, "global_iter": global_iter,
                 "cfg_digest": digest, "best_macro": best, "best_epoch": best_epoch, "history_row": row}
        if paste is not None:
            state["paste_state"] = {"probs": paste.probs.tolist()}
        if main_policy is not None:
            state["policy_state"] = {"weights": weights.tolist(), "utility_rows": epoch_rows,
                                     "cua": estimator.cua.state_dict() if estimator else None,
                                     "shared": estimator.shared.state_dict() if estimator else None}
        # Kill safety: best candidates are staged as pending files, committed by last.pt (which records
        # best_epoch), then promoted. Resume promotes or discards pendings according to last.pt.
        if improved:
            _save(_pending_path(out, "best.pt", epoch),
                  {"model": state["model"], "epoch": epoch, "best_macro": best, "best_epoch": best_epoch,
                   "cfg_digest": digest})
            _write_json(_pending_path(out, "val_predictions_best.json", epoch), preds_json, compact=True)
        _save(ckpt_path, state)
        if improved:
            _promote_pending(out, epoch)
        history.append(row)
        _write_history(out / "history.jsonl", history)
        if estimator is not None:
            utility_rows += epoch_rows
            _write_history(out / "utility.jsonl", utility_rows)
        prev_seconds = row["seconds"]

    _write_json(out / "COMPLETE.json", {"final_epoch": last_epoch,
                                        "best_macro_ap50": best if math.isfinite(best) else None})
    return 0
