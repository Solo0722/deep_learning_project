"""Per-class population-based augmentation (PBT); design in
docs/superpowers/plans/2026-11-22-cua-per-class-pbt.md.

N members share seed, data order and start weights and differ only in their edit logit tables
(TrainConfig.policy="table"). Training runs in windows of W epochs. After each window every member is
scored on the reference split; the next window starts every member from the best member's weights
(reference macro AP50) and, per class, from the column of the member with the best reference AP50 on
that class (pooled variant: one column, chosen by macro AP50). Member 0 keeps the selected table, the
others add Gaussian noise (explore).

Layout under runs_root/<pbt_id>/: pbt.json (settings), tables/w<k>-m<j>.json, w<k>-m<j>/ (an ordinary
train() run dir, plus ref_metrics.json), decisions.jsonl (one row per select), final.json.
Every step is idempotent, so any job in the chain can be resubmitted.

CLI: python -m cua.pbt <init|member|select|run> --pbt-id ID --data-root DIR --runs-root DIR
     [--window K] [--member J] [--deadline-seconds S] [--max-train-images N]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from cua.augment.menu import valid_menu
from cua.augment.rng import keyed_rng
from cua.data import DetectionDataset
from cua.evaluate import coco_metrics, predict
from cua.experiments.specs import BUDGET_EPOCHS
from cua.models import build_detector
from cua.records import read_classes, read_records
from cua.train import STOPPED_AT_DEADLINE, TrainConfig, _write_json, full_schedule, read_history, train

PBT_RE = re.compile(r"^PBT-([^-]+)-([^-]+)-s(\d+)-(pc|shared)(?:-n(\d+))?(?:-w(\d+))?$")


@dataclass(frozen=True)
class PBTConfig:
    pbt_id: str
    dataset: str
    detector: str
    seed: int
    shared: bool = False  # pooled control: one column for all classes
    members: int = 4
    window: int = 2  # epochs per window
    epochs: int = 20
    sigma: float = 1.0  # explore noise on logits
    table_eps: float = 0.1

    @property
    def n_windows(self) -> int:
        return -(-self.epochs // self.window)

    def window_epochs(self, k: int) -> tuple[int, int]:
        if not 1 <= k <= self.n_windows:
            raise ValueError(f"window must be in 1..{self.n_windows}, got {k}")
        a = (k - 1) * self.window + 1
        return a, min(k * self.window, self.epochs)


def resolve_pbt(pbt_id: str) -> PBTConfig:
    """PBT-<dataset>-<detector>-s<seed>-<pc|shared>[-n<members>][-w<window>]."""
    m = PBT_RE.match(pbt_id)
    if not m:
        raise KeyError(f"Unknown PBT id {pbt_id!r}: expected PBT-<dataset>-<detector>-s<seed>-<pc|shared>[-n<N>][-w<W>]")
    ds, det = m[1], m[2]
    if ds not in BUDGET_EPOCHS or det not in ("fasterrcnn", "retinanet"):
        raise KeyError(f"Unknown dataset/detector in {pbt_id!r}")
    kw = {}
    if m[5]:
        kw["members"] = int(m[5])
    if m[6]:
        kw["window"] = int(m[6])
    return PBTConfig(pbt_id=pbt_id, dataset=ds, detector=det, seed=int(m[3]), shared=m[4] == "shared",
                     epochs=BUDGET_EPOCHS[ds], **kw)


def member_id(k: int, j: int) -> str:
    return f"w{k}-m{j}"


def _root(p: PBTConfig, runs_root: Path) -> Path:
    return Path(runs_root) / p.pbt_id


def _table_path(p: PBTConfig, runs_root: Path, k: int, j: int) -> Path:
    return _root(p, runs_root) / "tables" / f"{member_id(k, j)}.json"


def _decisions(p: PBTConfig, runs_root: Path) -> list[dict]:
    f = _root(p, runs_root) / "decisions.jsonl"
    if not f.exists():
        return []
    return [json.loads(x) for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]


def _ds(p: PBTConfig, data_root: Path):
    ds_dir = Path(data_root) / p.dataset
    classes = read_classes(ds_dir / "classes.json")
    image_root = Path(json.loads((ds_dir / "source.json").read_text(encoding="utf-8"))["image_root"])
    if not image_root.is_absolute():
        image_root = ds_dir / image_root
    return ds_dir, classes, image_root


def _write_table(path: Path, menu, classes, logits: np.ndarray, shared: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"menu": list(menu), "classes": list(classes), "shared": shared,
                       "logits": np.asarray(logits, dtype=np.float64).tolist()})


def _read_logits(path: Path) -> np.ndarray:
    return np.asarray(json.loads(path.read_text(encoding="utf-8"))["logits"], dtype=np.float64)


def _explore(p: PBTConfig, base: np.ndarray, k: int) -> list[np.ndarray]:
    """Member 0 keeps `base`; member j adds N(0, sigma) noise keyed by (seed, window, member)."""
    out = [base.copy()]
    for j in range(1, p.members):
        rng = keyed_rng(p.seed, k, j, "pbt-explore")
        out.append(base + rng.normal(0.0, p.sigma, size=base.shape))
    return out


def member_config(p: PBTConfig, runs_root: Path, k: int, j: int, winner: int | None) -> TrainConfig:
    a, b = p.window_epochs(k)
    sched = full_schedule(p.epochs)[a - 1:b]
    init = None if k == 1 else f"{p.pbt_id}/{member_id(k - 1, winner)}"
    return TrainConfig(run_id=f"{p.pbt_id}/{member_id(k, j)}", dataset=p.dataset, detector=p.detector,
                       seed=p.seed, epoch_lr=sched, edit=None, init_run=init, first_epoch=a,
                       policy="table", table_path=str(_table_path(p, runs_root, k, j)), table_eps=p.table_eps)


# ---- steps ----

def init(p: PBTConfig, data_root: Path, runs_root: Path) -> None:
    root = _root(p, runs_root)
    settings = asdict(p)
    if (root / "pbt.json").exists():
        stored = json.loads((root / "pbt.json").read_text(encoding="utf-8"))
        if stored != settings:
            raise RuntimeError(f"{root / 'pbt.json'} differs from the current settings; refusing")
    else:
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "pbt.json", settings)
    _, classes, _ = _ds(p, data_root)
    menu = valid_menu(p.dataset)
    cols = 1 if p.shared else len(classes)
    tabs = _explore(p, np.zeros((1 + len(menu), cols)), 1)
    for j, t in enumerate(tabs):
        path = _table_path(p, runs_root, 1, j)
        if not path.exists():
            _write_table(path, menu, ["*"] if p.shared else classes, t, p.shared)


def _winner_before(p: PBTConfig, runs_root: Path, k: int) -> int | None:
    if k == 1:
        return None
    rows = [d for d in _decisions(p, runs_root) if d["window"] == k - 1]
    if not rows:
        raise RuntimeError(f"window {k - 1} has not been selected yet")
    return int(rows[-1]["weights_from"])


def reference_metrics(p: PBTConfig, data_root: Path, run_dir: Path, device: str, workers: int = 4) -> dict:
    ds_dir, classes, image_root = _ds(p, data_root)
    recs = read_records(ds_dir / "reference.jsonl")
    ck = torch.load(run_dir / "last.pt", map_location="cpu", weights_only=True)
    model = build_detector(p.detector, len(classes), 640, pretrained=False)
    model.load_state_dict(ck["model"])
    preds = predict(model, DetectionDataset(recs, image_root, None, 640), device, batch_size=8,
                    num_workers=workers)
    m = coco_metrics(recs, preds, classes)
    return {"macro_ap50": m["macro_ap50"],
            "per_class": {c: (m.get("per_class") or {}).get(c, {}).get("ap50") for c in classes},
            "epoch": int(ck["epoch"])}


def member(p: PBTConfig, data_root: Path, runs_root: Path, k: int, j: int, deadline: float | None = None,
           max_train_images: int | None = None, device: str | None = None, workers: int = 4) -> int:
    if not 0 <= j < p.members:
        raise ValueError(f"member must be in 0..{p.members - 1}, got {j}")
    run_dir = _root(p, runs_root) / member_id(k, j)
    if (run_dir / "ref_metrics.json").exists():
        return 0
    cfg = member_config(p, runs_root, k, j, _winner_before(p, runs_root, k))
    if not (run_dir / "COMPLETE.json").exists():
        code = train(cfg, data_root, runs_root, resume=True, deadline=deadline, max_train_images=max_train_images)
        if code != 0:
            return code
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    _write_json(run_dir / "ref_metrics.json", reference_metrics(p, data_root, run_dir, dev, workers))
    return 0


def _choose(p: PBTConfig, refs: list[dict], classes) -> tuple[int, list[int | None]]:
    """(weights_from, column source per table column). Ties go to the lowest member index; a class
    with no reference AP (no GT) keeps member 0's column (None)."""
    macro = [(-np.inf if r["macro_ap50"] is None else float(r["macro_ap50"])) for r in refs]
    wfrom = int(np.argmax(macro))
    if p.shared:
        return wfrom, [wfrom]
    src = []
    for c in classes:
        v = [r["per_class"].get(c) for r in refs]
        src.append(None if all(x is None for x in v) else
                   int(np.argmax([-np.inf if x is None else float(x) for x in v])))
    return wfrom, src


def select(p: PBTConfig, data_root: Path, runs_root: Path, k: int) -> dict:
    root = _root(p, runs_root)
    done = [d for d in _decisions(p, runs_root) if d["window"] == k]
    if done:
        return done[-1]
    _, classes, _ = _ds(p, data_root)
    menu = valid_menu(p.dataset)
    refs = []
    for j in range(p.members):
        f = root / member_id(k, j) / "ref_metrics.json"
        if not f.exists():
            raise RuntimeError(f"{f} missing: member {j} of window {k} has not finished")
        refs.append(json.loads(f.read_text(encoding="utf-8")))
    wfrom, src = _choose(p, refs, classes)
    logits = [_read_logits(_table_path(p, runs_root, k, j)) for j in range(p.members)]
    sel = logits[0].copy()
    for col, s in enumerate(src):
        if s is not None:
            sel[:, col] = logits[s][:, col]
    row = {"window": k, "weights_from": wfrom, "column_from": src, "selected": sel.tolist(),
           "ref_macro": [r["macro_ap50"] for r in refs],
           "ref_per_class": [r["per_class"] for r in refs]}
    if k < p.n_windows:
        for j, t in enumerate(_explore(p, sel, k + 1)):
            _write_table(_table_path(p, runs_root, k + 1, j), menu, ["*"] if p.shared else classes, t, p.shared)
    else:
        vals = []
        for j in range(p.members):
            h = read_history(root / member_id(k, j))
            vals.append(h[-1]["metrics"]["macro_ap50"] if h else None)
        best = int(np.argmax([-np.inf if v is None else float(v) for v in vals]))
        row["final"] = {"member": best, "val_macro_ap50_last": vals}
        _write_json(root / "final.json", {"member": best, "run": member_id(k, best), "val_macro_ap50": vals[best],
                                          "val_macro_ap50_all": vals})
    with open(root / "decisions.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    _cleanup(p, runs_root, k, wfrom)
    return row


def _cleanup(p: PBTConfig, runs_root: Path, k: int, wfrom: int) -> None:
    """Free checkpoints no later step reads: best.pt of every member of a non-final window, last.pt of
    window k's non-winners and of window k-1's winner (window k+1 starts from window k's winner)."""
    root = _root(p, runs_root)
    final = k == p.n_windows
    for j in range(p.members):
        d = root / member_id(k, j)
        if not final:
            for name in ("best.pt", "val_predictions_best.json", "val_predictions_last.json"):
                (d / name).unlink(missing_ok=True)
            if j != wfrom:
                (d / "last.pt").unlink(missing_ok=True)
    if k > 1:
        prev = [x for x in _decisions(p, runs_root) if x["window"] == k - 1]
        if prev:
            (root / member_id(k - 1, int(prev[-1]["weights_from"])) / "last.pt").unlink(missing_ok=True)


def run_all(p: PBTConfig, data_root: Path, runs_root: Path, deadline: float | None = None,
            max_train_images: int | None = None, device: str | None = None, workers: int = 4) -> int:
    init(p, data_root, runs_root)
    for k in range(1, p.n_windows + 1):
        for j in range(p.members):
            code = member(p, data_root, runs_root, k, j, deadline, max_train_images, device, workers)
            if code != 0:
                return code
        select(p, data_root, runs_root, k)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="cua.pbt")
    ap.add_argument("step", choices=("init", "member", "select", "run"))
    ap.add_argument("--pbt-id", required=True)
    ap.add_argument("--data-root", required=True, type=Path)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--window", type=int)
    ap.add_argument("--member", type=int)
    ap.add_argument("--deadline-seconds", type=float)
    ap.add_argument("--max-train-images", type=int)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    try:
        p = resolve_pbt(a.pbt_id)
    except KeyError as e:
        print(f"error: {e.args[0]}", file=sys.stderr)
        return 2
    deadline = None if a.deadline_seconds is None else time.monotonic() + a.deadline_seconds
    if a.step == "init":
        init(p, a.data_root, a.runs_root)
        return 0
    if a.step == "run":
        return run_all(p, a.data_root, a.runs_root, deadline, a.max_train_images, workers=a.workers)
    if a.window is None:
        ap.error("--window is required")
    if a.step == "select":
        row = select(p, a.data_root, a.runs_root, a.window)
        print(json.dumps({k: row[k] for k in ("window", "weights_from", "column_from", "ref_macro")}))
        return 0
    if a.member is None:
        ap.error("--member is required")
    code = member(p, a.data_root, a.runs_root, a.window, a.member, deadline, a.max_train_images,
                  workers=a.workers)
    if code == STOPPED_AT_DEADLINE:
        print(f"DEADLINE STOP: {a.pbt_id} window {a.window} member {a.member}; resubmit the same job")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
