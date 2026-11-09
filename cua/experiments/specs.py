"""Run specifications (the experiment matrix) and run-id resolution."""
from __future__ import annotations

import re

from cua.augment.menu import valid_menu
from cua.train import TrainConfig, full_schedule

DETECTORS = ("fasterrcnn", "retinanet")
DATASETS = ("rdd", "voc", "dacl10k")
BUDGET_EPOCHS = {"rdd": 20, "voc": 20, "dacl10k": 30}
_WARM_SCHEDULE = (1.0, 1.0, 1.0)
_ARM_SCHEDULE = (1.0, 1.0, 0.1)
_ARM_FIRST_EPOCH = 4


def _check(dataset: str, detector: str) -> None:
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset!r}")
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector: {detector!r}")


def budget_specs(dataset: str, detector: str, epochs: int | None = None) -> list[TrainConfig]:
    _check(dataset, detector)
    n = epochs or BUDGET_EPOCHS[dataset]
    return [TrainConfig(run_id=f"budget-{dataset}-{detector}-e{n}", dataset=dataset, detector=detector,
                        seed=0, epoch_lr=full_schedule(n), edit=None)]


def part_a_specs(dataset: str, detector: str, seeds: tuple[int, ...] = (0, 1, 2)) -> list[TrainConfig]:
    _check(dataset, detector)
    out: list[TrainConfig] = []
    for seed in seeds:
        stem = f"A-{dataset}-{detector}-s{seed}"
        warm = f"{stem}-warm"
        out.append(TrainConfig(run_id=warm, dataset=dataset, detector=detector, seed=seed,
                               epoch_lr=_WARM_SCHEDULE, edit=None))
        for edit in (None, *valid_menu(dataset)):
            out.append(TrainConfig(run_id=f"{stem}-{edit or 'base'}", dataset=dataset, detector=detector,
                                   seed=seed, epoch_lr=_ARM_SCHEDULE, edit=edit, edit_p=0.5,
                                   init_run=warm, first_epoch=_ARM_FIRST_EPOCH))
    return out


def proxy_specs(edits: list[str], epochs: int) -> list[TrainConfig]:
    menu = valid_menu("rdd")
    for e in edits:
        if e not in menu:
            raise ValueError(f"Edit {e!r} is not valid for dataset 'rdd'")
    return [TrainConfig(run_id=f"proxy-rdd-fasterrcnn-e{epochs}-{edit or 'base'}", dataset="rdd",
                        detector="fasterrcnn", seed=0, epoch_lr=full_schedule(epochs), edit=edit)
            for edit in (None, *edits)]


_BUDGET_RE = re.compile(r"^budget-([^-]+)-([^-]+)-e(\d+)$")
_A_RE = re.compile(r"^A-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_PROXY_RE = re.compile(r"^proxy-rdd-fasterrcnn-e(\d+)-([^-]+)$")


def resolve(run_id: str) -> TrainConfig:
    """Regenerate the spec family a run id belongs to and return the spec with that exact id."""
    candidates: list[TrainConfig] = []
    try:
        if m := _BUDGET_RE.match(run_id):
            candidates = budget_specs(m[1], m[2], int(m[3]))
        elif m := _A_RE.match(run_id):
            candidates = part_a_specs(m[1], m[2], seeds=(int(m[3]),))
        elif m := _PROXY_RE.match(run_id):
            candidates = proxy_specs([] if m[2] == "base" else [m[2]], int(m[1]))
    except ValueError:
        candidates = []
    for spec in candidates:
        if spec.run_id == run_id:
            return spec
    raise KeyError(f"Unknown run spec {run_id!r}: expected budget-<dataset>-<detector>-e<N>, "
                   f"A-<dataset>-<detector>-s<seed>-<edit|warm|base> or proxy-rdd-fasterrcnn-e<N>-<edit|base>")
