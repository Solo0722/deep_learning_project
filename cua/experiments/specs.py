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
    n = epochs if epochs is not None else BUDGET_EPOCHS[dataset]
    if n < 1:
        raise ValueError(f"epochs must be >= 1, got {n}")
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
    if epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {epochs}")
    if len(set(edits)) != len(edits):
        raise ValueError(f"Duplicate edits: {edits!r}")
    menu = valid_menu("rdd")
    for e in edits:
        if e not in menu:
            raise ValueError(f"Edit {e!r} is not valid for dataset 'rdd'")
    return [TrainConfig(run_id=f"proxy-rdd-fasterrcnn-e{epochs}-{edit or 'base'}", dataset="rdd",
                        detector="fasterrcnn", seed=0, epoch_lr=full_schedule(epochs), edit=edit,
                        edit_p=0.5)
            for edit in (None, *edits)]


# Part C (Plan 2). CUA_SETTINGS are the estimator/policy settings frozen after tuning (spec 9.4);
# they are passed explicitly so TrainConfig defaults never move.
CUA_SETTINGS = {"est_every": 200, "theta": "head_fpn", "epsilon": 0.1}
PART_C_POLICIES = ("base", "uniform", "measured", "shared", "cua")
TUNE_VARIANTS = {"default": {}, "k100": {"est_every": 100}, "k400": {"est_every": 400}, "head": {"theta": "head"},
                 "eps0.05": {"epsilon": 0.05}, "eps0.2": {"epsilon": 0.2}}
ABLATIONS = {"noharm": {"harm_block": False}, "noid": {"allow_identity": False}, "invap": {"weights": "inv_ap"}}
ABLATION_DATASETS = ("rdd", "dacl10k")


# Which tuned settings each policy uses. base/uniform use none, so they can start before tuning ends
# and keep their digests whatever CUA_SETTINGS becomes.
_USES_SETTINGS = {"base": (), "uniform": (), "measured": ("epsilon",),
                  "shared": ("est_every", "theta", "epsilon"), "cua": ("est_every", "theta", "epsilon")}


def _c_config(run_id: str, dataset: str, detector: str, seed: int, policy: str, **over) -> TrainConfig:
    tuned = {k: CUA_SETTINGS[k] for k in _USES_SETTINGS[policy]}
    return TrainConfig(run_id=run_id, dataset=dataset, detector=detector, seed=seed,
                       epoch_lr=full_schedule(BUDGET_EPOCHS[dataset]), edit=None, policy=policy,
                       **{**tuned, **over})


# CUA v2 (fixes after the Part C diagnosis): lookahead estimator with instance normalisation (fixes
# 1, 3), larger batch / slower average / |t| gate (fix 2), 1-epoch probe prior (fix 4). Arms
# shared2 and cua2 differ from shared and cua only in these settings.
V2_SETTINGS = {"estimator": "lookahead", "est_every": 400, "est_batch": 16, "ref_batch": 32, "beta": 0.98,
               "min_t": 2.0, "prior": "probe", "theta": "head_fpn", "epsilon": 0.1}
V2_POLICIES = {"shared2": "shared", "cua2": "cua"}


def part_c_specs(dataset: str, detector: str, seeds: tuple[int, ...] = (0, 1, 2),
                 policies: tuple[str, ...] = PART_C_POLICIES) -> list[TrainConfig]:
    _check(dataset, detector)
    for pol in policies:
        if pol not in PART_C_POLICIES and pol not in V2_POLICIES:
            raise ValueError(f"Unknown policy: {pol!r}")
    out = []
    for seed in seeds:
        for pol in policies:
            rid = f"C-{dataset}-{detector}-s{seed}-{pol}"
            if pol in V2_POLICIES:
                out.append(TrainConfig(run_id=rid, dataset=dataset, detector=detector, seed=seed,
                                       epoch_lr=full_schedule(BUDGET_EPOCHS[dataset]), edit=None,
                                       policy=V2_POLICIES[pol], **V2_SETTINGS))
            else:
                out.append(_c_config(rid, dataset, detector, seed, pol))
    return out


def tune_specs() -> list[TrainConfig]:
    """One factor at a time around CUA_SETTINGS (RDD, Faster R-CNN, seed 0, validation only)."""
    return [_c_config(f"T-rdd-fasterrcnn-s0-{name}", "rdd", "fasterrcnn", 0, "cua", **over)
            for name, over in TUNE_VARIANTS.items()]


def ablation_specs(dataset: str, seeds: tuple[int, ...] = (0, 1, 2)) -> list[TrainConfig]:
    if dataset not in ABLATION_DATASETS:
        raise ValueError(f"Ablations run on {ABLATION_DATASETS}, got {dataset!r}")
    return [_c_config(f"X-{dataset}-fasterrcnn-s{seed}-{name}", dataset, "fasterrcnn", seed, "cua", **over)
            for seed in seeds for name, over in ABLATIONS.items()]


# Class-adaptive copy-paste pilot. Every arm uses the uniform edit policy plus the same amount of
# pasting; arms differ only in which classes the pasted objects come from. The no-paste control is the
# existing C-<ds>-<det>-s<seed>-uniform run (identical config without copy-paste).
PASTE_ARMS = {"cprand": "random", "cpbal": "balanced", "cpada": "adaptive"}


def paste_specs(dataset: str, detector: str, seeds: tuple[int, ...] = (0, 1, 2),
                arms: tuple[str, ...] = tuple(PASTE_ARMS)) -> list[TrainConfig]:
    _check(dataset, detector)
    for a in arms:
        if a not in PASTE_ARMS:
            raise ValueError(f"Unknown copy-paste arm: {a!r}")
    return [TrainConfig(run_id=f"P-{dataset}-{detector}-s{seed}-{a}", dataset=dataset, detector=detector,
                        seed=seed, epoch_lr=full_schedule(BUDGET_EPOCHS[dataset]), edit=None, policy="uniform",
                        cp_mode=PASTE_ARMS[a])
            for seed in seeds for a in arms]


_P_RE = re.compile(r"^P-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_C_RE = re.compile(r"^C-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_T_RE = re.compile(r"^T-rdd-fasterrcnn-s0-([^-]+)$")
_X_RE = re.compile(r"^X-([^-]+)-fasterrcnn-s(\d+)-([^-]+)$")
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
        elif m := _P_RE.match(run_id):
            candidates = paste_specs(m[1], m[2], seeds=(int(m[3]),), arms=(m[4],))
        elif m := _C_RE.match(run_id):
            candidates = part_c_specs(m[1], m[2], seeds=(int(m[3]),), policies=(m[4],))
        elif _T_RE.match(run_id):
            candidates = tune_specs()
        elif m := _X_RE.match(run_id):
            candidates = ablation_specs(m[1], seeds=(int(m[2]),))
    except ValueError:
        candidates = []
    for spec in candidates:
        if spec.run_id == run_id:
            return spec
    raise KeyError(f"Unknown run spec {run_id!r}: expected budget-<dataset>-<detector>-e<N>, "
                   f"A-<dataset>-<detector>-s<seed>-<edit|warm|base>, proxy-rdd-fasterrcnn-e<N>-<edit|base>, "
                   f"C-<dataset>-<detector>-s<seed>-<policy>, P-<dataset>-<detector>-s<seed>-<cprand|cpbal|cpada>, T-rdd-fasterrcnn-s0-<variant> or "
                   f"X-<rdd|dacl10k>-fasterrcnn-s<seed>-<ablation>")
