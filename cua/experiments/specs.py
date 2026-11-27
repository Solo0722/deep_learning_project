"""Run specifications (the experiment matrix) and run-id resolution."""
from __future__ import annotations

import re
from dataclasses import replace

from cua.augment.menu import valid_menu
from cua.augment.pixel import PIXEL_EDITS, SCALABLE_EDITS, STRENGTH_LEVELS
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
# Same as the arm without the "m", but pasting object polygons instead of box rectangles (dacl10k only:
# it needs <data>/dacl10k/train_masks.json from cua.datasets.dacl10k_masks).
PASTE_MASK_ARMS = {"cpbalm": "balanced", "cpadam": "adaptive"}
# Road damage only occurs on the road, which fills the lower part of road-scene images.
PASTE_REGION = {"rdd": "lower", "dacl10k": "full", "voc": "full"}


def paste_specs(dataset: str, detector: str, seeds: tuple[int, ...] = (0, 1, 2),
                arms: tuple[str, ...] = tuple(PASTE_ARMS)) -> list[TrainConfig]:
    _check(dataset, detector)
    modes = {**PASTE_ARMS, **PASTE_MASK_ARMS}
    for a in arms:
        if a not in modes:
            raise ValueError(f"Unknown copy-paste arm: {a!r}")
        if a in PASTE_MASK_ARMS and dataset != "dacl10k":
            raise ValueError(f"mask arm {a!r} needs polygons, which only dacl10k has")
    return [TrainConfig(run_id=f"P-{dataset}-{detector}-s{seed}-{a}", dataset=dataset, detector=detector,
                        seed=seed, epoch_lr=full_schedule(BUDGET_EPOCHS[dataset]), edit=None, policy="uniform",
                        cp_mode=modes[a], cp_region=PASTE_REGION[dataset], cp_mask=a in PASTE_MASK_ARMS)
            for seed in seeds for a in arms]


# Positive control: the Part C arms with one planted edit added to the menu (cua.augment.menu.PLANT), which
# deletes the annotations of the most frequent training class. Each arm is otherwise identical to its C- run.
CONTROL_ARMS = ("uniform", "shared", "cua", "shared2", "cua2")


def control_specs(dataset: str, detector: str, seeds: tuple[int, ...] = (0, 1, 2),
                  arms: tuple[str, ...] = CONTROL_ARMS) -> list[TrainConfig]:
    for a in arms:
        if a not in CONTROL_ARMS:
            raise ValueError(f"Unknown control arm: {a!r}")
    out = []
    for c in part_c_specs(dataset, detector, seeds, arms):
        out.append(replace(c, run_id="K" + c.run_id[1:], plant="frequent"))
    return out


LOCAL_EDITS = tuple(e for e in valid_menu("dacl10k") if e in PIXEL_EDITS)


def local_specs(dataset: str = "dacl10k", detector: str = "fasterrcnn", seeds: tuple[int, ...] = (0, 1, 2),
                edits: tuple[str, ...] = LOCAL_EDITS) -> list[TrainConfig]:
    """Object-local pilot (reports/cua_local/PROTOCOL.md): each L- run is its Part A twin with edit_scope='object'.

    It branches from the existing A- warm checkpoint, and the A- base and whole-image arms are its comparisons,
    so no warm or base runs are generated here.
    """
    if dataset != "dacl10k":
        raise ValueError("object-local edits need polygon masks, which only dacl10k has")
    for e in edits:
        if e not in LOCAL_EDITS:
            raise ValueError(f"Unknown or non-pixel edit for the local pilot: {e!r}")
    out = []
    for a in part_a_specs(dataset, detector, seeds):
        if a.edit in edits:
            out.append(replace(a, run_id="L" + a.run_id[1:], edit_scope="object"))
    return out


STRENGTH_EDITS = SCALABLE_EDITS
STRENGTH_PILOT_LEVELS = (1, 2, 3)
STRENGTH_PILOT_P = 1.0


def strength_specs(dataset: str = "dacl10k", detector: str = "fasterrcnn", seeds: tuple[int, ...] = (0, 1, 2),
                   edits: tuple[str, ...] = STRENGTH_EDITS,
                   levels: tuple[int, ...] = STRENGTH_PILOT_LEVELS) -> list[TrainConfig]:
    """Strength pilot (reports/cua_strength/PROTOCOL.md): each S- run is its Part A twin with the edit applied to
    every image (edit_p = 1.0) at strength level 1, 2 or 3.

    It branches from the existing A- warm checkpoint and the A- base arm is its comparison (the base arm applies no
    edit, so its edit_p is irrelevant); no warm or base runs are generated here.
    """
    for e in edits:
        if e not in STRENGTH_EDITS:
            raise ValueError(f"Unknown or non-scalable edit for the strength pilot: {e!r}")
    for lv in levels:
        if lv not in STRENGTH_LEVELS:
            raise ValueError(f"Strength pilot levels must be in {STRENGTH_LEVELS}, got {lv!r}")
    out = []
    for a in part_a_specs(dataset, detector, seeds):
        if a.edit in edits:
            for lv in levels:
                out.append(replace(a, run_id=f"S{a.run_id[1:]}-x{lv}", edit_strength=lv, edit_p=STRENGTH_PILOT_P))
    return out


_S_RE = re.compile(r"^S-([^-]+)-([^-]+)-s(\d+)-([^-]+)-x(\d+)$")
_L_RE = re.compile(r"^L-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_P_RE = re.compile(r"^P-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
_K_RE = re.compile(r"^K-([^-]+)-([^-]+)-s(\d+)-([^-]+)$")
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
        elif m := _S_RE.match(run_id):
            candidates = strength_specs(m[1], m[2], seeds=(int(m[3]),), edits=(m[4],), levels=(int(m[5]),))
        elif m := _L_RE.match(run_id):
            candidates = local_specs(m[1], m[2], seeds=(int(m[3]),), edits=(m[4],))
        elif m := _K_RE.match(run_id):
            candidates = control_specs(m[1], m[2], seeds=(int(m[3]),), arms=(m[4],))
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
                   f"C-<dataset>-<detector>-s<seed>-<policy>, K-<dataset>-<detector>-s<seed>-<control arm>, L-dacl10k-<detector>-s<seed>-<pixel edit>, S-<dataset>-<detector>-s<seed>-<scalable edit>-x<1|2|3>, P-<dataset>-<detector>-s<seed>-<cprand|cpbal|cpada>, T-rdd-fasterrcnn-s0-<variant> or "
                   f"X-<rdd|dacl10k>-fasterrcnn-s<seed>-<ablation>")
