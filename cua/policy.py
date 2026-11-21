"""Per-image augmentation policies for the Part C comparison.

A policy maps the class set S of a training image to a distribution over candidate edits, where
`None` is the identity ("no edit"). `choose` samples from it; images without target objects bypass
the policy (identity with p = 0.5, else a uniform menu edit), identically in every policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

import numpy as np

HARM_Z = 1.64
MIN_ESTIMATES = 3
TAU_FLOOR = 1e-6


@dataclass
class UtilityTable:
    """Per (edit, class) utility: mean, standard error and number of updates. Shape (|O|, C);
    a shared table has C = 1."""
    mean: np.ndarray
    se: np.ndarray
    count: np.ndarray


class Policy(Protocol):
    def probs(self, class_set: frozenset[int]) -> tuple[tuple[str | None, ...], np.ndarray]: ...


class IdentityPolicy:
    """Base recipe: never adds an edit."""

    def probs(self, class_set):
        return (None,), np.ones(1)


class UniformPolicy:
    """One edit drawn uniformly from {identity} and the menu (TrivialAugment-style)."""

    def __init__(self, menu: tuple[str, ...]):
        self.candidates = (None, *menu)

    def probs(self, class_set):
        n = len(self.candidates)
        return self.candidates, np.full(n, 1.0 / n)


class UtilityPolicy:
    """Score, harm block and epsilon-softmax sampling over class utilities (spec §9.2, §9.3).

    score(o) = mean over c in S of w_c * mean(o, c), identity scored 0. An edit is blocked when, for
    any c in S with at least `min_estimates` updates, mean + z*se < 0. p = (1-eps) softmax(score/tau)
    + eps/|allowed|, tau = std of the allowed scores (floored). Before any cell has been updated the
    policy is uniform over {identity} and the menu.
    """

    def __init__(self, menu: tuple[str, ...], table_fn: Callable[[], UtilityTable], *, shared: bool,
                 epsilon: float = 0.1, harm_block: bool = True, allow_identity: bool = True,
                 weights_fn: Callable[[], np.ndarray] | None = None,
                 min_estimates: int = MIN_ESTIMATES, z: float = HARM_Z, min_t: float = 0.0):
        self.menu = tuple(menu)
        self.table_fn, self.shared, self.epsilon = table_fn, shared, epsilon
        self.harm_block, self.allow_identity = harm_block, allow_identity
        self.weights_fn, self.min_estimates, self.z = weights_fn, min_estimates, z
        self.min_t = float(min_t)
        self.fallbacks = 0

    def probs(self, class_set):
        t = self.table_fn()
        if not np.any(t.count > 0):
            n = len(self.menu) + 1
            return (None, *self.menu), np.full(n, 1.0 / n)
        cols = [0] if self.shared else sorted(class_set)
        mean, se, count = t.mean[:, cols], t.se[:, cols], t.count[:, cols]
        if self.min_t > 0:
            # CUA v2 (fix 2): a cell counts only once it is clearly non-zero; otherwise it scores 0.
            sure = (count >= self.min_estimates) & (np.abs(mean) >= self.min_t * se)
            mean = np.where(sure, mean, 0.0)
        if self.shared or self.weights_fn is None:
            w = np.ones(len(cols))
        else:
            w = np.asarray(self.weights_fn(), dtype=np.float64)[cols]
            w = w / w.mean() if w.mean() > 0 else np.ones(len(cols))
        scores = (mean * w).mean(axis=1)
        allowed = np.ones(len(self.menu), dtype=bool)
        if self.harm_block:
            harmful = (count >= self.min_estimates) & (mean + self.z * se < 0)
            allowed &= ~harmful.any(axis=1)
        cands = [o for o, a in zip(self.menu, allowed) if a]
        cand_scores = [s for s, a in zip(scores, allowed) if a]
        if self.allow_identity:
            cands, cand_scores = [None, *cands], [0.0, *cand_scores]
        if not cands:
            self.fallbacks += 1
            return (None,), np.ones(1)
        s = np.asarray(cand_scores, dtype=np.float64)
        tau = max(float(np.std(s)), TAU_FLOOR)
        e = np.exp((s - s.max()) / tau)
        p = (1.0 - self.epsilon) * e / e.sum() + self.epsilon / len(s)
        return tuple(cands), p / p.sum()


def choose(policy: Policy, class_set: frozenset[int], rng: np.random.Generator,
           menu: tuple[str, ...]) -> str | None:
    """Sample one edit. Images without target objects: identity (p = 0.5) else a uniform menu edit."""
    if not class_set:
        if rng.random() < 0.5 or not menu:
            return None
        return menu[int(rng.integers(0, len(menu)))]
    cands, p = policy.probs(class_set)
    return cands[int(rng.choice(len(cands), p=p))]


# ---- CUA v2 fix 4: a cheap measured prior combined with the online estimate ----

def _z(t: UtilityTable, min_count: int) -> np.ndarray:
    ok = (t.count >= min_count) & (t.se > 0)
    return np.where(ok, t.mean / np.where(t.se > 0, t.se, 1.0), 0.0), ok


def shared_prior(prior: UtilityTable) -> UtilityTable:
    """Collapse a per-class prior to one column: class-mean effect and its standard error."""
    c = prior.mean.shape[1]
    return UtilityTable(mean=prior.mean.mean(axis=1, keepdims=True),
                        se=np.sqrt((prior.se ** 2).sum(axis=1, keepdims=True)) / c,
                        count=prior.count.min(axis=1, keepdims=True))


def combine_with_prior(online: UtilityTable, prior: UtilityTable, min_count: int = MIN_ESTIMATES) -> UtilityTable:
    """Stouffer combination on the z scale (the two sources have different units): where the online
    cell has at least `min_count` updates, z = (z_prior + z_online) / sqrt(2); otherwise z = z_prior.
    Returned as mean = z, se = 1, so the policy's harm block and |t| gate read z directly."""
    zp, okp = _z(prior, 1)
    zo, oko = _z(online, min_count)
    z = np.where(oko & okp, (zp + zo) / np.sqrt(2.0), np.where(okp, zp, zo))
    count = np.maximum(online.count, np.where(okp, min_count, 0))
    return UtilityTable(mean=z, se=np.ones_like(z), count=count.astype(np.int64))
