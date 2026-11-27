"""Batch sampler that picks each training image's edit with the current policy (main process)."""
from __future__ import annotations

import math
from typing import Callable

from cua.augment.rng import keyed_rng
from cua.policy import Policy, choose


def _key(edit: str | None) -> str:
    return "none" if edit is None else edit


class PolicyBatchSampler:
    """Yields lists of (index, edit) in `order`. Each choice uses keyed_rng(seed, epoch, index,
    "policy") and the policy returned by `policy_fn()` at the moment the batch is drawn, so utility
    updates made during the epoch reach later batches. Counts chosen edits overall and per class."""

    def __init__(self, records, policy_fn: Callable[[], Policy], batch_size: int, order: list[int],
                 seed: int, epoch: int, menu: tuple[str, ...], classes: tuple[str, ...]):
        self.records, self.policy_fn, self.batch_size = records, policy_fn, int(batch_size)
        self.order, self.seed, self.epoch = list(order), int(seed), int(epoch)
        self.menu, self.classes = tuple(menu), tuple(classes)
        self._counts: dict[str, int] = {}
        self._by_class: dict[str, dict[str, int]] = {c: {} for c in self.classes}

    def __len__(self) -> int:
        return math.ceil(len(self.order) / self.batch_size)

    def __iter__(self):
        for s in range(0, len(self.order), self.batch_size):
            batch = []
            for i in self.order[s:s + self.batch_size]:
                cs = self.records[i].class_set()
                edit = choose(self.policy_fn(), cs, keyed_rng(self.seed, self.epoch, i, "policy"), self.menu)
                k = _key(edit)
                self._counts[k] = self._counts.get(k, 0) + 1
                for c in cs:
                    d = self._by_class[self.classes[c]]
                    d[k] = d.get(k, 0) + 1
                batch.append((i, edit))
            yield batch

    def stats(self) -> dict:
        return {"edit_counts": dict(self._counts), "edit_by_class": {c: dict(v) for c, v in self._by_class.items()}}
