"""Online class-utility estimator (spec §9.1, with the 2026-11-16 decisions in Plan 2).

u(o, c) = lr * <g_o - g_0, h_c> / (L_c(R) + delta): the first-order extra reduction in class c's
reference loss from one SGD step on the edited instead of the unedited estimation batch, relative
to the current class loss. g are gradients of the full training loss on the estimation batch; h_c is
the gradient of the per-class masked loss on a shared stratified reference batch; all are restricted
to theta_s, with every other parameter frozen during the estimation passes.

The estimation passes always run in float32, whatever the training run's AMP setting: summed class
losses overflowed fp16 in the backward pass on CCAST (NaN/inf utilities), and reduced precision
changes the sign of small utilities.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from cua.estimator.class_loss import class_losses
from cua.estimator.params import only_theta, theta_params
from cua.policy import UtilityTable

REF_CHUNK = 8


def utility_from_grads(g0: torch.Tensor, go: torch.Tensor, h: torch.Tensor, lc: torch.Tensor,
                       lr: float, delta: float = 0.01) -> np.ndarray:
    """(|O|, C) utilities from flattened gradients: g0 (P,), go (|O|, P), h (C, P), lc (C,)."""
    u = lr * ((go - g0.unsqueeze(0)) @ h.T) / (lc.unsqueeze(0) + delta)
    return u.detach().double().cpu().numpy()


class EmaTracker:
    """Exponential moving mean and variance per cell. A cell's first update sets mean = u, var = 0;
    later: d = u - mean, mean += (1-beta) d, var = beta (var + (1-beta) d^2).
    se = sqrt(var (1-beta) / (1+beta))."""

    def __init__(self, shape: tuple[int, int], beta: float):
        self.beta = float(beta)
        self.mean = np.zeros(shape)
        self.var = np.zeros(shape)
        self.count = np.zeros(shape, dtype=np.int64)

    def update(self, u: np.ndarray, valid: np.ndarray) -> None:
        u = np.asarray(u, dtype=np.float64)
        valid = np.asarray(valid, dtype=bool)
        first = valid & (self.count == 0)
        later = valid & (self.count > 0)
        self.mean[first] = u[first]
        self.var[first] = 0.0
        d = u - self.mean
        b = self.beta
        self.mean[later] += (1.0 - b) * d[later]
        self.var[later] = b * (self.var[later] + (1.0 - b) * d[later] ** 2)
        self.count[valid] += 1

    def table(self) -> UtilityTable:
        b = self.beta
        return UtilityTable(mean=self.mean.copy(), se=np.sqrt(self.var * (1.0 - b) / (1.0 + b)),
                            count=self.count.copy())

    def state_dict(self) -> dict:
        return {"beta": self.beta, "mean": self.mean.tolist(), "var": self.var.tolist(),
                "count": self.count.tolist()}

    def load_state_dict(self, state: dict) -> None:
        self.beta = float(state["beta"])
        self.mean = np.asarray(state["mean"], dtype=np.float64).reshape(self.mean.shape)
        self.var = np.asarray(state["var"], dtype=np.float64).reshape(self.var.shape)
        self.count = np.asarray(state["count"], dtype=np.int64).reshape(self.count.shape)


def stratified_reference(records, n: int, rng: np.random.Generator) -> list[int]:
    """n distinct record indices: one image per class (classes visited in random order), then a
    uniform fill without replacement."""
    classes = sorted(set().union(*(r.class_set() for r in records)))
    if len(classes) > n:
        raise ValueError(f"reference batch of {n} cannot cover {len(classes)} classes")
    chosen: list[int] = []
    taken: set[int] = set()
    for c in rng.permutation(classes):
        pool = [i for i, r in enumerate(records) if int(c) in r.class_set() and i not in taken]
        if not pool:
            continue
        i = pool[int(rng.integers(0, len(pool)))]
        chosen.append(i)
        taken.add(i)
    rest = [i for i in range(len(records)) if i not in taken]
    k = min(n - len(chosen), len(rest))
    if k > 0:
        chosen += [rest[int(j)] for j in rng.choice(len(rest), size=k, replace=False)]
    return chosen


def _flat(grads, params) -> torch.Tensor:
    return torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).float()
                      for g, p in zip(grads, params)])


class Estimator:
    """Holds the class tracker (|O| x C) and the shared tracker (|O| x 1) and updates both."""

    def __init__(self, model, detector: str, theta: str, menu: tuple, num_classes: int, *,
                 est_batch: int = 8, ref_batch: int = 64, delta: float = 0.01, beta: float = 0.9,
                 amp: bool = False, device: torch.device | None = None, workers: int = 4):
        self.model, self.detector, self.theta = model, detector, theta
        self.menu, self.num_classes = tuple(menu), int(num_classes)
        self.est_batch, self.ref_batch, self.delta = est_batch, ref_batch, delta
        self.amp = bool(amp)
        self.device = device if device is not None else next(model.parameters()).device
        self.workers = max(1, int(workers))
        self.cua = EmaTracker((len(self.menu), self.num_classes), beta)
        self.shared = EmaTracker((len(self.menu), 1), beta)

    def _to_device(self, items):
        images = [im.to(self.device) for im, _ in items]
        targets = [{"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)} for _, t in items]
        return images, targets

    def _grad(self, loss, params) -> torch.Tensor:
        grads = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        return _flat(grads, params)

    def estimate(self, train_render, ref_render, n_train: int, ref_records, rng: np.random.Generator,
                 lr: float, epoch: int, global_iter: int) -> dict:
        t0 = time.monotonic()
        g0, go, h, lc, valid_c = self.gradients(train_render, ref_render, n_train, ref_records, rng)
        u = utility_from_grads(g0, go, h, lc, lr, self.delta)
        valid = np.broadcast_to(valid_c, u.shape).copy()
        if not np.isfinite(u[valid]).all():
            raise RuntimeError(f"Non-finite class utility at epoch {epoch}, global iteration {global_iter}: "
                               f"{u.tolist()}")
        self.cua.update(u, valid)
        u_shared = np.zeros((len(self.menu), 1))
        if valid_c.any():
            u_shared[:, 0] = u[:, valid_c].mean(axis=1)
        self.shared.update(u_shared, np.full(u_shared.shape, bool(valid_c.any())))
        return {"u": u, "Lc": lc.cpu().numpy(), "valid": valid, "u_shared": u_shared,
                "seconds": time.monotonic() - t0}

    def gradients(self, train_render, ref_render, n_train: int, ref_records, rng: np.random.Generator):
        """One estimation round's raw quantities: g0 (P,), go (|O|, P), h (C, P), lc (C,) as tensors
        and valid_c (C,) bool. `estimate` turns them into utilities; the offline utility check
        (cua.analysis.utility_check) scores other functions of the same quantities."""
        model, params = self.model, theta_params(self.model, self.detector, self.theta)
        b_idx = [int(i) for i in rng.choice(n_train, size=min(self.est_batch, n_train), replace=False)]
        r_idx = stratified_reference(ref_records, min(self.ref_batch, len(ref_records)), rng)
        pass_seed = int(rng.integers(0, 2**31 - 1))
        jobs = [(i, None) for i in b_idx] + [(i, o) for o in self.menu if o is not None for i in b_idx]
        with ThreadPoolExecutor(self.workers) as ex:
            rendered = list(ex.map(lambda a: train_render(*a), jobs))
            refs = list(ex.map(ref_render, r_idx))
        by_edit = {None: rendered[:len(b_idx)]}
        for k, o in enumerate([o for o in self.menu if o is not None]):
            by_edit[o] = rendered[len(b_idx) * (k + 1):len(b_idx) * (k + 2)]

        devices = [self.device] if self.device.type == "cuda" else []
        autocast = dict(device_type=self.device.type, enabled=False)  # fp32 always (module docstring)
        with torch.random.fork_rng(devices=devices), only_theta(model, params):
            # reference: h_c and L_c, accumulated over chunks
            h = torch.zeros(self.num_classes, sum(p.numel() for p in params), device=self.device)
            sums = torch.zeros(self.num_classes, device=self.device)
            counts = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
            for s in range(0, len(refs), REF_CHUNK):
                images, targets = self._to_device(refs[s:s + REF_CHUNK])
                torch.manual_seed(pass_seed)
                with torch.autocast(**autocast):
                    cl = class_losses(model, self.detector, images, targets, self.num_classes)
                for c in range(self.num_classes):
                    if int(cl.counts[c]) > 0:
                        h[c] += self._grad(cl.sums[c].float(), params)
                sums += cl.sums.detach().float()
                counts += cl.counts
            valid_c = (counts > 0).cpu().numpy()
            denom = counts.clamp(min=1).float()
            h = h / denom.unsqueeze(1)
            lc = sums / denom

            def batch_grad(items):
                images, targets = self._to_device(items)
                torch.manual_seed(pass_seed)
                with torch.autocast(**autocast):
                    loss = sum(model(images, targets).values())
                return self._grad(loss.float(), params)

            g0 = batch_grad(by_edit[None])
            go = torch.stack([g0 if o is None else batch_grad(by_edit[o]) for o in self.menu])
        return g0, go, h, lc, valid_c
