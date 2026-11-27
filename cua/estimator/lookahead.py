"""Lookahead class-utility estimator (CUA v2; the fixes after the Part C diagnosis).

The v1 estimator (cua.estimator.utility) scored an edit by a one-step, first-order gradient
alignment. The offline check (cua.analysis.utility_check) found that signal carried no class-level
information and over-rated mosaic. v2 measures the effect directly:

  1. Start from the current weights. For the clean batch and for each edit o, reset theta_s, take
     `steps` real SGD steps (lr = training lr x lr_scale) on the batch rendered with o, then
     evaluate the per-class reference loss L_c.
     u(o, c) = (L_c(after clean steps) - L_c(after o steps)) / (L_c(before) + delta)
     Positive = training on the edited images lowers class c's reference loss more than training
     on the same images unedited. All options share the images, the reference batch and the RNG
     seeds, so the comparison is paired.
  2. instance_norm: each edited step's loss is scaled by (objects in the clean step / objects in the
     edited step), so an edit that packs more objects into the batch (mosaic) does not get a larger
     update just for that.

  3. Each step's gradient is clipped to global norm `clip` (RetinaNet's head diverged within 4
     unclipped steps at 10x the training lr on CCAST). A round whose utilities are still not finite
     is skipped (trackers not updated, `skipped` set) instead of stopping the training run.

Only theta_s moves and it is restored afterwards; optimizer state and .grad are never touched.
Every pass runs in float32. The trackers and the returned dict match Estimator, so the trainer and
the policy use either one unchanged.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from cua.estimator.class_loss import class_losses
from cua.estimator.params import only_theta, theta_params
from cua.estimator.utility import REF_CHUNK, EmaTracker, stratified_reference


def _n_objects(items) -> int:
    return int(sum(int(t["labels"].numel()) for _, t in items))


class LookaheadEstimator:
    def __init__(self, model, detector: str, theta: str, menu: tuple, num_classes: int, *,
                 est_batch: int = 16, steps: int = 4, ref_batch: int = 32, lr_scale: float = 10.0,
                 instance_norm: bool = True, delta: float = 0.01, beta: float = 0.98, clip: float = 5.0,
                 device: torch.device | None = None, workers: int = 4):
        if steps < 1 or est_batch < steps:
            raise ValueError(f"need est_batch >= steps >= 1, got est_batch={est_batch}, steps={steps}")
        self.model, self.detector, self.theta = model, detector, theta
        self.menu, self.num_classes = tuple(menu), int(num_classes)
        self.est_batch, self.steps, self.ref_batch = int(est_batch), int(steps), int(ref_batch)
        self.lr_scale, self.instance_norm, self.delta = float(lr_scale), bool(instance_norm), float(delta)
        self.clip = float(clip)
        self.skipped = 0
        self.device = device if device is not None else next(model.parameters()).device
        self.workers = max(1, int(workers))
        self.cua = EmaTracker((len(self.menu), self.num_classes), beta)
        self.shared = EmaTracker((len(self.menu), 1), beta)

    def _to_device(self, items):
        images = [im.to(self.device) for im, _ in items]
        targets = [{"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)} for _, t in items]
        return images, targets

    def _ref_losses(self, refs, seed: int):
        sums = torch.zeros(self.num_classes, device=self.device, dtype=torch.float64)
        counts = torch.zeros(self.num_classes, dtype=torch.int64, device=self.device)
        with torch.no_grad():
            for s in range(0, len(refs), REF_CHUNK):
                images, targets = self._to_device(refs[s:s + REF_CHUNK])
                torch.manual_seed(seed)
                cl = class_losses(self.model, self.detector, images, targets, self.num_classes)
                sums += cl.sums.double()
                counts += cl.counts
        return sums / counts.clamp(min=1).double(), counts > 0

    def _steps(self, items, clean, params, step_lr: float, seed: int, scale_objects: bool) -> None:
        k = self.steps
        chunks = [items[i::k] for i in range(k)]
        clean_chunks = [clean[i::k] for i in range(k)]
        for j, (chunk, cchunk) in enumerate(zip(chunks, clean_chunks)):
            images, targets = self._to_device(chunk)
            torch.manual_seed(seed + j)
            loss = sum(self.model(images, targets).values()).float()
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            f = 1.0
            if self.clip > 0:
                norm = torch.sqrt(sum((g.double() ** 2).sum() for g in grads if g is not None))
                f = min(1.0, self.clip / max(float(norm), 1e-12)) if torch.isfinite(norm) else 0.0
            if scale_objects:  # after the clip, so clipping cannot undo fix 1
                f *= _n_objects(cchunk) / max(_n_objects(chunk), 1)
            if f != 1.0:
                grads = [None if g is None else g * f for g in grads]
            with torch.no_grad():
                for p, g in zip(params, grads):
                    if g is not None:
                        p.sub_(step_lr * g)

    def estimate(self, train_render, ref_render, n_train: int, ref_records, rng: np.random.Generator,
                 lr: float, epoch: int, global_iter: int) -> dict:
        t0 = time.monotonic()
        model, params = self.model, theta_params(self.model, self.detector, self.theta)
        b_idx = [int(i) for i in rng.choice(n_train, size=min(self.est_batch, n_train), replace=False)]
        r_idx = stratified_reference(ref_records, min(self.ref_batch, len(ref_records)), rng)
        pass_seed = int(rng.integers(0, 2**31 - 1 - self.steps))
        jobs = [(i, None) for i in b_idx] + [(i, o) for o in self.menu if o is not None for i in b_idx]
        with ThreadPoolExecutor(self.workers) as ex:
            rendered = list(ex.map(lambda a: train_render(*a), jobs))
            refs = list(ex.map(ref_render, r_idx))
        n = len(b_idx)
        by_edit = {None: rendered[:n]}
        for k, o in enumerate([o for o in self.menu if o is not None]):
            by_edit[o] = rendered[n * (k + 1):n * (k + 2)]

        step_lr = float(lr) * self.lr_scale
        devices = [self.device] if self.device.type == "cuda" else []
        snapshot = [p.detach().clone() for p in params]

        def restore():
            with torch.no_grad():
                for p, s in zip(params, snapshot):
                    p.copy_(s)

        try:
            with torch.random.fork_rng(devices=devices), only_theta(model, params), \
                    torch.autocast(device_type=self.device.type, enabled=False):
                lc0, valid_t = self._ref_losses(refs, pass_seed)
                after = {}
                for o in (None, *[o for o in self.menu if o is not None]):
                    restore()
                    self._steps(by_edit[o], by_edit[None], params, step_lr, pass_seed,
                                scale_objects=self.instance_norm and o is not None)
                    after[o], _ = self._ref_losses(refs, pass_seed)
        finally:
            restore()

        clean = after[None]
        rows = [torch.zeros_like(clean) if o is None else (clean - after[o]) for o in self.menu]
        u = (torch.stack(rows) / (lc0 + self.delta).unsqueeze(0)).cpu().numpy()
        valid_c = valid_t.cpu().numpy()
        valid = np.broadcast_to(valid_c, u.shape).copy()
        if not np.isfinite(u[valid]).all():
            self.skipped += 1
            z = np.zeros_like(u)
            return {"u": z, "Lc": np.nan_to_num(lc0.cpu().numpy()), "valid": np.zeros_like(valid),
                    "u_shared": np.zeros((len(self.menu), 1)), "skipped": True,
                    "seconds": time.monotonic() - t0}
        self.cua.update(u, valid)
        u_shared = np.zeros((len(self.menu), 1))
        if valid_c.any():
            u_shared[:, 0] = u[:, valid_c].mean(axis=1)
        self.shared.update(u_shared, np.full(u_shared.shape, bool(valid_c.any())))
        return {"u": u, "Lc": lc0.cpu().numpy(), "valid": valid, "u_shared": u_shared,
                "seconds": time.monotonic() - t0}
