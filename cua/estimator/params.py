"""The parameter subset theta_s used by the class-utility estimator, and a context that limits
autograd to it."""
from __future__ import annotations

from contextlib import contextmanager

import torch

from cua.models import trainable_head_parameters


def theta_params(model: torch.nn.Module, detector: str, theta: str) -> list[torch.nn.Parameter]:
    """'head': the detector head (Faster R-CNN box head + predictor; RetinaNet classification and
    regression heads). 'head_fpn': plus the top FPN level (inner_blocks[-1], layer_blocks[-1])."""
    params = list(trainable_head_parameters(model, detector))
    if theta == "head_fpn":
        fpn = model.backbone.fpn
        params += [p for m in (fpn.inner_blocks[-1], fpn.layer_blocks[-1]) for p in m.parameters()]
    elif theta != "head":
        raise ValueError(f"theta must be 'head' or 'head_fpn', got {theta!r}")
    seen, out = set(), []
    for p in params:
        if id(p) not in seen:
            seen.add(id(p))
            out.append(p)
    return out


@contextmanager
def only_theta(model: torch.nn.Module, params: list[torch.nn.Parameter]):
    """Turn requires_grad off for every parameter outside `params` (so the backbone is run without
    building a graph); restore the original flags on exit, also on exceptions."""
    keep = {id(p) for p in params}
    saved = [(p, p.requires_grad) for p in model.parameters()]
    try:
        for p, _ in saved:
            if id(p) not in keep:
                p.requires_grad_(False)
        yield
    finally:
        for p, flag in saved:
            p.requires_grad_(flag)
