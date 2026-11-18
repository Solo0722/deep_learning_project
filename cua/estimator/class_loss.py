"""Detection losses split by ground-truth class (spec §9.1: L_c restricted to proposals/anchors
matched to class-c objects; background terms are excluded).

`sums[c]` is the un-normalised sum, over every proposal (Faster R-CNN) or foreground anchor
(RetinaNet) matched to class c, of its classification plus box-regression loss; `counts[c]` is the
number of such units. `n_norm` is torchvision's normaliser for the box term (sampled proposals for
Faster R-CNN; max(1, foreground anchors) for RetinaNet), so sum_c box_sums / n_norm reproduces
torchvision's box loss (for RetinaNet with one image per call, as torchvision normalises per image).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision.ops import boxes as box_ops
from torchvision.ops import sigmoid_focal_loss


@dataclass
class ClassLosses:
    sums: torch.Tensor  # (C,) differentiable, class index 0-based
    counts: torch.Tensor  # (C,) int64
    n_norm: int


def _scatter(values: torch.Tensor, classes: torch.Tensor, num_classes: int) -> torch.Tensor:
    out = values.new_zeros(num_classes)
    return out.index_add(0, classes, values) if values.numel() else out


def _frcnn(model, images, targets, num_classes):
    images, targets = model.transform(images, targets)
    features = model.backbone(images.tensors)
    proposals, _ = model.rpn(images, features, targets)
    rh = model.roi_heads
    proposals, _, labels, reg_targets = rh.select_training_samples(proposals, targets)
    x = rh.box_head(rh.box_roi_pool(features, proposals, images.image_sizes))
    logits, box_reg = rh.box_predictor(x)
    labels, reg_targets = torch.cat(labels), torch.cat(reg_targets)
    pos = torch.where(labels > 0)[0]
    lab = labels[pos]
    ce = F.cross_entropy(logits.float(), labels, reduction="none")[pos]
    box_reg = box_reg.reshape(box_reg.shape[0], -1, 4)
    box = F.smooth_l1_loss(box_reg[pos, lab].float(), reg_targets[pos], beta=1 / 9, reduction="none").sum(1)
    cls_idx = lab - 1
    counts = torch.bincount(cls_idx, minlength=num_classes)[:num_classes]
    return _scatter(ce, cls_idx, num_classes), _scatter(box, cls_idx, num_classes), counts, labels.numel()


def _retinanet(model, images, targets, num_classes):
    images, targets = model.transform(images, targets)
    features = list(model.backbone(images.tensors).values())
    head_out = model.head(features)
    anchors = model.anchor_generator(images, features)
    cls_sum = box_sum = None
    counts = torch.zeros(num_classes, dtype=torch.int64, device=anchors[0].device)
    n_fg = 0
    reg_head = model.head.regression_head
    for i, (t, anc) in enumerate(zip(targets, anchors)):
        if t["boxes"].numel() == 0:
            continue
        matched = model.proposal_matcher(box_ops.box_iou(t["boxes"], anc))
        fg = torch.where(matched >= 0)[0]
        n_fg += fg.numel()
        lab = t["labels"][matched[fg]]
        logits = head_out["cls_logits"][i][fg].float()
        onehot = torch.zeros_like(logits)
        onehot[torch.arange(fg.numel(), device=logits.device), lab] = 1.0
        focal = sigmoid_focal_loss(logits, onehot, reduction="none").sum(1)
        target_reg = reg_head.box_coder.encode_single(t["boxes"][matched[fg]], anc[fg])
        box = F.l1_loss(head_out["bbox_regression"][i][fg].float(), target_reg, reduction="none").sum(1)
        cls_idx = lab - 1
        c_s, b_s = _scatter(focal, cls_idx, num_classes), _scatter(box, cls_idx, num_classes)
        cls_sum = c_s if cls_sum is None else cls_sum + c_s
        box_sum = b_s if box_sum is None else box_sum + b_s
        counts += torch.bincount(cls_idx, minlength=num_classes)[:num_classes]
    if cls_sum is None:
        zero = head_out["cls_logits"].sum() * 0.0
        cls_sum = box_sum = zero.expand(num_classes).clone()
    return cls_sum, box_sum, counts, max(1, n_fg)


def class_losses(model, detector: str, images: list[torch.Tensor], targets: list[dict],
                 num_classes: int, parts: bool = False):
    """Per-class loss sums for a batch (model in train mode; targets with 1-based labels).
    With parts=True also returns (classification sums, box sums)."""
    if detector == "fasterrcnn":
        c_s, b_s, counts, n_norm = _frcnn(model, images, targets, num_classes)
    elif detector == "retinanet":
        c_s, b_s, counts, n_norm = _retinanet(model, images, targets, num_classes)
    else:
        raise ValueError(f"Unknown detector: {detector!r}")
    out = ClassLosses(sums=c_s + b_s, counts=counts, n_norm=int(n_norm))
    return (out, (c_s, b_s)) if parts else out
