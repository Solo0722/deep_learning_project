"""Faster R-CNN and RetinaNet builders. Labels are 1-based; index 0 is unused (also for RetinaNet)."""
from __future__ import annotations

from typing import Literal

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


def build_detector(name: Literal["fasterrcnn", "retinanet"], num_classes: int, size: int = 640,
                   pretrained: bool = True) -> torch.nn.Module:
    """Detector expecting size x size letterboxed inputs (torchvision's resize is a no-op)."""
    weights = "COCO_V1" if pretrained else None
    common = dict(weights=weights, weights_backbone=None, trainable_backbone_layers=5,
                  min_size=size, max_size=size)
    if name == "fasterrcnn":
        model = fasterrcnn_resnet50_fpn(box_score_thresh=0.001, box_nms_thresh=0.5,
                                        box_detections_per_img=100, **common)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes + 1)
    elif name == "retinanet":
        model = retinanet_resnet50_fpn(score_thresh=0.001, nms_thresh=0.5, detections_per_img=100, **common)
        old = model.head.classification_head
        model.head.classification_head = RetinaNetClassificationHead(
            old.cls_logits.in_channels, model.anchor_generator.num_anchors_per_location()[0], num_classes + 1)
    else:
        raise ValueError(f"Unknown detector: {name!r}")
    return model


def trainable_head_parameters(model: torch.nn.Module, name: str) -> list[torch.nn.Parameter]:
    if name == "fasterrcnn":
        mods = [model.roi_heads.box_head, model.roi_heads.box_predictor]
    elif name == "retinanet":
        mods = [model.head.classification_head, model.head.regression_head]
    else:
        raise ValueError(f"Unknown detector: {name!r}")
    return [p for m in mods for p in m.parameters()]
