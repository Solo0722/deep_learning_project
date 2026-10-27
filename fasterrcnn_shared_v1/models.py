"""Explicit COCO-to-RDD construction; no automatic downloads or trainer defaults."""
import math

import torch

from integration import require


def build_model(name, size, device, evaluation, pretrained_path=None):
    if name == "fasterrcnn_resnet50_fpn":
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.ops import FrozenBatchNorm2d
        backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                                      norm_layer=FrozenBatchNorm2d, trainable_layers=5)
        model = FasterRCNN(backbone, num_classes=91, min_size=size, max_size=size,
                           box_score_thresh=evaluation["score_threshold"],
                           box_nms_thresh=evaluation["nms_iou"],
                           box_detections_per_img=evaluation["max_detections"])
        # COCO_V1's official builder sets epsilon to zero after loading.
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.0
        transferred = []
        if pretrained_path is not None:
            source = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            model.load_state_dict(source, strict=True)
            transferred = sorted(k for k in source if not k.startswith("roi_heads.box_predictor."))
            del source
        features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(features, 5)

        def unchanged_geometry(module, inputs, output):
            require(all(tuple(shape) == (size, size) for shape in output[0].image_sizes),
                    "Unexpected Faster R-CNN resize")
            require(tuple(output[0].tensors.shape[-2:]) == (size, size), "Unexpected padding")
            if len(inputs) > 1 and inputs[1] is not None:
                require(len(inputs[1]) == len(output[1]), "Target dictionaries lost")
                for before, after in zip(inputs[1], output[1]):
                    require(torch.equal(before["boxes"], after["boxes"]), "Prepared boxes changed")

        model.transform.register_forward_hook(unchanged_geometry)
        head = model.roi_heads.box_predictor.cls_score.weight
        effective = {"normalization": "FrozenBatchNorm2d, eps=0.0; ImageNet RGB mean/std",
                     "trainable_backbone_stages": 5,
                     "reinitialized": "ROI classification and class-specific box regression predictors",
                     "image_mean": model.transform.image_mean, "image_std": model.transform.image_std}
    else:
        require(name == "yolov8n", "Unknown model")
        from ultralytics.cfg import get_cfg
        from ultralytics.nn.tasks import DetectionModel
        # Keep COCO's hidden classifier width. Building nc=4 first changes it.
        model = DetectionModel("yolov8n.yaml", ch=3, nc=80, verbose=False)
        require(model.model[-1].nc == 80 and not model.end2end, "Unexpected source YOLO head")
        transferred = []
        output_keys = {f"model.{len(model.model)-1}.cv3.{i}.2.{suffix}"
                       for i in range(3) for suffix in ("weight", "bias")}
        if pretrained_path is not None:
            # This object-format checkpoint is loaded only after the runner verifies
            # its exact official-release SHA-256. No auto-install/download loader.
            source = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            original = source.get("ema") or source["model"]
            require(isinstance(original, torch.nn.Module), "Invalid official YOLO checkpoint")
            state = original.float().state_dict()
            model.load_state_dict(state, strict=True)
            transferred = sorted(set(state) - output_keys)
            del source, original, state
        detect = model.model[-1]
        widths = []
        for branch, stride in zip(detect.cv3, detect.stride):
            previous = branch[-1]
            require(len(branch) == 3 and isinstance(previous, torch.nn.Conv2d) and previous.out_channels == 80,
                    "Unexpected YOLO output classifier")
            widths.append(previous.in_channels)
            branch[-1] = torch.nn.Conv2d(previous.in_channels, 4, 1)
            # Initialize only fresh class biases; preserve pretrained regression biases.
            torch.nn.init.constant_(branch[-1].bias, math.log(5 / 4 / (640 / float(stride)) ** 2))
        detect.nc, detect.no = 4, detect.reg_max * 4 + 4
        model.yaml["nc"] = 4
        model.names = dict(enumerate(("D00", "D10", "D20", "D40")))
        model.args = get_cfg()
        head = detect.cv3[0][-1].weight
        effective = {"normalization": "BatchNorm2d with running statistics; RGB [0,1]",
                     "yaml": model.yaml, "classifier_hidden_widths": widths,
                     "reinitialized": sorted(output_keys),
                     "loss_gains": {k: getattr(model.args, k) for k in ("box", "cls", "dfl")}}
    model.to(device)
    head = (model.roi_heads.box_predictor.cls_score.weight if name == "fasterrcnn_resnet50_fpn"
            else model.model[-1].cv3[0][-1].weight)
    effective.update(transferred_state_keys=transferred,
                     parameters=sum(p.numel() for p in model.parameters()),
                     trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model, head, effective


def infer(model, name, images, evaluation, device):
    model.eval()
    with torch.inference_mode():
        if name == "fasterrcnn_resnet50_fpn":
            result = model([image.to(device) for image in images])
            return [{"boxes": p["boxes"].cpu(), "scores": p["scores"].cpu(),
                     "labels": p["labels"].cpu() - 1} for p in result]
        from ultralytics.utils.nms import non_max_suppression
        decoded = model(torch.stack(images).to(device))[0]
        require(decoded.ndim == 3 and tuple(decoded.shape[:2]) == (len(images), 8),
                "Unexpected YOLO decoded shape")
        require(torch.isfinite(decoded).all().item(), "Nonfinite YOLO decoded predictions")
        result = non_max_suppression(decoded, conf_thres=evaluation["score_threshold"],
                                    iou_thres=evaluation["nms_iou"], nc=4,
                                    multi_label=evaluation["yolo_multi_label"],
                                    max_det=evaluation["max_detections"], max_time_img=5.0)
        return [{"boxes": p[:, :4].cpu(), "scores": p[:, 4].cpu(),
                 "labels": p[:, 5].cpu()} for p in result]
