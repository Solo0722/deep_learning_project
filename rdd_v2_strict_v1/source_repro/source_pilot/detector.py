"""Pinned Faster R-CNN checkpoint, ordinary predictions, immutable observer."""
import hashlib
import random
import time
from contextlib import contextmanager

import numpy as np

from diagnostic import file_hash, require


def tensor_digest(items):
    result = hashlib.sha256()
    for name, value in sorted(items):
        result.update(name.encode())
        if value is None:
            result.update(b"NONE")
        else:
            value = value.detach().cpu().contiguous()
            result.update(str((str(value.dtype), tuple(value.shape))).encode())
            result.update(value.numpy().tobytes())
    return result.hexdigest()


def rng_digest():
    import torch
    result = hashlib.sha256(repr((random.getstate(), np.random.get_state())).encode())
    # Include NumPy's full array, not its potentially truncated repr.
    result.update(np.random.get_state()[1].tobytes())
    result.update(torch.get_rng_state().numpy().tobytes())
    for state in torch.cuda.get_rng_state_all():
        result.update(state.cpu().numpy().tobytes())
    return result.hexdigest()


class FrozenDetector:
    def __init__(self, path, cfg):
        import torch
        from torchvision.models.detection import FasterRCNN
        from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.ops import FrozenBatchNorm2d
        expected = cfg["checkpoint"]
        require(path.name == expected["file"] and path.stat().st_size == expected["bytes"], "Wrong checkpoint file/size")
        require(file_hash(path) == expected["sha256"], "Checkpoint checksum mismatch")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        require(payload["format"] == 1 and payload["progress"]["epoch"] == 1
                and payload["progress"]["step"] == 6716, "Wrong warmup checkpoint progress")
        torch.manual_seed(cfg["seed"])
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None,
                                      norm_layer=FrozenBatchNorm2d, trainable_layers=5)
        evaluation, size = cfg["evaluation"], cfg["geometry"]["size"]
        model = FasterRCNN(backbone, num_classes=91, min_size=size, max_size=size,
                          box_score_thresh=evaluation["score_threshold"],
                          box_nms_thresh=evaluation["nms_iou"],
                          box_detections_per_img=evaluation["max_detections"])
        for module in model.modules():
            if isinstance(module, FrozenBatchNorm2d):
                module.eps = 0.
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 5)
        model.load_state_dict(payload["model"], strict=True)
        require(tensor_digest(model.state_dict().items()) == tensor_digest(payload["model"].items()), "Loaded state differs")
        self.checkpoint_model_sha256 = tensor_digest(payload["model"].items())
        del payload
        def unchanged_geometry(module, args, output):
            require(all(shape == (size, size) for shape in output[0].image_sizes), "Unexpected detector resizing")
            require(tuple(output[0].tensors.shape[-2:]) == (size, size), "Unexpected detector padding")
        model.transform.register_forward_hook(unchanged_geometry)
        self.model = model.cuda().eval().requires_grad_(False)
        self.images_forwarded = 0
        self.forward_calls = 0
        self.limit = cfg["max_detector_images"]
        self.forward_seconds = 0.
        self.audit = {}

    @contextmanager
    def frozen(self):
        import torch
        state = tensor_digest(self.model.state_dict().items())
        grads = tensor_digest((n, p.grad) for n, p in self.model.named_parameters())
        rng = rng_digest()
        require(not any(m.training for m in self.model.modules()), "Detector not fully in eval mode")
        with torch.inference_mode():
            try:
                yield self
            finally:
                require(tensor_digest(self.model.state_dict().items()) == state, "Detector state mutated")
                require(tensor_digest((n, p.grad) for n, p in self.model.named_parameters()) == grads, "Gradient mutation")
                require(rng_digest() == rng, "Global RNG changed during frozen probe")
                require(not any(m.training for m in self.model.modules()), "Model mode changed")
                self.audit = {"model_sha256_before_after": state, "gradient_sha256_before_after": grads,
                              "rng_sha256_before_after": rng, "unchanged": True}

    def predict(self, images):
        import torch
        require(self.images_forwarded + len(images) <= self.limit, "Detector forward budget exceeded")
        require(not torch.is_grad_enabled(), "Inference context required")
        require(all(im.shape == (640, 640, 3) and im.dtype == np.uint8 for im in images), "Invalid image tensor source")
        tensors = [torch.from_numpy(im.transpose(2, 0, 1).copy()).to("cuda", dtype=torch.float32)/255. for im in images]
        torch.cuda.synchronize()
        start = time.perf_counter()
        output = self.model(tensors)  # NO ground-truth argument, proposals or auxiliary loss.
        torch.cuda.synchronize()
        self.forward_seconds += time.perf_counter()-start
        self.images_forwarded += len(images)
        self.forward_calls += 1
        return [{"boxes": p["boxes"].cpu().tolist(), "scores": p["scores"].cpu().tolist(),
                 "labels": (p["labels"]-1).cpu().tolist()} for p in output]
