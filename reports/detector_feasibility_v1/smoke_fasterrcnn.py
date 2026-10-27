"""Synthetic CUDA smoke check only; no dataset use or pretrained downloads."""
import os
import sys
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn

expected = Path("/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/.venvs/rdd_detection_v1")
if Path(sys.prefix).resolve() != expected.resolve():
    raise SystemExit("STOP: rdd_detection_v1 is not active.")
if not os.environ.get("PBS_JOBID"):
    raise SystemExit("STOP: no PBS allocation.")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("STOP: expected exactly one visible CUDA GPU.")

torch.set_num_threads(4)
torch.manual_seed(42)
print("Building Faster R-CNN with random weights...", flush=True)
model = fasterrcnn_resnet50_fpn(
    weights=None, weights_backbone=None, num_classes=5,
    min_size=320, max_size=320,
).cuda()
images = [torch.rand(3, 320, 320, device="cuda") for _ in range(2)]
targets = [
    {
        "boxes": torch.tensor(
            [[16, 24, 90, 220], [40, 230, 260, 265],
             [100, 80, 230, 200], [240, 60, 300, 120]],
            dtype=torch.float32, device="cuda",
        ),
        "labels": torch.tensor([1, 2, 3, 4], device="cuda"),
    },
    {
        "boxes": torch.empty((0, 4), device="cuda"),
        "labels": torch.empty((0,), dtype=torch.int64, device="cuda"),
    },
]

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer.zero_grad(set_to_none=True)
losses = model(images, targets)
assert losses and all(torch.isfinite(v).all().item() for v in losses.values())
total_loss = sum(losses.values())
assert torch.isfinite(total_loss).item()
total_loss.backward()
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        assert torch.isfinite(parameter.grad).all().item(), name
head = model.roi_heads.box_predictor.cls_score.weight
assert head.grad is not None and head.grad.abs().sum().item() > 0
before = head.detach().clone()
optimizer.step()
assert not torch.equal(before, head.detach()), "Classifier weights did not update"
print("Losses:", {k: round(v.item(), 6) for k, v in losses.items()})
print("Forward, backward, and optimizer step: PASS", flush=True)

model.eval()
with torch.inference_mode():
    predictions = model(images)
assert len(predictions) == 2
for prediction in predictions:
    boxes, scores, labels = (prediction[k] for k in ("boxes", "scores", "labels"))
    assert boxes.shape == (len(scores), 4) and labels.shape == scores.shape
    assert torch.isfinite(boxes).all().item()
    assert torch.isfinite(scores).all().item()
    assert ((scores >= 0) & (scores <= 1)).all().item()
    assert ((labels >= 1) & (labels <= 4)).all().item()
    assert (boxes[:, 2:] > boxes[:, :2]).all().item()
    assert ((boxes >= 0) & (boxes <= 320)).all().item()
torch.cuda.synchronize()
print("Inference: PASS")
print("FASTER R-CNN SYNTHETIC SMOKE: PASS")
