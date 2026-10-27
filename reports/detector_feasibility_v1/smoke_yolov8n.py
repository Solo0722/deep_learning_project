"""Synthetic CUDA smoke for Ultralytics 8.4.182; no dataset or pretrained weights."""
import os
import sys
from pathlib import Path
from importlib.metadata import version

import torch
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.nms import non_max_suppression

expected = Path("/mmfs1/scratch/solomon.owusu.1/road_damage_augmentation_research/.venvs/rdd_detection_v1")
if Path(sys.prefix).resolve() != expected.resolve():
    raise SystemExit("STOP: rdd_detection_v1 is not active.")
if not os.environ.get("PBS_JOBID"):
    raise SystemExit("STOP: no PBS allocation.")
if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
    raise SystemExit("STOP: expected exactly one visible CUDA GPU.")
if version("ultralytics-opencv-headless") != "8.4.182":
    raise SystemExit("STOP: this check targets Ultralytics 8.4.182.")

torch.set_num_threads(4)
torch.manual_seed(42)
print("Building YOLOv8n with random weights...", flush=True)
model = DetectionModel("yolov8n.yaml", ch=3, nc=4, verbose=False).cuda()
model.args = get_cfg()
assert model.model[-1].nc == 4 and not model.end2end
images = torch.rand(2, 3, 320, 320, device="cuda")
xyxy = torch.tensor(
    [[16, 24, 90, 220], [40, 230, 260, 265],
     [100, 80, 230, 200], [240, 60, 300, 120]],
    dtype=torch.float32, device="cuda",
)
xywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2,
                  xyxy[:, 2:] - xyxy[:, :2]), dim=1) / 320
batch = {
    "img": images,
    "batch_idx": torch.zeros(4, device="cuda"),
    "cls": torch.arange(4, dtype=torch.float32, device="cuda").view(-1, 1),
    "bboxes": xywh,
}

model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
optimizer.zero_grad(set_to_none=True)
loss_vector, loss_items = model(batch)
assert torch.isfinite(loss_vector).all().item()
assert isinstance(loss_items, dict) and loss_items
assert all(torch.isfinite(v).all().item() for v in loss_items.values())
total_loss = loss_vector.sum()
assert torch.isfinite(total_loss).item()
total_loss.backward()
for name, parameter in model.named_parameters():
    if parameter.grad is not None:
        assert torch.isfinite(parameter.grad).all().item(), name
head = model.model[-1].cv3[0][-1].weight
assert head.grad is not None and head.grad.abs().sum().item() > 0
before = head.detach().clone()
optimizer.step()
assert not torch.equal(before, head.detach()), "Classifier weights did not update"
print("Losses:", {k: round(v.item(), 6) for k, v in loss_items.items()})
print("Forward, backward, and optimizer step: PASS", flush=True)

model.eval()
with torch.inference_mode():
    decoded = model(images)[0]
    assert decoded.ndim == 3 and decoded.shape[:2] == (2, 8)
    assert torch.isfinite(decoded).all().item()
    assert (decoded[:, 2:4] > 0).all().item()
    assert ((decoded[:, 4:] >= 0) & (decoded[:, 4:] <= 1)).all().item()
    # Zero confidence threshold exercises NMS on untrained predictions.
    predictions = non_max_suppression(
        decoded, conf_thres=0.0, iou_thres=0.5, nc=4,
        max_det=100, max_time_img=5.0,
    )
assert len(predictions) == 2
for prediction in predictions:
    assert prediction.ndim == 2 and prediction.shape[1] == 6
    assert 0 < len(prediction) <= 100
    assert torch.isfinite(prediction).all().item()
    boxes, scores, labels = prediction[:, :4], prediction[:, 4], prediction[:, 5]
    assert (boxes[:, 2:] > boxes[:, :2]).all().item()
    assert ((scores >= 0) & (scores <= 1)).all().item()
    assert ((labels >= 0) & (labels < 4) & (labels == labels.round())).all().item()
torch.cuda.synchronize()
print("Inference and NMS: PASS")
print("YOLOv8n SYNTHETIC SMOKE: PASS")
