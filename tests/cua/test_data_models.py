
import numpy as np
import pytest
import torch
from PIL import Image

from cua.augment.menu import ArmTransform
from cua.data import DetectionDataset, collate
from cua.models import build_detector, trainable_head_parameters
from cua.records import Record

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")


def _make(tmp_path, n=3, w=80, h=60, boxes=None, labels=None, ignore=None, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for k in range(n):
        name = f"im{k}.png"
        Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)).save(tmp_path / name)
        b = np.array([[10., 10., 40., 30.]]) if boxes is None else np.asarray(boxes, dtype=np.float64)
        l = np.array([2]) if labels is None else np.asarray(labels)
        ig = np.zeros(len(l), dtype=bool) if ignore is None else np.asarray(ignore)
        recs.append(Record(name, w, h, b.reshape(-1, 4), l, ig, f"g{k}", "s"))
    return recs


def test_empty_target_tensor_shapes(tmp_path):
    recs = _make(tmp_path, n=1, boxes=np.zeros((0, 4)), labels=np.zeros(0, dtype=np.int64), ignore=np.zeros(0, bool))
    img, t = DetectionDataset(recs, tmp_path, transform=None, size=64)[0]
    assert img.shape == (3, 64, 64) and img.dtype == torch.float32 and img.is_contiguous()
    assert t["boxes"].shape == (0, 4) and t["boxes"].dtype == torch.float32
    assert t["labels"].shape == (0,) and t["labels"].dtype == torch.int64
    assert t["ignore"].shape == (0, 4)


def test_ignored_box_excluded_and_labels_one_based(tmp_path):
    recs = _make(tmp_path, n=1, boxes=[[10, 10, 40, 30], [20, 20, 50, 40]], labels=[0, 3], ignore=[False, True])
    _, t = DetectionDataset(recs, tmp_path, transform=None, size=64)[0]
    assert t["boxes"].shape == (1, 4) and t["labels"].tolist() == [1]
    assert t["ignore"].shape == (1, 4) and t["ignore"].dtype == torch.float32
    assert not torch.equal(t["ignore"], t["boxes"])
    assert t["image_id"] == 0 and isinstance(t["image_id"], int)


def test_geometry_only_without_transform(tmp_path):
    recs = _make(tmp_path)
    _, t = DetectionDataset(recs, tmp_path, transform=None, size=64)[0]
    assert t["geometry"]["size"] == 64
    tr = ArmTransform("rdd", None, seed=0, p=1.0)
    _, t2 = DetectionDataset(recs, tmp_path, transform=tr, size=640)[0]
    assert "geometry" not in t2


def test_mosaic_runs_and_boxes_inside_canvas(tmp_path):
    recs = _make(tmp_path)
    ds = DetectionDataset(recs, tmp_path, ArmTransform("rdd", "mosaic", seed=0, p=1.0), size=640)
    for i in range(3):
        img, t = ds[i]
        assert img.shape == (3, 640, 640) and 0 <= float(img.min()) and float(img.max()) <= 1
        b = t["boxes"]
        assert b.shape[1] == 4 and len(b) == len(t["labels"])
        if len(b):
            assert (b[:, 2] > b[:, 0]).all() and (b[:, 3] > b[:, 1]).all()
            assert (b >= 0).all() and (b <= 640).all()


def test_set_epoch_changes_output(tmp_path):
    recs = _make(tmp_path)
    ds = DetectionDataset(recs, tmp_path, ArmTransform("rdd", "noise", seed=0, p=1.0), size=640)
    a, _ = ds[0]
    ds.set_epoch(1)
    b, _ = ds[0]
    ds.set_epoch(0)
    c, _ = ds[0]
    assert not torch.equal(a, b) and torch.equal(a, c)


def test_transform_requires_640(tmp_path):
    with pytest.raises(ValueError, match="640"):
        DetectionDataset(_make(tmp_path), tmp_path, ArmTransform("rdd", None, seed=0), size=64)


def test_collate(tmp_path):
    ds = DetectionDataset(_make(tmp_path), tmp_path, None, size=64)
    imgs, tgts = collate([ds[0], ds[1]])
    assert isinstance(imgs, list) and len(imgs) == 2 and isinstance(tgts, list) and len(tgts) == 2


@pytest.mark.parametrize("name", ["fasterrcnn", "retinanet"])
def test_detector_trains_one_step_on_cpu(name):
    m = build_detector(name, num_classes=4, size=320, pretrained=False)
    images = [torch.rand(3, 320, 320), torch.rand(3, 320, 320)]
    targets = [{"boxes": torch.tensor([[10., 10., 100., 100.]]), "labels": torch.tensor([3])},
               {"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)}]
    loss = sum(m(images, targets).values()); loss.backward()
    assert torch.isfinite(loss)
    empty = [{"boxes": torch.zeros(0, 4), "labels": torch.zeros(0, dtype=torch.int64)} for _ in range(2)]
    loss = sum(m(images, empty).values()); loss.backward()
    assert torch.isfinite(loss)


@pytest.mark.parametrize("name", ["fasterrcnn", "retinanet"])
def test_eval_forward_and_transform_no_rescale(name):
    m = build_detector(name, num_classes=4, size=320, pretrained=False).eval()
    x = [torch.rand(3, 320, 320)]
    il, _ = m.transform(x)
    assert tuple(il.tensors.shape[-2:]) == (320, 320) and il.image_sizes == [(320, 320)]
    with torch.no_grad():
        out = m(x)
    assert set(out[0]) >= {"boxes", "scores", "labels"}


@pytest.mark.parametrize("name", ["fasterrcnn", "retinanet"])
def test_trainable_head_parameters(name):
    m = build_detector(name, num_classes=4, size=320, pretrained=False)
    ps = trainable_head_parameters(m, name)
    assert ps and all(p.requires_grad for p in ps)
    backbone = {id(p) for p in m.backbone.parameters()}
    assert not any(id(p) in backbone for p in ps)
    with pytest.raises(ValueError):
        trainable_head_parameters(m, "nope")


def test_retinanet_never_emits_label_zero_and_class0_gets_no_grad():
    torch.manual_seed(0)
    m = build_detector("retinanet", num_classes=4, size=320, pretrained=False)
    m.eval()
    with torch.no_grad():
        out = m([torch.rand(3, 320, 320), torch.rand(3, 320, 320)])
    assert sum(len(o["labels"]) for o in out) > 0
    assert all((o["labels"] != 0).all() for o in out)
    m.train()
    images = [torch.rand(3, 320, 320)]
    feats = m.backbone(m.transform(images)[0].tensors)
    logits = m.head.classification_head(list(feats.values()))
    assert (logits[..., 0] == -1e4).all()
    logits[..., 1:].sum().backward()
    head = m.head.classification_head
    # class-0 rows of the final conv (anchor-major layout: index a*K + 0) get zero gradient
    w = head.cls_logits.weight.grad.view(-1, 5, *head.cls_logits.weight.shape[1:])
    assert (w[:, 0] == 0).all() and (w[:, 1:] != 0).any()
