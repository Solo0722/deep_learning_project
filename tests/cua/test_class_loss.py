"""Per-class masked losses agree with torchvision's own loss terms (CPU, tiny untrained models)."""
import pytest
import torch

from cua.estimator.class_loss import class_losses
from cua.estimator.params import only_theta, theta_params
from cua.models import build_detector

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
NC = 3


def _batch(n_images=2):
    g = torch.Generator().manual_seed(0)
    images = [torch.rand(3, 320, 320, generator=g) for _ in range(n_images)]
    targets = [
        {"boxes": torch.tensor([[20.0, 30.0, 120.0, 150.0], [150.0, 160.0, 300.0, 300.0]]),
         "labels": torch.tensor([1, 2])},
        {"boxes": torch.tensor([[40.0, 40.0, 200.0, 220.0]]), "labels": torch.tensor([1])},
    ][:n_images]
    return images, targets


def _model(name):
    torch.manual_seed(0)
    m = build_detector(name, NC, size=320, pretrained=False)
    m.train()
    return m


def test_frcnn_box_part_matches_torchvision():
    m = _model("fasterrcnn")
    images, targets = _batch()
    torch.manual_seed(5)
    ref = m(images, targets)["loss_box_reg"]
    torch.manual_seed(5)
    cl, (cls_sums, box_sums) = class_losses(m, "fasterrcnn", images, targets, NC, parts=True)
    assert float(box_sums.sum().detach() / cl.n_norm) == pytest.approx(float(ref.detach()), rel=1e-5)
    assert torch.allclose(cl.sums, cls_sums + box_sums)


def test_retinanet_box_part_matches_torchvision():
    m = _model("retinanet")
    images, targets = _batch(n_images=1)
    ref = m(images, targets)["bbox_regression"]
    cl, (_, box_sums) = class_losses(m, "retinanet", images, targets, NC, parts=True)
    assert float(box_sums.sum().detach() / cl.n_norm) == pytest.approx(float(ref.detach()), rel=1e-5)


@pytest.mark.parametrize("name", ["fasterrcnn", "retinanet"])
def test_counts_and_absent_class(name):
    m = _model(name)
    images, targets = _batch()
    cl = class_losses(m, name, images, targets, NC)
    assert cl.sums.shape == (NC,) and cl.counts.shape == (NC,)
    assert int(cl.counts[0]) > 0 and int(cl.counts[1]) > 0
    assert int(cl.counts[2]) == 0 and float(cl.sums[2].detach()) == 0.0
    assert cl.sums.requires_grad


def test_frcnn_counts_match_sampled_labels(monkeypatch):
    m = _model("fasterrcnn")
    images, targets = _batch()
    seen = {}
    orig = m.roi_heads.select_training_samples

    def spy(proposals, tg):
        out = orig(proposals, tg)
        seen["labels"] = torch.cat(out[2])
        return out

    monkeypatch.setattr(m.roi_heads, "select_training_samples", spy)
    cl = class_losses(m, "fasterrcnn", images, targets, NC)
    for c in range(NC):
        assert int(cl.counts[c]) == int((seen["labels"] == c + 1).sum())


@pytest.mark.parametrize("name,theta,extra", [("fasterrcnn", "head", 0), ("fasterrcnn", "head_fpn", 1),
                                              ("retinanet", "head", 0), ("retinanet", "head_fpn", 1)])
def test_theta_params(name, theta, extra):
    m = _model(name)
    ps = theta_params(m, name, theta)
    fpn_top = {id(p) for mod in (m.backbone.fpn.inner_blocks[-1], m.backbone.fpn.layer_blocks[-1])
               for p in mod.parameters()}
    assert (len(fpn_top & {id(p) for p in ps}) > 0) == bool(extra)
    assert len({id(p) for p in ps}) == len(ps)


def test_only_theta_restores_flags_and_limits_grads():
    m = _model("fasterrcnn")
    ps = theta_params(m, "fasterrcnn", "head")
    before = {n: p.requires_grad for n, p in m.named_parameters()}
    images, targets = _batch()
    with only_theta(m, ps):
        loss = sum(m(images, targets).values())
        loss.backward()
    ids = {id(p) for p in ps}
    assert all((p.grad is not None) == (id(p) in ids) for p in m.parameters() if before[[n for n, q in m.named_parameters() if q is p][0]])
    assert {n: p.requires_grad for n, p in m.named_parameters()} == before
    with pytest.raises(RuntimeError):
        with only_theta(m, ps):
            raise RuntimeError("boom")
    assert {n: p.requires_grad for n, p in m.named_parameters()} == before
