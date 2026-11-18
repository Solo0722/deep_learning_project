import math

import numpy as np
import pytest
import torch

from cua.estimator import utility as U
from cua.estimator.utility import EmaTracker, Estimator, stratified_reference, utility_from_grads
from cua.models import build_detector

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
NC = 3


def test_utility_from_grads_exact():
    g0 = torch.tensor([1.0, 0.0])
    go = torch.tensor([[2.0, 1.0], [1.0, 0.0]])
    h = torch.tensor([[1.0, 1.0], [0.0, 2.0]])
    lc = torch.tensor([0.99, 1.99])
    u = utility_from_grads(g0, go, h, lc, lr=0.5, delta=0.01)
    # (go-g0) = [[1,1],[0,0]]; dot with h -> [[2,2],[0,0]]; *0.5 / (Lc+0.01) -> [[1,0.5],[0,0]]
    assert u.ravel().tolist() == pytest.approx([1.0, 0.5, 0.0, 0.0])


def test_ema_and_se():
    t = EmaTracker((1, 2), beta=0.9)
    t.update(np.array([[1.0, 7.0]]), np.array([[True, False]]))
    t.update(np.array([[3.0, 7.0]]), np.array([[True, False]]))
    tab = t.table()
    assert tab.mean[0, 0] == pytest.approx(1.2)
    assert tab.se[0, 0] == pytest.approx(math.sqrt(0.36 * 0.1 / 1.9))
    assert tab.count.tolist() == [[2, 0]]
    assert tab.mean[0, 1] == 0.0 and tab.se[0, 1] == 0.0


def test_state_roundtrip():
    t = EmaTracker((2, 2), beta=0.9)
    t.update(np.arange(4.0).reshape(2, 2), np.ones((2, 2), bool))
    t.update(np.ones((2, 2)), np.ones((2, 2), bool))
    t2 = EmaTracker((2, 2), beta=0.9)
    t2.load_state_dict(t.state_dict())
    a, b = t.table(), t2.table()
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.se, b.se) and np.array_equal(a.count, b.count)


def test_stratified_reference_covers_all_classes(mk):
    recs = [mk(image=f"img/{i}.jpg", labels=[i % 5]) for i in range(40)] + [mk(image="img/x.jpg", labels=[4, 0])]
    idx = stratified_reference(recs, 8, np.random.default_rng(0))
    assert len(idx) == 8 and len(set(idx)) == 8
    assert set().union(*(recs[i].class_set() for i in idx)) == set(range(5))
    assert idx == stratified_reference(recs, 8, np.random.default_rng(0))
    with pytest.raises(ValueError):
        stratified_reference(recs, 3, np.random.default_rng(0))


# ---- estimator on a tiny CPU detector ----

def _renderers():
    g = torch.Generator().manual_seed(0)
    imgs = [torch.rand(3, 320, 320, generator=g) for _ in range(6)]
    tg = [{"boxes": torch.tensor([[20.0, 30.0, 140.0, 160.0]]), "labels": torch.tensor([1 + i % NC])}
          for i in range(6)]

    def train_render(i, edit):
        im = imgs[i]
        if edit == "brightness":
            im = (im * 1.3).clamp(0, 1)
        return im, tg[i]

    def ref_render(i):
        return imgs[i], tg[i]

    return train_render, ref_render


def _estimator(menu):
    torch.manual_seed(0)
    m = build_detector("fasterrcnn", NC, size=320, pretrained=False)
    m.train()
    return Estimator(m, "fasterrcnn", "head", menu, NC, est_batch=2, ref_batch=3, amp=False,
                     device=torch.device("cpu"))


def _ref_records(mk):
    return [mk(image=f"img/{i}.jpg", labels=[i % NC]) for i in range(6)]


def test_identity_edit_row_is_zero(mk):
    tr, rr = _renderers()
    est = _estimator((None, "brightness"))
    out = est.estimate(tr, rr, 6, _ref_records(mk), np.random.default_rng(0), lr=0.01, epoch=2, global_iter=200)
    assert np.all(out["u"][0][out["valid"][0]] == 0.0)
    assert out["u"].shape == (2, NC)


def test_estimate_shapes_finite_and_updates_trackers(mk):
    tr, rr = _renderers()
    est = _estimator(("brightness",))
    params_before = [p.detach().clone() for p in est.model.parameters()]
    out = est.estimate(tr, rr, 6, _ref_records(mk), np.random.default_rng(1), lr=0.01, epoch=2, global_iter=200)
    assert out["u"].shape == (1, NC) and np.isfinite(out["u"]).all()
    assert out["valid"].all() and out["seconds"] > 0
    assert out["u_shared"].shape == (1, 1)
    assert est.cua.table().count.tolist() == [[1, 1, 1]]
    assert est.shared.table().count.tolist() == [[1]]
    assert all(torch.equal(a, b) for a, b in zip(params_before, est.model.parameters()))
    assert all(p.grad is None for p in est.model.parameters())


def test_nonfinite_raises(mk, monkeypatch):
    tr, rr = _renderers()
    est = _estimator(("brightness",))
    monkeypatch.setattr(U, "utility_from_grads", lambda *a, **k: np.full((1, NC), np.nan))
    with pytest.raises(RuntimeError, match="global iteration 200"):
        est.estimate(tr, rr, 6, _ref_records(mk), np.random.default_rng(1), lr=0.01, epoch=2, global_iter=200)


def test_estimator_runs_in_fp32_even_when_training_uses_amp(mk):
    """On CCAST (A10, fp16 autocast) summed class losses overflowed in the backward pass and every
    utility of three RDD classes came out NaN/inf. The estimator's passes must ignore AMP."""
    tr, rr = _renderers()
    a = _estimator(("brightness",))
    out_fp32 = a.estimate(tr, rr, 6, _ref_records(mk), np.random.default_rng(1), lr=0.01, epoch=2, global_iter=200)
    b = _estimator(("brightness",))
    b.amp = True
    out_amp = b.estimate(tr, rr, 6, _ref_records(mk), np.random.default_rng(1), lr=0.01, epoch=2, global_iter=200)
    assert np.array_equal(out_fp32["u"], out_amp["u"])
