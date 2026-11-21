"""CUA v2: lookahead estimator (fixes 1, 3), |t| gate (fix 2), probe prior (fix 4), specs and trainer."""
import json

import numpy as np
import pytest
import torch

import cua.train as T
from cua.estimator.lookahead import LookaheadEstimator
from cua.experiments.specs import V2_SETTINGS, part_c_specs, resolve
from cua.models import build_detector
from cua.policy import UtilityPolicy, UtilityTable, combine_with_prior, shared_prior

pytestmark = pytest.mark.filterwarnings("ignore:Changing trainable_backbone_layers")
NC = 3


def table(mean, se, count=5):
    mean = np.asarray(mean, float)
    return UtilityTable(mean=mean, se=np.asarray(se, float), count=np.full(mean.shape, count, dtype=np.int64))


# ---- fix 2: |t| gate ----

def test_min_t_gate_zeroes_uncertain_cells():
    t = table([[1.0], [0.5]], [[0.1], [1.0]])  # edit a sure (t=10), edit b not (t=0.5)
    gated = UtilityPolicy(("a", "b"), lambda: t, shared=True, min_t=2.0, harm_block=False, epsilon=0.0)
    plain = UtilityPolicy(("a", "b"), lambda: t, shared=True, harm_block=False, epsilon=0.0)
    pg = dict(zip(*gated.probs(frozenset({0}))))
    pp = dict(zip(*plain.probs(frozenset({0}))))
    assert pg["b"] == pytest.approx(pg[None])  # b scored 0 like identity
    assert pp["b"] > pp[None]


# ---- fix 4: prior ----

def test_combine_uses_prior_alone_until_online_has_estimates():
    prior = table([[2.0, -4.0]], [[1.0, 1.0]], count=3)
    online = table([[0.0, 0.0]], [[0.0, 0.0]], count=0)
    c = combine_with_prior(online, prior)
    assert np.allclose(c.mean, [[2.0, -4.0]]) and np.allclose(c.se, 1.0) and (c.count >= 3).all()


def test_combine_is_stouffer_on_the_z_scale():
    prior = table([[2.0]], [[1.0]], count=3)
    online = table([[0.03]], [[0.01]], count=10)  # z = 3, in other units
    c = combine_with_prior(online, prior)
    assert c.mean[0, 0] == pytest.approx((2.0 + 3.0) / np.sqrt(2))


def test_prior_harm_blocks_before_any_online_estimate():
    prior = table([[-5.0, 1.0], [1.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]], count=3)
    online = table(np.zeros((2, 2)), np.zeros((2, 2)), count=0)
    pol = UtilityPolicy(("a", "b"), lambda: combine_with_prior(online, prior), shared=False)
    cands, _ = pol.probs(frozenset({0}))
    assert "a" not in cands and "b" in cands


def test_shared_prior_means_over_classes():
    sp = shared_prior(table([[1.0, 3.0]], [[3.0, 4.0]], count=3))
    assert sp.mean[0, 0] == pytest.approx(2.0) and sp.se[0, 0] == pytest.approx(2.5)


# ---- fixes 1, 3: lookahead estimator on a tiny CPU detector ----

def _renderers(n=8):
    g = torch.Generator().manual_seed(0)
    imgs = [torch.rand(3, 320, 320, generator=g) for _ in range(n)]
    tg = [{"boxes": torch.tensor([[20.0, 30.0, 140.0, 160.0]]), "labels": torch.tensor([1 + i % NC])}
          for i in range(n)]

    def train_render(i, edit):
        im, t = imgs[i], tg[i]
        if edit == "brightness":
            im = (im * 1.3).clamp(0, 1)
        if edit == "many":  # stands in for mosaic: same image, four copies of the object
            t = {"boxes": torch.tensor([[20.0, 30.0, 140.0, 160.0], [160, 30, 300, 160],
                                        [20, 170, 140, 300], [160, 170, 300, 300]]),
                 "labels": t["labels"].repeat(4)}
        return im, t

    return train_render, lambda i: (imgs[i], tg[i])


def _est(menu, **kw):
    torch.manual_seed(0)
    m = build_detector("fasterrcnn", NC, size=320, pretrained=False)
    m.train()
    kw = {"est_batch": 4, "steps": 2, "ref_batch": 3, **kw}
    return LookaheadEstimator(m, "fasterrcnn", "head", menu, NC, device=torch.device("cpu"), **kw)


@pytest.mark.slow
def test_lookahead_shapes_restores_weights_and_leaves_grads(mk):
    tr, rr = _renderers()
    est = _est((None, "brightness"))
    before = [p.detach().clone() for p in est.model.parameters()]
    refs = [mk(image=f"img/{i}.jpg", labels=[i % NC]) for i in range(8)]
    out = est.estimate(tr, rr, 8, refs, np.random.default_rng(0), lr=0.01, epoch=2, global_iter=400)
    assert out["u"].shape == (2, NC) and np.isfinite(out["u"]).all()
    assert np.all(out["u"][0] == 0.0)  # identity row
    assert np.any(out["u"][1] != 0.0)
    assert all(torch.equal(a, b) for a, b in zip(before, est.model.parameters()))
    assert all(p.grad is None for p in est.model.parameters())
    assert est.cua.table().count.tolist() == [[1] * NC] * 2 and est.shared.table().count.tolist() == [[1], [1]]


@pytest.mark.slow
@pytest.mark.parametrize("clip", [0.0, 1e-3])
def test_instance_norm_scales_the_edited_step_by_object_ratio(clip):
    """'many' has 4x the objects of the clean images, so its step is 1/4 of the unscaled one,
    also when the clip is active (scaling is applied after the clip)."""
    from cua.estimator.params import only_theta, theta_params
    tr, _ = _renderers(4)
    items = [tr(i, "many") for i in range(4)]
    clean = [tr(i, None) for i in range(4)]
    deltas = {}
    for scale in (False, True):
        est = _est(("many",), clip=clip, steps=1)
        params = theta_params(est.model, "fasterrcnn", "head")
        before = [p.detach().clone() for p in params]
        with only_theta(est.model, params):
            est._steps(items, clean, params, step_lr=1.0 if clip else 1e-3, seed=0, scale_objects=scale)
        deltas[scale] = torch.cat([(p.detach() - b).reshape(-1) for p, b in zip(params, before)])
    assert (deltas[True] - 0.25 * deltas[False]).abs().max() < 1e-2 * (0.25 * deltas[False]).abs().max()


def test_lookahead_validates_steps():
    with pytest.raises(ValueError):
        LookaheadEstimator(torch.nn.Linear(1, 1), "fasterrcnn", "head", ("a",), 2, est_batch=2, steps=4)


# ---- specs, config ----

def test_v2_specs_and_resolve():
    specs = part_c_specs("rdd", "fasterrcnn", seeds=(0,), policies=("shared2", "cua2"))
    assert [s.run_id for s in specs] == ["C-rdd-fasterrcnn-s0-shared2", "C-rdd-fasterrcnn-s0-cua2"]
    assert [s.policy for s in specs] == ["shared", "cua"]
    for s in specs:
        assert all(getattr(s, k) == v for k, v in V2_SETTINGS.items())
        assert resolve(s.run_id) == s


def test_v1_digests_unchanged_by_new_fields():
    v1 = part_c_specs("rdd", "fasterrcnn", seeds=(0,), policies=("cua",))[0]
    d = json.loads(json.dumps(v1.__dict__, default=list))
    assert "estimator" in d  # field exists ...
    import dataclasses, hashlib
    a = dataclasses.asdict(v1)
    for f in dataclasses.fields(v1):
        if f.name in T._PLAN2_FIELDS and a[f.name] == f.default:
            del a[f.name]
    assert "estimator" not in a and "prior" not in a and "min_t" not in a  # ... but not in the digest
    assert v1.digest() == hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()


def test_validate_rejects_bad_v2_values():
    base = part_c_specs("rdd", "fasterrcnn", seeds=(0,), policies=("cua2",))[0]
    for bad in (dict(estimator="x"), dict(prior="x"), dict(min_t=-1.0), dict(lookahead_steps=0)):
        with pytest.raises(ValueError):
            T._validate(dataclasses_replace(base, **bad))
    with pytest.raises(ValueError):
        T._validate(dataclasses_replace(base, policy="uniform"))  # prior needs shared/cua


def dataclasses_replace(c, **kw):
    import dataclasses
    return dataclasses.replace(c, **kw)


# ---- probe table ----

def test_epoch_deltas_reads_the_first_arm_epoch(tmp_path):
    from cua.measured import epoch_deltas

    def run(name, per_epoch):
        d = tmp_path / name
        d.mkdir()
        (d / "COMPLETE.json").write_text("{}")
        rows = [{"epoch": e, "metrics": {"per_class": {"X": {"ap50": v}}}} for e, v in per_epoch.items()]
        (d / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    run("A-rdd-fasterrcnn-s0-base", {4: 10.0, 6: 20.0})
    run("A-rdd-fasterrcnn-s0-blur", {4: 13.0, 6: 18.0})
    run("A-rdd-fasterrcnn-s0-warm", {1: 1.0})
    rows = epoch_deltas(tmp_path, 4)
    assert [(r["edit"], r["cls"], r["delta"]) for r in rows] == [("blur", "X", 3.0)]
    assert [r["delta"] for r in epoch_deltas(tmp_path, 6)] == [-2.0]


@pytest.mark.slow
def test_clip_keeps_a_huge_step_finite_and_nonfinite_rounds_are_skipped(mk, monkeypatch):
    tr, rr = _renderers()
    refs = [mk(image=f"img/{i}.jpg", labels=[i % NC]) for i in range(8)]
    est = _est((None, "brightness"), lr_scale=1e4, clip=1.0)
    out = est.estimate(tr, rr, 8, refs, np.random.default_rng(0), lr=0.01, epoch=2, global_iter=400)
    assert np.isfinite(out["u"]).all() and not out.get("skipped")
    bad = _est((None, "brightness"))
    monkeypatch.setattr(bad, "_ref_losses", lambda refs, seed: (torch.full((NC,), float("nan"), dtype=torch.float64),
                                                                torch.ones(NC, dtype=torch.bool)))
    res = bad.estimate(tr, rr, 8, refs, np.random.default_rng(0), lr=0.01, epoch=2, global_iter=400)
    assert res["skipped"] and bad.skipped == 1 and bad.cua.table().count.sum() == 0
