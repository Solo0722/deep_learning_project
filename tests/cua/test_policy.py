import numpy as np
import pytest

from cua.policy import IdentityPolicy, UniformPolicy, UtilityPolicy, UtilityTable, choose

MENU = ("a", "b")


def table(mean, se=None, count=5):
    mean = np.asarray(mean, dtype=np.float64)
    se = np.zeros_like(mean) if se is None else np.asarray(se, dtype=np.float64)
    return UtilityTable(mean=mean, se=se, count=np.full(mean.shape, count, dtype=np.int64))


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def probs_dict(policy, s):
    cands, p = policy.probs(frozenset(s))
    assert p.sum() == pytest.approx(1.0)
    return dict(zip(cands, p))


@pytest.mark.parametrize("shared", [False, True])
def test_uniform_before_any_estimate(shared):
    cols = 1 if shared else 2
    t = table(np.zeros((2, cols)), count=0)
    pol = UtilityPolicy(MENU, lambda: t, shared=shared)
    d = probs_dict(pol, {0, 1})
    assert set(d) == {None, "a", "b"}
    assert all(v == pytest.approx(1 / 3) for v in d.values())


def test_identity_scored_zero_and_never_blocked():
    t = table([[-5.0, -5.0], [-5.0, -5.0]], se=[[0.1, 0.1], [0.1, 0.1]])
    d = probs_dict(UtilityPolicy(MENU, lambda: t, shared=False), {0, 1})
    assert d == {None: pytest.approx(1.0)}


def test_score_softmax_epsilon():
    t = table([[1.0, 3.0], [-1.0, -1.0]])
    d = probs_dict(UtilityPolicy(MENU, lambda: t, shared=False, epsilon=0.1, harm_block=False), {0, 1})
    scores = np.array([0.0, 2.0, -1.0])
    expected = 0.9 * softmax(scores / np.std(scores)) + 0.1 / 3
    assert [d[None], d["a"], d["b"]] == pytest.approx(expected.tolist())


def test_harm_block_uses_any_present_class():
    t = table([[1.0, -0.5], [0.2, 0.2]], se=[[0.0, 0.1], [0.0, 0.0]], count=3)
    pol = UtilityPolicy(MENU, lambda: t, shared=False)
    assert "a" not in probs_dict(pol, {0, 1})
    assert "a" in probs_dict(pol, {0})
    t2 = table([[1.0, -0.5], [0.2, 0.2]], se=[[0.0, 0.1], [0.0, 0.0]], count=2)
    assert "a" in probs_dict(UtilityPolicy(MENU, lambda: t2, shared=False), {0, 1})


def test_all_blocked_without_identity_falls_back():
    t = table([[-1.0, -1.0], [-1.0, -1.0]])
    pol = UtilityPolicy(MENU, lambda: t, shared=False, allow_identity=False)
    cands, p = pol.probs(frozenset({0}))
    assert cands == (None,) and p.tolist() == [1.0]
    assert pol.fallbacks == 1


def test_no_identity_option_when_disabled():
    t = table([[1.0, 1.0], [0.5, 0.5]])
    d = probs_dict(UtilityPolicy(MENU, lambda: t, shared=False, allow_identity=False), {0})
    assert set(d) == {"a", "b"}


def test_shared_ignores_class_set():
    t = table([[1.0], [-0.2]])
    pol = UtilityPolicy(MENU, lambda: t, shared=True, harm_block=False)
    assert probs_dict(pol, {0}) == probs_dict(pol, {1})


def test_class_policy_depends_on_class_set():
    t = table([[1.0, -1.0], [-1.0, 1.0]])
    pol = UtilityPolicy(MENU, lambda: t, shared=False, harm_block=False)
    assert probs_dict(pol, {0})["a"] > probs_dict(pol, {0})["b"]
    assert probs_dict(pol, {1})["a"] < probs_dict(pol, {1})["b"]


def test_inv_ap_weights_change_score():
    t = table([[1.0, -3.0], [0.0, 0.0]])
    equal = probs_dict(UtilityPolicy(MENU, lambda: t, shared=False, harm_block=False), {0, 1})
    weighted = probs_dict(UtilityPolicy(MENU, lambda: t, shared=False, harm_block=False,
                                        weights_fn=lambda: np.array([2.0, 0.0])), {0, 1})
    assert equal["a"] < equal[None]
    assert weighted["a"] > weighted[None]


def test_tau_floor_with_equal_scores():
    t = table([[0.0, 0.0], [0.0, 0.0]])
    d = probs_dict(UtilityPolicy(MENU, lambda: t, shared=False), {0, 1})
    assert all(np.isfinite(v) and v == pytest.approx(1 / 3) for v in d.values())


def test_no_object_image():
    rng = np.random.default_rng(0)
    pol = UniformPolicy(MENU)
    picks = [choose(pol, frozenset(), rng, MENU) for _ in range(10000)]
    assert picks.count(None) / len(picks) == pytest.approx(0.5, abs=0.02)
    assert picks.count("a") / len(picks) == pytest.approx(0.25, abs=0.02)


def test_uniform_and_identity_policies():
    assert probs_dict(UniformPolicy(MENU), {0}) == {None: pytest.approx(1 / 3), "a": pytest.approx(1 / 3),
                                                     "b": pytest.approx(1 / 3)}
    assert probs_dict(IdentityPolicy(), {0}) == {None: 1.0}
    rng = np.random.default_rng(1)
    assert {choose(IdentityPolicy(), frozenset({0}), rng, MENU) for _ in range(50)} == {None}
