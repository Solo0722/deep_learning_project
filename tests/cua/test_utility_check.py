import json

import numpy as np
import torch

from cua.analysis import utility_check as UC
from cua.estimator.utility import utility_from_grads


def test_dot_matches_the_estimator_up_to_lr():
    g = torch.Generator().manual_seed(0)
    g0, go, h, lc = torch.randn(7, generator=g), torch.randn(3, 7, generator=g), torch.randn(2, 7, generator=g), torch.rand(2, generator=g)
    v = UC.variants(g0, go, h, lc)
    assert np.allclose(v["dot"] * 0.01, utility_from_grads(g0, go, h, lc, lr=0.01))


def test_scaling_an_edit_gradient_changes_dot_but_not_normed_or_cosdiff():
    g = torch.Generator().manual_seed(1)
    g0, h, lc = torch.randn(5, generator=g), torch.randn(2, 5, generator=g), torch.rand(2, generator=g)
    go = torch.stack([g0 * 1.0, g0 * 3.0])  # same direction as g0, edit 2 three times larger
    v = UC.variants(g0, go, h, lc)
    assert np.allclose(v["normed"], 0.0, atol=1e-6) and np.allclose(v["cosdiff"], 0.0, atol=1e-6)
    assert np.allclose(v["dot"][0], 0.0, atol=1e-6) and np.abs(v["dot"][1]).min() > 0.1
    assert np.allclose(v["ratio"][:, 0], [1.0, 3.0], atol=1e-6)


def test_summarize_one_correlates_with_part_a(tmp_path):
    menu, classes = ["a", "b", "c"], ["X", "Y"]
    pa = {("a", "X"): 1.0, ("a", "Y"): 2.0, ("b", "X"): 3.0, ("b", "Y"): 4.0, ("c", "X"): 5.0, ("c", "Y"): 6.0}
    good = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    rows = [{"valid": [True, True], "dot": good, "normed": good, "cosdiff": good,
             "ratio": [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]}] * 2
    r = UC.summarize_one({"run": "r", "ckpt": "c", "menu": menu, "classes": classes, "rows": rows}, pa)
    assert r["n_cells"] == 6 and abs(r["all_cells"]["dot"] - 1.0) < 1e-9
    assert abs(r["within_edit"]["cosdiff"] - 1.0) < 1e-9 and abs(r["edit_level"]["ratio"] - 1.0) < 1e-9
    p = tmp_path / "d.csv"
    p.write_text("dataset,detector,seed,edit,cls,delta\nrdd,frcnn,0,a,X,1\nrdd,frcnn,1,a,X,3\nvoc,frcnn,0,a,X,9\n")
    assert UC.part_a_means(p, "rdd", "frcnn") == {("a", "X"): 2.0}
