from pathlib import Path

import pytest

from cua.datasets.rdd import RDD_CLASSES, load_rdd
from cua.gates import cooccurrence_rate

FROZEN_SPLITS = Path(__file__).parents[2] / "detection_splits_v1"


@pytest.mark.skipif(not FROZEN_SPLITS.exists(), reason="frozen RDD splits not present")
def test_rdd_counts_match_frozen_split():
    s = load_rdd(FROZEN_SPLITS)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (26863, 5760, 5760)
    assert round(cooccurrence_rate(s["train"]), 3) == 0.349
    assert RDD_CLASSES == ("D00", "D10", "D20", "D40")
