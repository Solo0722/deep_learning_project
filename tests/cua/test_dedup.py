import numpy as np
import pytest

import cua.dedup as dedup
from cua.dedup import cross_split_duplicates, dhash, duplicate_groups


def test_dhash_is_64_bit_int(noise_image):
    h = dhash(noise_image(seed=0))
    assert isinstance(h, int) and 0 <= h < 2**64


def test_dhash_groups_near_duplicates(smooth_image, brighten):
    img = smooth_image(seed=0)
    shifted = brighten(img, 1.05)
    assert bin(dhash(img) ^ dhash(shifted)).count("1") < 4  # comfortably inside max_distance
    g = duplicate_groups([dhash(img), dhash(shifted), dhash(smooth_image(seed=9))])
    assert g[0] == g[1] != g[2]


def test_hash_records_names_unreadable_image(tmp_path, mk):
    (tmp_path / "img").mkdir()
    (tmp_path / "img" / "a.jpg").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="img/a.jpg"):
        dedup.hash_records(tmp_path, [mk(image="img/a.jpg")])


def test_brighten_clips_to_uint8(brighten):
    out = brighten(np.full((2, 2, 3), 250, dtype=np.uint8), 1.5)
    assert out.dtype == np.uint8 and (out == 255).all()


def test_groups_are_transitive_and_use_smallest_index():
    # b is 4 bits from a, d is 2 bits from a and b; c is 8+ bits from all of them.
    a, b, c, d = 0, 0b1111, 0b1111111100000000, 0b11
    g = duplicate_groups([c, b, a, d], max_distance=4)
    assert g[1] == g[2] == g[3] == 1
    assert g[0] == 0


def _random_hashes(n=300, seed=1):
    rng = np.random.default_rng(seed)
    base = [int(x) for x in rng.integers(0, 2**63, size=n // 2, dtype=np.int64)]
    out = list(base)
    for h in base:  # near-duplicate partner with 0-3 flipped bits
        for f in rng.integers(0, 64, size=int(rng.integers(0, 4))):
            h ^= 1 << int(f)
        out.append(h)
    return out


def test_popcount_table_matches_builtin():
    x = np.random.default_rng(0).integers(0, 2**63, size=1000, dtype=np.int64).astype(np.uint64)
    x[0], x[1] = 0, np.uint64(2**64 - 1)
    assert dedup._popcount_table(x[:2]).tolist() == [0, 64]
    if hasattr(np, "bitwise_count"):
        assert (dedup._popcount_table(x) == dedup._popcount_builtin(x)).all()
    # 2-D shape is preserved
    assert dedup._popcount_table(x[:6].reshape(2, 3)).shape == (2, 3)


def test_duplicate_groups_same_on_both_popcount_paths(monkeypatch):
    hashes = _random_hashes()
    monkeypatch.setattr(dedup, "_popcount", dedup._popcount_table)
    table = duplicate_groups(hashes)
    table_cross = cross_split_duplicates(hashes[:100], hashes[100:])
    if hasattr(np, "bitwise_count"):
        monkeypatch.setattr(dedup, "_popcount", dedup._popcount_builtin)
        assert duplicate_groups(hashes) == table
        assert cross_split_duplicates(hashes[:100], hashes[100:]) == table_cross
    assert len(set(table)) < len(hashes)  # the planted near-duplicates really merged


def test_chunked_result_equals_unchunked(monkeypatch):
    hashes = _random_hashes()
    whole = duplicate_groups(hashes)
    whole_cross = cross_split_duplicates(hashes[:100], hashes[100:])
    monkeypatch.setattr(dedup, "_CHUNK_ELEMENTS", 700)  # a few rows per chunk
    assert duplicate_groups(hashes) == whole
    assert cross_split_duplicates(hashes[:100], hashes[100:]) == whole_cross


def test_cross_split_duplicates_counts_pairs_within_distance():
    a = [0, 2**64 - 1]
    b = [1, 3, 2**63]
    assert cross_split_duplicates(a, b, max_distance=4) == 3   # 0 vs each of b; all-ones vs none
    assert cross_split_duplicates(a, b, max_distance=1) == 2
    assert cross_split_duplicates([], b) == 0
    assert cross_split_duplicates(a, []) == 0
