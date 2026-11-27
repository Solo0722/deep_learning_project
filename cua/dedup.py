"""Near-duplicate screening: 64-bit difference hashes and chunked Hamming-distance grouping."""
from __future__ import annotations

import numpy as np
from PIL import Image

_CHUNK_ELEMENTS = 4_000_000  # max uint64 elements in one XOR block (~32 MB)

_BYTE_POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(rgb: np.ndarray) -> int:
    """64-bit difference hash: greyscale, resize to 9 wide x 8 high, compare horizontal neighbours."""
    with Image.fromarray(np.asarray(rgb, dtype=np.uint8)) as image:
        with image.convert("L") as grey:
            with grey.resize((9, 8), Image.Resampling.LANCZOS) as small:
                px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    value = 0
    for b in bits:
        value = (value << 1) | int(b)
    return value


def hash_records(root, records) -> list[int]:
    """dhash of each record's image under `root`; an unreadable image raises ValueError naming it."""
    from cua.image import load_rgb

    out = []
    for r in records:
        try:
            out.append(dhash(load_rgb(root, r)))
        except (OSError, ValueError) as e:
            raise ValueError(f"cannot read image {r.image} under {root}: {e}") from e
    return out


def _popcount_table(x: np.ndarray) -> np.ndarray:
    """Per-element bit count of a uint64 array via a byte lookup table (numpy < 2.0 fallback)."""
    x = np.ascontiguousarray(x, dtype=np.uint64)
    per_byte = _BYTE_POPCOUNT[x.view(np.uint8).reshape(x.shape + (8,))]
    return per_byte.sum(axis=-1, dtype=np.int64)


def _popcount_builtin(x: np.ndarray) -> np.ndarray:
    return np.bitwise_count(x).astype(np.int64)


_popcount = _popcount_builtin if hasattr(np, "bitwise_count") else _popcount_table


def _as_u64(hashes) -> np.ndarray:
    return np.array([int(h) for h in hashes], dtype=np.uint64)


def _close_pairs(a: np.ndarray, b: np.ndarray, max_distance: int, upper_only: bool):
    """Yield (i, j) index arrays with Hamming distance <= max_distance, in bounded-memory chunks."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return
    rows = max(1, _CHUNK_ELEMENTS // m)
    for start in range(0, n, rows):
        stop = min(n, start + rows)
        dist = _popcount(a[start:stop, None] ^ b[None, :])
        hit = dist <= max_distance
        if upper_only:
            hit &= np.arange(m)[None, :] > np.arange(start, stop)[:, None]
        ii, jj = np.nonzero(hit)
        yield ii + start, jj


def duplicate_groups(hashes: list[int], max_distance: int = 4) -> list[int]:
    """Union-find group id per image; the id is the smallest member index of the group."""
    h = _as_u64(hashes)
    parent = list(range(len(h)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ii, jj in _close_pairs(h, h, max_distance, upper_only=True):
        for i, j in zip(ii.tolist(), jj.tolist()):
            ri, rj = find(i), find(j)
            if ri != rj:
                lo, hi = (ri, rj) if ri < rj else (rj, ri)
                parent[hi] = lo
    return [find(i) for i in range(len(h))]


def cross_split_duplicates(a: list[int], b: list[int], max_distance: int = 4) -> int:
    """Number of (a_i, b_j) hash pairs within max_distance."""
    total = 0
    for ii, _ in _close_pairs(_as_u64(a), _as_u64(b), max_distance, upper_only=False):
        total += len(ii)
    return total
