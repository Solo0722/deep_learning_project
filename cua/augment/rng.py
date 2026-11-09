"""Keyed RNG: independent, reproducible streams per (seed, epoch, index, stream)."""
from __future__ import annotations

import hashlib

import numpy as np


def keyed_rng(seed: int, epoch: int, index: int, stream: str) -> np.random.Generator:
    """Generator seeded from SHA-256 of the key; streams are independent of each other."""
    digest = hashlib.sha256(f"{seed}|{epoch}|{index}|{stream}".encode("utf-8")).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))
