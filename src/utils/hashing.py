"""Content hashing used to prove the hidden truth never changes."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def array_sha256(arr: np.ndarray) -> str:
    h = hashlib.sha256()
    a = np.ascontiguousarray(arr)
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
