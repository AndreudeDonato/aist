"""Cache stamps: what a cached result was computed from (code version, settings, input-file hashes).

A cache is rebuilt when its stored stamp differs from the current one, so a change of the thermo or
clustering settings, of an input file (whatever its mtime) or of ``config.CACHE_VERSION`` (bumped when a
code change alters cached numbers) is never missed.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os

import numpy as np

from figures.config import CACHE_VERSION

_DIGESTS: dict[tuple[str, int, int], str] = {}


def file_digest(path: str) -> str:
    """SHA-1 of a file, memoised per (path, size, mtime) within the process."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha1()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _DIGESTS[key] = h.hexdigest()
    return _DIGESTS[key]


def gc_config_settings(cfg) -> dict:
    """The ``GCConfig`` fields that change results (paths and threshold-traj selections left out)."""
    return {k: v for k, v in dataclasses.asdict(cfg).items() if k not in ("reference_file", "selection_ranges")}


def make_stamp(kind: str, *, inputs=(), **settings) -> str:
    parts = {
        "cache_version": CACHE_VERSION,
        "kind": kind,
        "inputs": {os.path.basename(p): file_digest(p) for p in inputs if os.path.exists(p)},
        **settings,
    }
    return json.dumps(parts, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))


def npz_stamp(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with np.load(path, allow_pickle=True) as data:
        return str(data["cache_stamp"]) if "cache_stamp" in data.files else None


def write_npz_stamp(path: str, stamp: str) -> None:
    data = dict(np.load(path, allow_pickle=True))
    data["cache_stamp"] = np.array(stamp)
    np.savez(path, **data)


def file_stamp(path: str) -> str | None:
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return fh.read()


def write_file_stamp(path: str, stamp: str) -> None:
    with open(path, "w") as fh:
        fh.write(stamp)
