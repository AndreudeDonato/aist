"""Lock RNGs and BLAS threading so ACE clustering repeats bit-for-bit."""

from __future__ import annotations

import os
import random

import numpy as np


def set_reproducible_seed(seed: int) -> None:
    """Seed Python/NumPy RNGs and force single-thread BLAS (best-effort)."""
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    random.seed(seed)
    np.random.seed(seed)
