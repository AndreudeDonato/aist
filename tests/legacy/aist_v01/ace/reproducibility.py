"""Lock RNGs and BLAS/OpenMP threading so ACE clustering repeats bit-for-bit."""

from __future__ import annotations

import random

import numpy as np

_THREAD_LIMITS = None


def set_reproducible_seed(seed: int) -> None:
    """Seed Python/NumPy RNGs and limit BLAS/OpenMP pools to one thread for the rest of the process.

    Setting ``OMP_NUM_THREADS`` etc. has no effect once numpy/sklearn are imported, so the pools are
    limited at runtime with threadpoolctl.
    """
    global _THREAD_LIMITS
    from threadpoolctl import threadpool_limits

    if _THREAD_LIMITS is None:
        _THREAD_LIMITS = threadpool_limits(limits=1)
    random.seed(seed)
    np.random.seed(seed)
