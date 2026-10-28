"""ACE-style (SOAP) per-atom and structure clustering.

The pipeline (``aist.ace.pipeline``: dscribe, umap, numba) is imported only when one of its entry points
is used, so ``import aist_v01.ace.<module>`` stays light.
"""

__all__ = ["configure_outputs", "main", "set_data_dir"]


def __getattr__(name):
    if name in __all__:
        from aist_v01.ace import pipeline

        return getattr(pipeline, name)
    raise AttributeError(f"module 'aist.ace' has no attribute {name!r}")
