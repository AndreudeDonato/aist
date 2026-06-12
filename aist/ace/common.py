"""Label helpers + discrete colour palettes for ACE atom / structure clusters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def ace_atom_label(species: str, cluster_id: int) -> str:
    """Per-atom motif label (compatible with the visualizer JSON)."""
    return f"{species}-A{int(cluster_id)}"


def discrete_palette(cmap_name: str, k: int):
    """``k`` evenly-spaced colours sampled from a matplotlib colormap."""
    cmap = plt.get_cmap(cmap_name)
    if k <= 1:
        return [cmap(0.5)]
    return [cmap(i / (k - 1)) for i in range(k)]


def struct_cluster_palette(k: int):
    """Qualitative palette for structure-level clusters (tab10 / tab20)."""
    return plt.get_cmap("tab10" if k <= 10 else "tab20")


def gradient_colormap(colors, name: str = "ace_gradient"):
    """Continuous colormap that linearly blends through ``colors`` (index 0 … k−1)."""
    from matplotlib.colors import LinearSegmentedColormap

    colors = [np.asarray(c) for c in colors]
    if len(colors) == 0:
        return plt.get_cmap("viridis")
    if len(colors) == 1:
        return LinearSegmentedColormap.from_list(name, [colors[0], colors[0]], N=256)
    return LinearSegmentedColormap.from_list(name, colors, N=256)


DATASET_COLORS = {
    "PBE": "#2166ac",
    "PBE0": "#b2182b",
}


def composition_features_for_struct(
    n_structures: int,
    per_struct_atom_clusters: list[np.ndarray],
    species: str,
    k_clusters: int,
) -> np.ndarray:
    """Per-structure fractions of each atom cluster for one species."""
    _ = species  # kept for symmetry with caller; KMeans never produces noise
    out = np.zeros((n_structures, k_clusters), dtype=float)
    for si in range(n_structures):
        labs = np.asarray(per_struct_atom_clusters[si], dtype=int)
        if labs.size == 0:
            continue
        counts = np.bincount(labs[labs >= 0], minlength=k_clusters).astype(float)
        total = counts.sum()
        if total > 0:
            out[si] = counts / total
    return out
