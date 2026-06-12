"""All ACE-clustering plots (UMAP+KMeans pipeline -> every atom has a cluster)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde

from .common import DATASET_COLORS, discrete_palette, struct_cluster_palette


plt.rcParams["svg.fonttype"] = "none"


# ---------------------------------------------------------------------------
# Atom-level 2D scatter (UMAP space)
# ---------------------------------------------------------------------------


def draw_atom_scatter_on_ax(
    ax,
    fit,
    species: str,
    cmap_name: str,
    *,
    fontsize: int = 7,
    paper_style: bool = False,
):
    """Draw atom UMAP/PCA scatter on an existing axes (from cached ``fit`` dict)."""
    k = int(fit["k"])
    Xp = np.asarray(fit["X_plot"])
    labels = np.asarray(fit["labels"], dtype=int)
    colors = discrete_palette(cmap_name, k) if k > 0 else []
    mean_ox = fit.get("mean_oxidation")
    ax_labels = fit.get("plot_axes", ("UMAP-1", "UMAP-2"))

    for c in range(k):
        mask = labels == c
        if not mask.any():
            continue
        if paper_style:
            leg = f"{species}-A{c}"
        elif mean_ox is not None and len(mean_ox) > c and species == "Ir":
            leg = f"{species}-A{c} (n={int(mask.sum())}, O/Ir={mean_ox[c]:.2f})"
        else:
            leg = f"{species}-A{c} (n={int(mask.sum())})"
        ax.scatter(
            Xp[mask, 0], Xp[mask, 1],
            c=[colors[c]], s=5, alpha=0.65, edgecolors="none",
            label=leg,
        )
    ax.set_xlabel(ax_labels[0], fontsize=fontsize)
    ax.set_ylabel(ax_labels[1], fontsize=fontsize)
    if paper_style:
        ax.set_title(f"{species} atoms — UMAP", fontsize=fontsize)
    else:
        sil = fit.get("silhouette_best", fit.get("silhouette_info", {}).get("best_score", float("nan")))
        var = fit.get("cumulative_variance", float("nan"))
        ax.set_title(
            f"{species} atoms — UMAP clusters (k={k}, sil={sil:.2f}, PCA var={var:.2f})",
            fontsize=fontsize,
        )
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(fontsize=fontsize - 1, frameon=False, loc="best", markerscale=2, ncol=2)


def plot_atom_scatter(fit, species, cmap_name, output_path):
    """2D scatter of every atom of one species, coloured by KMeans cluster id."""
    fig, ax = plt.subplots(figsize=(110 / 25.4, 90 / 25.4), dpi=800)
    draw_atom_scatter_on_ax(ax, fit, species, cmap_name, fontsize=8)
    fig.tight_layout(pad=4 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] {species} atom scatter -> {output_path}")


# ---------------------------------------------------------------------------
# Structure-level 2D PCA scatter (per dataset)
# ---------------------------------------------------------------------------


def draw_struct_pca_on_ax(ax, struct_cache: dict, *, fontsize: int = 8):
    """Pooled structure PCA scatter from ``ace_struct_embeddings.npz``-style dict."""
    from matplotlib.lines import Line2D

    k = int(struct_cache["k"])
    Xp = np.asarray(struct_cache["X_plot"])
    labels = np.asarray(struct_cache["labels"], dtype=int)
    ds_idx = np.asarray(struct_cache["dataset_index"], dtype=int)
    ds_names = [str(x) for x in np.asarray(struct_cache["dataset_names"]).tolist()]
    cmap = struct_cluster_palette(max(k, 1))
    ax_labels = struct_cache.get("plot_axes", ("PC1", "PC2"))
    markers = {"PBE": "o", "PBE0": "s"}

    for di, name in enumerate(ds_names):
        sel = ds_idx == di
        if not sel.any():
            continue
        mk = markers.get(name, "o")
        ds_color = DATASET_COLORS.get(name, "black")
        for c in range(k):
            mask = sel & (labels == c)
            if not mask.any():
                continue
            ax.scatter(
                Xp[mask, 0], Xp[mask, 1],
                c=[cmap(c % cmap.N)], s=16, alpha=0.75,
                marker=mk, edgecolors=ds_color, linewidths=0.35,
            )

    cluster_handles = [
        Line2D([0], [0], marker="s", color="w", markerfacecolor=cmap(c % cmap.N),
               markersize=6, label=f"C{c}")
        for c in range(k)
    ]
    ds_handles = [
        Line2D([0], [0], marker=markers.get(name, "o"), color="w",
               markerfacecolor="none", markeredgecolor=DATASET_COLORS.get(name, "k"),
               markersize=6, label=name)
        for name in ds_names
    ]
    sil = float(struct_cache.get("silhouette_best", np.nan))
    var = float(struct_cache.get("cumulative_variance", np.nan))
    ax.set_xlabel(ax_labels[0], fontsize=fontsize)
    ax.set_ylabel(ax_labels[1], fontsize=fontsize)
    ax.set_title(
        f"Pooled structure PCA (k={k}, sil={sil:.2f}, var={var:.2f})",
        fontsize=fontsize,
    )
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.legend(handles=cluster_handles + ds_handles, fontsize=fontsize - 1,
              frameon=False, loc="best", ncol=2)


def plot_struct_pca(datasets_loaded, struct_fit, output_path):
    """Pooled 2D PCA: colour = structure cluster C, marker = dataset (PBE / PBE0)."""
    if struct_fit is None:
        return
    from aist.ace.io import build_struct_embeddings_cache

    fig, ax = plt.subplots(figsize=(95 / 25.4, 95 / 25.4), dpi=800)
    draw_struct_pca_on_ax(ax, build_struct_embeddings_cache(datasets_loaded, struct_fit), fontsize=8)
    fig.tight_layout(pad=6 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] structure PCA plot -> {output_path}")


# ---------------------------------------------------------------------------
# Composition (n_O / n_Ir) histograms per structure cluster
# ---------------------------------------------------------------------------


def plot_composition_densities(datasets_loaded, struct_fit, output_path):
    """Per dataset: oxidation KDE and n_Ir histogram per pooled structure cluster."""
    if struct_fit is None or not datasets_loaded:
        return
    k = struct_fit["k"]
    n = len(datasets_loaded)
    fig, axes = plt.subplots(2, n, figsize=(95 * n / 25.4, 110 / 25.4), dpi=800)
    if n == 1:
        axes = axes.reshape(2, 1)
    cmap = struct_cluster_palette(max(k, 1))

    for col, d in enumerate(datasets_loaded):
        labels = d["struct_labels"]
        ratio = d["n_O"] / np.maximum(d["n_Ir"], 1)

        ax = axes[0][col]
        x_min = float(ratio.min())
        x_max = float(ratio.max())
        grid = np.linspace(x_min, x_max, 256) if x_max > x_min else np.array([x_min])
        for c in range(k):
            mask = labels == c
            n_c = int(mask.sum())
            vals = ratio[mask]
            if vals.size < 2 or vals.std() < 1e-6:
                continue
            try:
                kde = gaussian_kde(vals)
                density = kde(grid) * n_c
                ax.plot(grid, density, color=cmap(c % cmap.N),
                        linewidth=1.2, label=f"C{c} (n={n_c})")
            except Exception:
                continue
        ax.set_xlabel("oxidation n_O / n_Ir", fontsize=7)
        ax.set_ylabel("structures per unit n_O/n_Ir", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title(f"{d['label']} oxidation (pooled C)", fontsize=8)
        ax.legend(fontsize=5, frameon=False, loc="upper right")

        ax = axes[1][col]
        n_ir = d["n_Ir"]
        size_min = int(n_ir.min()) if n_ir.size else 0
        size_max = int(n_ir.max()) if n_ir.size else 0
        size_bins = np.arange(size_min - 0.5, size_max + 1.5)
        size_centers = np.arange(size_min, size_max + 1)
        for c in range(k):
            mask = labels == c
            if not mask.any():
                continue
            counts, _ = np.histogram(n_ir[mask], bins=size_bins)
            ax.step(size_centers, counts, where="mid",
                    color=cmap(c % cmap.N), linewidth=1.2,
                    label=f"C{c} (n={int(mask.sum())})")
        ax.set_xlim(size_min - 0.5, size_max + 0.5)
        ax.set_ylim(bottom=0.0)
        ax.set_xticks(np.arange(size_min, size_max + 1))
        ax.set_xlabel("Ir cluster size  n_Ir", fontsize=7)
        ax.set_ylabel("number of structures", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title(f"{d['label']} Ir cluster size", fontsize=8)
        ax.legend(fontsize=5, frameon=False, loc="upper right")

    fig.tight_layout(pad=8 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] composition densities -> {output_path}")


# ---------------------------------------------------------------------------
# Atom-type composition per structure cluster (stacked bars)
# ---------------------------------------------------------------------------


def plot_struct_cluster_composition(
    datasets_loaded,
    struct_fit,
    ir_fit,
    o_fit,
    output_path,
    ir_cmap_name="coolwarm",
    o_cmap_name="plasma_r",
):
    """Atom-cluster fractions per pooled C cluster; columns = PBE vs PBE0."""
    if struct_fit is None or not datasets_loaded:
        return

    n = len(datasets_loaded)
    fig, axes = plt.subplots(2, n, figsize=(95 * n / 25.4, 130 / 25.4), dpi=800)
    if n == 1:
        axes = axes.reshape(2, 1)

    k = struct_fit["k"]
    k_Ir = ir_fit["k"]
    k_O = o_fit["k"]
    ir_colors = discrete_palette(ir_cmap_name, k_Ir)
    o_colors = discrete_palette(o_cmap_name, k_O)
    ir_bar_labels = [f"Ir-A{c}" for c in range(k_Ir)]
    o_bar_labels = [f"O-A{c}" for c in range(k_O)]
    cluster_ids = [c for c in range(k)]
    cluster_labels = [f"C{c}" for c in cluster_ids]

    for col, d in enumerate(datasets_loaded):
        labels = d["struct_labels"]

        Ir_block = np.zeros((len(cluster_ids), k_Ir))
        O_block = np.zeros((len(cluster_ids), k_O))
        for ci, cid in enumerate(cluster_ids):
            struct_rows = np.where(labels == cid)[0]
            if struct_rows.size == 0:
                continue
            Ir_block[ci] = d["ir_cluster_fractions"][struct_rows].mean(axis=0)
            O_block[ci] = d["o_cluster_fractions"][struct_rows].mean(axis=0)

        ax = axes[0][col]
        bottom = np.zeros(len(cluster_ids))
        for j in range(k_Ir):
            ax.bar(cluster_labels, Ir_block[:, j],
                   bottom=bottom, color=ir_colors[j],
                   edgecolor="black", linewidth=0.3,
                   label=ir_bar_labels[j])
            bottom += Ir_block[:, j]
        ax.set_title(f"{d['label']} — Ir motifs (pooled C)", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Ir-atom fraction", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(fontsize=5, frameon=False, loc="upper right", ncol=2)

        ax = axes[1][col]
        bottom = np.zeros(len(cluster_ids))
        for j in range(k_O):
            ax.bar(cluster_labels, O_block[:, j],
                   bottom=bottom, color=o_colors[j],
                   edgecolor="black", linewidth=0.3,
                   label=o_bar_labels[j])
            bottom += O_block[:, j]
        ax.set_title(f"{d['label']} — O motifs in pooled C", fontsize=8)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("O-atom fraction", fontsize=7)
        ax.tick_params(axis="both", labelsize=7)
        ax.legend(fontsize=5, frameon=False, loc="upper right", ncol=2)

    fig.tight_layout(pad=8 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] structure cluster composition -> {output_path}")


# ---------------------------------------------------------------------------
# Atom-type population vs oxidation (whole dataset, stacked along O/Ir)
# ---------------------------------------------------------------------------


def _kernel_smoothed_population_stack(
    ratio: np.ndarray,
    k: int,
    *,
    n_bins: int = 100,
    bandwidth=None,
    smooth_sigma: float = 1.0,
    accumulate,
):
    """Kernel-weight structures onto a regular ``ratio`` grid.

    ``accumulate(counts, si, weight)`` adds weighted contributions from
    structure ``si`` into ``counts`` (length ``k``).
    """
    ratio = np.asarray(ratio, dtype=float)
    x_min, x_max = float(ratio.min()), float(ratio.max())
    if x_max <= x_min:
        x_grid = np.array([x_min])
    else:
        x_grid = np.linspace(x_min, x_max, int(n_bins))

    span = max(x_max - x_min, 1e-9)
    bw = float(bandwidth) if bandwidth is not None else span / max(len(x_grid) * 0.55, 4.0)

    Y = np.zeros((k, len(x_grid)))
    for gi, xi in enumerate(x_grid):
        counts = np.zeros(k, dtype=float)
        for si in range(ratio.size):
            w = np.exp(-0.5 * ((ratio[si] - xi) / bw) ** 2)
            if w < 1e-12:
                continue
            accumulate(counts, si, w)
        total = counts.sum()
        if total > 0:
            Y[:, gi] = 100.0 * counts / total

    if smooth_sigma > 0 and Y.shape[1] > 3:
        Y = gaussian_filter1d(Y, sigma=float(smooth_sigma), axis=1, mode="nearest")
        col_sum = Y.sum(axis=0, keepdims=True)
        np.divide(Y, col_sum, out=Y, where=col_sum > 0)
        Y *= 100.0

    return x_grid, Y


def _atom_population_stack_vs_oxidation(
    d,
    k,
    cluster_key,
    *,
    n_bins=100,
    bandwidth=None,
    smooth_sigma=1.0,
):
    """Smoothed atom-cluster population (%) vs ``n_O / n_Ir``."""
    ratio = np.asarray(d["n_O"], dtype=float) / np.maximum(
        np.asarray(d["n_Ir"], dtype=float), 1.0,
    )
    per_struct = d[cluster_key]

    def accumulate(counts, si, w):
        for cid in per_struct[si]:
            cid = int(cid)
            if 0 <= cid < k:
                counts[cid] += w

    return _kernel_smoothed_population_stack(
        ratio, k,
        n_bins=n_bins, bandwidth=bandwidth, smooth_sigma=smooth_sigma,
        accumulate=accumulate,
    )


def _struct_population_stack_vs_oxidation(
    d,
    labels,
    k,
    *,
    n_bins=100,
    bandwidth=None,
    smooth_sigma=1.0,
):
    """Smoothed structure-cluster population (%) vs ``n_O / n_Ir``."""
    ratio = np.asarray(d["n_O"], dtype=float) / np.maximum(
        np.asarray(d["n_Ir"], dtype=float), 1.0,
    )
    labels = np.asarray(labels, dtype=int)

    def accumulate(counts, si, w):
        cid = int(labels[si])
        if 0 <= cid < k:
            counts[cid] += w

    return _kernel_smoothed_population_stack(
        ratio, k,
        n_bins=n_bins, bandwidth=bandwidth, smooth_sigma=smooth_sigma,
        accumulate=accumulate,
    )


def plot_atom_population_vs_oxidation(
    datasets_loaded,
    ir_fit,
    o_fit,
    output_path,
    ir_cmap_name="coolwarm",
    o_cmap_name="plasma_r",
    *,
    n_bins=100,
    bandwidth=None,
    smooth_sigma=1.0,
):
    """Multi-panel stack: atom-cluster population (%) vs ``n_O / n_Ir`` for full dataset.

    Layout matches ``plot_struct_cluster_composition``: Ir row, O row, one column
    per dataset.  Populations are kernel-smoothed along oxidation so uneven
    structure sampling does not produce jagged stack boundaries.
    """
    if not datasets_loaded:
        return

    n = len(datasets_loaded)
    k_Ir = ir_fit["k"]
    k_O = o_fit["k"]
    ir_colors = discrete_palette(ir_cmap_name, k_Ir)
    o_colors = discrete_palette(o_cmap_name, k_O)
    ir_labels = [f"Ir-A{c}" for c in range(k_Ir)]
    o_labels = [f"O-A{c}" for c in range(k_O)]

    fig, axes = plt.subplots(2, n, figsize=(95 * n / 25.4, 110 / 25.4), dpi=800)
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, d in enumerate(datasets_loaded):
        ratio = d["n_O"] / np.maximum(d["n_Ir"], 1)
        x_min = float(ratio.min())
        x_max = float(ratio.max())

        for row, (k, cluster_key, colors, labels, species) in enumerate([
            (k_Ir, "per_struct_ir_cluster", ir_colors, ir_labels, "Ir"),
            (k_O, "per_struct_o_cluster", o_colors, o_labels, "O"),
        ]):
            ax = axes[row][col]
            x_plot, Y = _atom_population_stack_vs_oxidation(
                d, k, cluster_key,
                n_bins=n_bins, bandwidth=bandwidth, smooth_sigma=smooth_sigma,
            )
            if Y.size == 0:
                ax.text(
                    0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray",
                )
            else:
                polys = ax.stackplot(
                    x_plot, Y,
                    colors=colors,
                    linewidth=0.0,
                    edgecolor="none",
                    labels=labels,
                )
                for pc in polys:
                    pc.set_edgecolor("none")
                    pc.set_linewidth(0.0)

            ax.set_xlim(x_min - 0.02 * (x_max - x_min + 1e-9),
                        x_max + 0.02 * (x_max - x_min + 1e-9))
            ax.set_ylim(0.0, 100.0)
            ax.set_xlabel("oxidation  n_O / n_Ir", fontsize=7)
            ax.set_ylabel("population (%)", fontsize=7)
            ax.tick_params(axis="both", labelsize=6)
            ax.set_title(f"{d['label']} — {species} atom types", fontsize=8)
            ax.legend(fontsize=5, frameon=False, loc="upper left", ncol=2)

    fig.tight_layout(pad=8 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] atom population vs oxidation -> {output_path}")


def plot_struct_population_vs_oxidation(
    datasets_loaded,
    struct_fit,
    output_path,
    *,
    n_bins=100,
    bandwidth=None,
    smooth_sigma=1.0,
):
    """Rows = PBE / PBE0: pooled C0..CN population (%) vs ``n_O / n_Ir`` (same colours)."""
    if struct_fit is None or not datasets_loaded:
        return

    k = struct_fit["k"]
    cmap = struct_cluster_palette(max(k, 1))
    colors = [cmap(c % cmap.N) for c in range(k)]
    cluster_labels = [f"C{c}" for c in range(k)]

    ratio_all = np.concatenate([
        d["n_O"] / np.maximum(d["n_Ir"], 1) for d in datasets_loaded
    ])
    x_min = float(ratio_all.min())
    x_max = float(ratio_all.max())

    n = len(datasets_loaded)
    fig, axes = plt.subplots(n, 1, figsize=(95 / 25.4, 55 * n / 25.4), dpi=800)
    if n == 1:
        axes = [axes]

    for ax, d in zip(axes, datasets_loaded):
        x_plot, Y = _struct_population_stack_vs_oxidation(
            d, d["struct_labels"], k,
            n_bins=n_bins, bandwidth=bandwidth, smooth_sigma=smooth_sigma,
        )
        if Y.size == 0 or Y.sum() <= 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=7, color="gray")
        else:
            polys = ax.stackplot(
                x_plot, Y, colors=colors, linewidth=0.0,
                edgecolor="none", labels=cluster_labels,
            )
            for pc in polys:
                pc.set_edgecolor("none")
                pc.set_linewidth(0.0)

        ax.set_xlim(x_min - 0.02 * (x_max - x_min + 1e-9),
                    x_max + 0.02 * (x_max - x_min + 1e-9))
        ax.set_ylim(0.0, 100.0)
        ax.set_ylabel("population (%)", fontsize=7)
        ax.tick_params(axis="both", labelsize=6)
        ax.set_title(f"{d['label']} — pooled structure clusters", fontsize=8)
        if d is datasets_loaded[-1]:
            ax.set_xlabel("oxidation  n_O / n_Ir", fontsize=7)
        ax.legend(fontsize=5, frameon=False, loc="upper left", ncol=2)

    fig.tight_layout(pad=8 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] structure population vs oxidation -> {output_path}")


def plot_struct_cluster_dataset_split(datasets_loaded, struct_fit, output_path):
    """Bar chart: within each pooled C, fraction of structures from PBE vs PBE0."""
    if struct_fit is None or not datasets_loaded:
        return

    k = struct_fit["k"]
    cluster_labels = [f"C{c}" for c in range(k)]
    fig, ax = plt.subplots(figsize=(95 / 25.4, 70 / 25.4), dpi=800)
    x_pos = np.arange(k)
    width = 0.65
    bottom = np.zeros(k)

    for d in datasets_loaded:
        labels = d["struct_labels"]
        fracs = np.zeros(k)
        for c in range(k):
            n_in_c = sum(int((dd["struct_labels"] == c).sum()) for dd in datasets_loaded)
            n_ds = int((labels == c).sum())
            fracs[c] = 100.0 * n_ds / n_in_c if n_in_c > 0 else 0.0
        color = DATASET_COLORS.get(d["label"], "gray")
        ax.bar(
            x_pos, fracs, width, bottom=bottom, label=d["label"],
            color=color, edgecolor="black", linewidth=0.3,
        )
        bottom += fracs

    pooled_labels = np.concatenate([d["struct_labels"] for d in datasets_loaded])
    for c in range(k):
        n_c = int((pooled_labels == c).sum())
        if n_c > 0:
            ax.text(c, 102, f"n={n_c}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(cluster_labels)
    ax.set_ylim(0, 108)
    ax.set_ylabel("structures in dataset (%)", fontsize=8)
    ax.set_xlabel("pooled structure cluster (ordered by oxidation)", fontsize=8)
    ax.set_title("PBE vs PBE0 prevalence per pooled cluster", fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.tick_params(axis="both", labelsize=7)

    fig.tight_layout(pad=8 / 25.4)
    fig.savefig(output_path, dpi=800)
    plt.close(fig)
    print(f"[save] structure cluster PBE/PBE0 split -> {output_path}")


kernel_smoothed_population_stack = _kernel_smoothed_population_stack
atom_population_stack_vs_oxidation = _atom_population_stack_vs_oxidation
struct_population_stack_vs_oxidation = _struct_population_stack_vs_oxidation


def population_stack_vs_values(
    x_values: np.ndarray,
    labels: np.ndarray,
    k: int,
    *,
    n_bins: int = 100,
    bandwidth=None,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Smoothed population (%) vs an arbitrary 1D axis (e.g. Tchar)."""
    labels = np.asarray(labels, dtype=int)

    def accumulate(counts, si, w):
        cid = int(labels[si])
        if 0 <= cid < k:
            counts[cid] += w

    return _kernel_smoothed_population_stack(
        np.asarray(x_values, dtype=float),
        k,
        n_bins=n_bins,
        bandwidth=bandwidth,
        smooth_sigma=smooth_sigma,
        accumulate=accumulate,
    )
