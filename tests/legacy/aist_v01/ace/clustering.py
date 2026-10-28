"""PCA -> UMAP -> KMeans for atoms; PCA -> KMeans for structures.

Atom SOAP descriptors: scaler -> PCA (var) -> UMAP -> silhouette KMeans.
Structure motif fractions are low-dimensional; PCA + KMeans is sufficient.
"""

from __future__ import annotations

import numpy as np
import umap
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Silhouette-based KMeans k selection
# ---------------------------------------------------------------------------


def _silhouette_sample_score(X: np.ndarray, labels: np.ndarray, *, n_sample: int,
                             rng: np.random.Generator) -> float:
    """Silhouette computed on a random subset (full silhouette is O(n^2))."""
    n = X.shape[0]
    if n <= n_sample:
        return float(silhouette_score(X, labels))
    idx = rng.choice(n, size=n_sample, replace=False)
    return float(silhouette_score(X[idx], labels[idx]))


def kmeans_with_silhouette(
    X: np.ndarray,
    *,
    k_min: int,
    k_max: int,
    sil_sample: int = 5000,
    random_state: int = 42,
) -> tuple[KMeans, np.ndarray, dict]:
    """Run KMeans for ``k`` in ``[k_min, k_max]`` and pick the silhouette winner.

    Returns ``(best_model, best_labels, info)`` where ``info`` contains the
    per-k silhouette scores and the chosen ``k``.
    """
    rng = np.random.default_rng(random_state)
    scores: dict[int, float] = {}
    best = None
    best_score = -np.inf
    for k in range(max(2, k_min), max(2, k_max) + 1):
        km = KMeans(
            n_clusters=k,
            n_init=10,
            algorithm="lloyd",
            random_state=random_state,
        )
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2:
            continue
        s = _silhouette_sample_score(X, labels, n_sample=sil_sample, rng=rng)
        scores[k] = s
        if s > best_score:
            best_score = s
            best = (km, labels, k)
    if best is None:
        km = KMeans(
            n_clusters=max(2, k_min),
            n_init=10,
            algorithm="lloyd",
            random_state=random_state,
        )
        labels = km.fit_predict(X)
        best = (km, labels, max(2, k_min))
        scores[best[2]] = float("nan")
    km, labels, k = best
    return km, labels, {"k": int(k), "silhouettes": scores, "best_score": float(best_score)}


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Renumber so cluster 0 is the largest, 1 the next, etc."""
    if labels.size == 0:
        return labels.copy()
    uniq, counts = np.unique(labels, return_counts=True)
    order = np.argsort(-counts)
    remap = {int(uniq[old_pos]): new_id for new_id, old_pos in enumerate(order)}
    out = labels.copy()
    for old_id, new_id in remap.items():
        out[labels == old_id] = new_id
    return out


# ---------------------------------------------------------------------------
# Per-atom pipeline:  StandardScaler -> PCA(var) -> KMeans
# ---------------------------------------------------------------------------


def _pca_plot_axes(pca: PCA) -> tuple[str, str]:
    evr = pca.explained_variance_ratio_
    x1 = f"PC1 ({evr[0] * 100:.1f}%)"
    x2 = (
        f"PC2 ({evr[1] * 100:.1f}%)"
        if evr.size >= 2
        else "PC2"
    )
    return x1, x2


def cluster_atoms_pca_kmeans(
    X_raw: np.ndarray,
    *,
    pca_variance_target: float,
    k_min: int = 2,
    k_max: int = 8,
    standardize: bool = True,
    random_state: int = 42,
    sil_sample: int = 5000,
    label_for_print: str = "",
):
    """Cluster per-atom SOAP descriptors with PCA + silhouette-selected KMeans."""
    n = X_raw.shape[0]
    if n == 0:
        raise RuntimeError(f"No data points for {label_for_print or 'cluster'}")

    if standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X_std = X_raw

    pca = PCA(n_components=float(pca_variance_target), svd_solver="full")
    X_pca = pca.fit_transform(X_std)
    cum_var = float(np.sum(pca.explained_variance_ratio_))

    km, labels_raw, sil_info = kmeans_with_silhouette(
        X_pca,
        k_min=k_min, k_max=k_max,
        sil_sample=sil_sample, random_state=random_state,
    )
    labels = _relabel_by_size(labels_raw)
    k = int(labels.max()) + 1

    centroids_pca = np.array(
        [X_pca[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(X_pca.shape[1])
         for c in range(k)]
    )

    cluster_sizes = np.array([(labels == c).sum() for c in range(k)], dtype=int)
    if label_for_print:
        sizes_str = ", ".join(f"A{c}={cluster_sizes[c]}" for c in range(k))
        sil_str = ", ".join(f"k={kk}:{ss:.3f}" for kk, ss in sorted(sil_info["silhouettes"].items()))
        print(
            f"[{label_for_print}] n={n}  pca_dim={X_pca.shape[1]} (var={cum_var:.3f})  "
            f"k={k} ({sizes_str})  silhouette={sil_info['best_score']:.3f}  [{sil_str}]"
        )

    plot = X_pca[:, :2] if X_pca.shape[1] >= 2 else np.hstack(
        [X_pca, np.zeros((n, 2 - X_pca.shape[1]))]
    )

    return {
        "X_raw": X_raw,
        "X_std": X_std,
        "X_pca": X_pca,
        "X_plot": plot,
        "labels": labels,
        "k": k,
        "centroids_pca": centroids_pca,
        "cluster_sizes": cluster_sizes,
        "scaler": scaler,
        "pca": pca,
        "kmeans": km,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "silhouette_info": sil_info,
        "plot_axes": _pca_plot_axes(pca),
        "random_state": random_state,
    }


def cluster_atoms_umap_kmeans(
    X_raw: np.ndarray,
    *,
    pca_variance_target: float,
    umap_n_components: int = 2,
    umap_n_neighbors: int = 30,
    umap_min_dist: float = 0.0,
    umap_metric: str = "euclidean",
    k_min: int = 2,
    k_max: int = 8,
    standardize: bool = True,
    random_state: int = 42,
    sil_sample: int = 5000,
    label_for_print: str = "",
):
    """Cluster per-atom SOAP descriptors: scaler -> PCA(var) -> UMAP -> KMeans."""
    n = X_raw.shape[0]
    if n == 0:
        raise RuntimeError(f"No data points for {label_for_print or 'cluster'}")

    if standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X_std = X_raw

    pca = PCA(n_components=float(pca_variance_target), svd_solver="full")
    X_pca = pca.fit_transform(X_std)
    cum_var = float(np.sum(pca.explained_variance_ratio_))

    reducer = umap.UMAP(
        n_components=int(umap_n_components),
        n_neighbors=int(umap_n_neighbors),
        min_dist=float(umap_min_dist),
        metric=umap_metric,
        random_state=random_state,
        init="pca",
        n_jobs=1,
    )
    X_umap = reducer.fit_transform(X_pca)

    km, labels_raw, sil_info = kmeans_with_silhouette(
        X_umap,
        k_min=k_min, k_max=k_max,
        sil_sample=sil_sample, random_state=random_state,
    )
    labels = _relabel_by_size(labels_raw)
    k = int(labels.max()) + 1

    centroids_umap = np.array(
        [X_umap[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(X_umap.shape[1])
         for c in range(k)]
    )

    cluster_sizes = np.array([(labels == c).sum() for c in range(k)], dtype=int)
    if label_for_print:
        sizes_str = ", ".join(f"A{c}={cluster_sizes[c]}" for c in range(k))
        sil_str = ", ".join(f"k={kk}:{ss:.3f}" for kk, ss in sorted(sil_info["silhouettes"].items()))
        print(
            f"[{label_for_print}] n={n}  pca_dim={X_pca.shape[1]} (var={cum_var:.3f})  "
            f"umap_dim={X_umap.shape[1]}  k={k} ({sizes_str})  "
            f"silhouette={sil_info['best_score']:.3f}  [{sil_str}]"
        )

    return {
        "X_raw": X_raw,
        "X_std": X_std,
        "X_pca": X_pca,
        "X_umap": X_umap,
        "X_plot": X_umap[:, :2] if X_umap.shape[1] >= 2 else X_umap,
        "labels": labels,
        "k": k,
        "centroids_umap": centroids_umap,
        "cluster_sizes": cluster_sizes,
        "scaler": scaler,
        "pca": pca,
        "umap": reducer,
        "kmeans": km,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "silhouette_info": sil_info,
        "plot_axes": ("UMAP-1", "UMAP-2"),
        "random_state": random_state,
    }


# ---------------------------------------------------------------------------
# Per-structure pipeline: StandardScaler -> PCA(var) -> KMeans
# ---------------------------------------------------------------------------


def cluster_structures_pca_kmeans(
    X_raw: np.ndarray,
    *,
    pca_variance_target: float,
    k_min: int = 2,
    k_max: int = 8,
    standardize: bool = True,
    random_state: int = 42,
    sil_sample: int = 5000,
    label_for_print: str = "",
    feature_weights: np.ndarray | None = None,
):
    """Cluster per-structure descriptors (composition fractions) with PCA + KMeans.

    ``feature_weights`` (length = n_features) is applied **after** standardization
    so a value > 1 genuinely emphasises that column (a pre-scaling would be undone
    by the StandardScaler). Use it to up-weight an interpretable feature such as
    the molecular-O fraction without otherwise altering the descriptor.
    """
    n = X_raw.shape[0]
    if n == 0:
        raise RuntimeError(f"No data points for {label_for_print or 'cluster'}")

    if standardize:
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X_raw)
    else:
        scaler = None
        X_std = X_raw

    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float).reshape(1, -1)
        if w.shape[1] != X_std.shape[1]:
            raise ValueError(
                f"feature_weights length {w.shape[1]} != n_features {X_std.shape[1]}"
            )
        X_std = X_std * w

    pca = PCA(n_components=float(pca_variance_target), svd_solver="full")
    X_pca = pca.fit_transform(X_std)
    cum_var = float(np.sum(pca.explained_variance_ratio_))

    km, labels_raw, sil_info = kmeans_with_silhouette(
        X_pca,
        k_min=k_min, k_max=k_max,
        sil_sample=sil_sample, random_state=random_state,
    )
    labels = labels_raw.copy()
    k = int(labels.max()) + 1

    centroids_pca = np.array(
        [X_pca[labels == c].mean(axis=0) if (labels == c).any() else np.zeros(X_pca.shape[1])
         for c in range(k)]
    )
    cluster_sizes = np.array([(labels == c).sum() for c in range(k)], dtype=int)
    if label_for_print:
        sizes_str = ", ".join(f"C{c}={cluster_sizes[c]}" for c in range(k))
        sil_str = ", ".join(f"k={kk}:{ss:.3f}" for kk, ss in sorted(sil_info["silhouettes"].items()))
        print(
            f"[{label_for_print}] n={n}  pca_dim={X_pca.shape[1]} (var={cum_var:.3f})  "
            f"k={k} ({sizes_str})  silhouette={sil_info['best_score']:.3f}  [{sil_str}]"
        )

    plot = X_pca[:, :2] if X_pca.shape[1] >= 2 else np.hstack([X_pca, np.zeros((n, 2 - X_pca.shape[1]))])

    return {
        "X_raw": X_raw,
        "X_std": X_std,
        "X_pca": X_pca,
        "X_plot": plot,
        "labels": labels,
        "k": k,
        "centroids_pca": centroids_pca,
        "cluster_sizes": cluster_sizes,
        "scaler": scaler,
        "pca": pca,
        "kmeans": km,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_variance": cum_var,
        "silhouette_info": sil_info,
        "plot_axes": (f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)",
                      f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)"
                      if pca.explained_variance_ratio_.size >= 2 else "PC2"),
        "random_state": random_state,
    }


def reorder_ir_atom_clusters_by_oxidation(
    ir_fit,
    provenance,
    struct_n_O: np.ndarray,
    struct_n_Ir: np.ndarray,
    *,
    label_for_print: str = "",
):
    """Re-label Ir atom clusters by ascending mean parent-structure ``n_O / n_Ir``.

    Each Ir atom inherits the oxidation ratio of the structure it belongs to;
    clusters are renumbered so Ir-A0 is bluest (lowest mean O/Ir) and Ir-A{k-1}
    reddest (highest).
    """
    ratio = (
        np.asarray(struct_n_O, dtype=float)
        / np.maximum(np.asarray(struct_n_Ir, dtype=float), 1.0)
    )
    labels = ir_fit["labels"]
    k = ir_fit["k"]
    means = np.zeros(k)
    for c in range(k):
        rows = np.where(labels == c)[0]
        if rows.size == 0:
            continue
        struct_ids = np.fromiter(
            (provenance[int(r)][0] for r in rows), dtype=int, count=rows.size,
        )
        means[c] = float(ratio[struct_ids].mean())

    order = np.argsort(means)
    remap = np.empty(k, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id

    ir_fit["labels"] = remap[labels]
    if "centroids_umap" in ir_fit:
        ir_fit["centroids_umap"] = ir_fit["centroids_umap"][order]
    if "centroids_pca" in ir_fit:
        ir_fit["centroids_pca"] = ir_fit["centroids_pca"][order]
    ir_fit["cluster_sizes"] = np.array(
        [(ir_fit["labels"] == c).sum() for c in range(k)], dtype=int,
    )
    if ir_fit.get("kmeans") is not None:
        ir_fit["kmeans"].cluster_centers_ = ir_fit["kmeans"].cluster_centers_[order]
    ir_fit["mean_oxidation"] = means[order]

    if label_for_print:
        ox_str = ", ".join(
            f"Ir-A{c}: n_O/n_Ir={means[order[c]]:.3f} (n={ir_fit['cluster_sizes'][c]})"
            for c in range(k)
        )
        print(f"[{label_for_print}] ordered by structure oxidation -> {ox_str}")


# ---------------------------------------------------------------------------
# Structure cluster reordering by oxidation
# ---------------------------------------------------------------------------


def distribute_struct_labels_to_datasets(datasets_loaded, pooled_labels: np.ndarray):
    """Slice pooled structure-cluster labels back onto each dataset dict."""
    offset = 0
    for d in datasets_loaded:
        n = len(d["structures"])
        d["struct_labels"] = np.asarray(pooled_labels[offset: offset + n], dtype=int)
        offset += n


def reorder_struct_clusters_by_oxidation(result, n_O, n_Ir, *, label_for_print: str = ""):
    """Re-label structure clusters by ascending mean ``n_O / n_Ir``.

    After this, C0 is the least oxidised group and C{k-1} the most oxidised.
    """
    if result is None:
        return
    fit = result["fit"]
    k = fit["k"]
    ratio = np.asarray(n_O, dtype=float) / np.maximum(np.asarray(n_Ir, dtype=float), 1.0)
    means = np.zeros(k)
    for c in range(k):
        mask = fit["labels"] == c
        means[c] = float(ratio[mask].mean()) if mask.any() else 0.0
    order = np.argsort(means)
    remap = np.empty(k, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    fit["labels"] = remap[fit["labels"]]
    fit["centroids_pca"] = fit["centroids_pca"][order]
    fit["cluster_sizes"] = np.array([(fit["labels"] == c).sum() for c in range(k)], dtype=int)
    if fit.get("kmeans") is not None:
        fit["kmeans"].cluster_centers_ = fit["kmeans"].cluster_centers_[order]
    if label_for_print:
        ox_str = ", ".join(
            f"C{c}: n_O/n_Ir={means[order[c]]:.3f} (n={fit['cluster_sizes'][c]})"
            for c in range(k)
        )
        print(f"[{label_for_print}] ordered by oxidation -> {ox_str}")


# ---------------------------------------------------------------------------
# Provenance helper
# ---------------------------------------------------------------------------


def assign_atom_clusters_to_structures(
    structures,
    provenance,
    flat_labels: np.ndarray,
    species: str,
):
    """Distribute pooled atom labels back into per-structure per-atom lists.

    KMeans assigns every atom (no -1), so the placeholders here are only ever
    overwritten by real cluster ids.
    """
    n_struct = len(structures)
    per_struct: list[list[int]] = [[] for _ in range(n_struct)]
    n_species_per_struct = [
        int(np.sum(np.asarray(a.get_chemical_symbols()) == species))
        for a in structures
    ]
    for si, n_sp in enumerate(n_species_per_struct):
        per_struct[si] = [-1] * n_sp

    species_index_in_struct: dict[int, dict[int, int]] = {}
    for si, atoms in enumerate(structures):
        species_index_in_struct[si] = {}
        p = 0
        for ai, sym in enumerate(atoms.get_chemical_symbols()):
            if sym == species:
                species_index_in_struct[si][ai] = p
                p += 1

    for row, (si, ai) in enumerate(provenance):
        pos = species_index_in_struct[si][ai]
        per_struct[si][pos] = int(flat_labels[row])

    return [np.array(lst, dtype=int) for lst in per_struct]


# O atom motifs are reported as three physical types, always in this order (O-A0, O-A1, O-A2);
# figures/config.py FIG04_O_BRIDGING_IDX / FIG04_O_TERMINAL_IDX rely on it.
O_MOTIF_ORDER = ("bridging", "terminal", "molecular")
O_IR_BOND_MAX = 2.3   # A, Ir-O bond for counting the Ir neighbours of an O atom
O_O_BOND_MAX = 1.6    # A, O-O contact that makes an O atom molecular (O2, peroxo, superoxo)


def o_bonding_type(atoms, index: int) -> str:
    """Geometric type of O atom ``index``: molecular (an O within 1.6 A), terminal (one Ir), bridging (2+)."""
    sym = np.asarray(atoms.get_chemical_symbols())
    d = np.linalg.norm(atoms.positions - atoms.positions[index], axis=1)
    if np.any((sym == "O") & (d < O_O_BOND_MAX) & (np.arange(len(sym)) != index)):
        return "molecular"
    n_ir = int(np.sum((sym == "Ir") & (d < O_IR_BOND_MAX)))
    return "terminal" if n_ir == 1 else "bridging" if n_ir >= 2 else "unbound"


def reorder_o_atom_clusters_by_bonding(o_fit, provenance, structures, *, min_purity: float = 0.9,
                                       label_for_print: str = ""):
    """Name the O atom clusters by bonding and renumber them bridging, terminal, molecular.

    The O clustering is unsupervised, so nothing forces its three clusters to be the three physical O
    motifs, and size-ordered labels can swap when the data change. Each O atom is typed from its geometry
    (``o_bonding_type``); every cluster must be at least ``min_purity`` one type and the clusters must cover
    terminal, bridging and molecular one-to-one, otherwise this raises.
    """
    labels = o_fit["labels"]
    k = int(o_fit["k"])
    types = np.array([o_bonding_type(structures[si], ai) for si, ai in provenance], dtype=object)
    names, purity = [], []
    for c in range(k):
        t = types[labels == c]
        vals, counts = np.unique(t, return_counts=True)
        best = int(np.argmax(counts))
        names.append(str(vals[best]))
        purity.append(float(counts[best] / max(1, t.size)))
    report = ", ".join(f"cluster {c}: {names[c]} {100 * purity[c]:.1f}%" for c in range(k))
    if sorted(names) != sorted(O_MOTIF_ORDER) or min(purity) < min_purity:
        raise RuntimeError(
            f"O atom clusters are not the three O motifs (terminal, bridging, molecular): {report}. "
            "Check the O clustering settings (o_atom_k, UMAP neighbours) before using these labels."
        )
    order = [names.index(n) for n in O_MOTIF_ORDER]
    remap = np.empty(k, dtype=int)
    for new_id, old_id in enumerate(order):
        remap[old_id] = new_id
    o_fit["labels"] = remap[labels]
    for key in ("centroids_umap", "centroids_pca"):
        if key in o_fit:
            o_fit[key] = o_fit[key][order]
    o_fit["cluster_sizes"] = np.array([(o_fit["labels"] == c).sum() for c in range(k)], dtype=int)
    if o_fit.get("kmeans") is not None:
        o_fit["kmeans"].cluster_centers_ = o_fit["kmeans"].cluster_centers_[order]
    o_fit["motif_names"] = list(O_MOTIF_ORDER)
    o_fit["motif_purity"] = [purity[i] for i in order]
    if label_for_print:
        print(f"[{label_for_print}] motifs by bonding -> " + ", ".join(
            f"O-A{i}: {n} ({100 * o_fit['motif_purity'][i]:.1f}% pure, n={o_fit['cluster_sizes'][i]})"
            for i, n in enumerate(O_MOTIF_ORDER)))
    return o_fit
