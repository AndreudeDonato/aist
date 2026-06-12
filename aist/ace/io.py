"""CSV / trajectory writers + per-atom JSON for the viewer + visualizer copier."""

from __future__ import annotations

import csv
import json
import os
import shutil

import numpy as np
from ase.io import write
from matplotlib.colors import to_hex

from .common import ace_atom_label, discrete_palette


# ---------------------------------------------------------------------------
# CSV + trajectory writers
# ---------------------------------------------------------------------------


def save_atom_embedding_npz(fit: dict, path: str) -> None:
    """Cache atom UMAP coordinates + cluster labels for figure plotting."""
    sil = fit.get("silhouette_info") or {}
    np.savez(
        path,
        X_plot=np.asarray(fit["X_plot"], dtype=float),
        labels=np.asarray(fit["labels"], dtype=int),
        k=int(fit["k"]),
        mean_oxidation=np.asarray(fit.get("mean_oxidation", []), dtype=float),
        cumulative_variance=float(fit.get("cumulative_variance", np.nan)),
        silhouette_best=float(sil.get("best_score", np.nan)),
        plot_axes=np.asarray(fit.get("plot_axes", ("UMAP-1", "UMAP-2")), dtype=object),
    )
    print(f"[save] atom embedding cache -> {path}")


def build_struct_embeddings_cache(datasets_loaded, struct_fit: dict) -> dict:
    """In-memory structure embedding cache (same fields as the NPZ)."""
    blocks_x = []
    blocks_lab = []
    blocks_ds = []
    for di, d in enumerate(datasets_loaded):
        blocks_x.append(np.asarray(d["struct_x_plot"], dtype=float))
        blocks_lab.append(np.asarray(d["struct_labels"], dtype=int))
        blocks_ds.append(np.full(len(d["structures"]), di, dtype=int))

    sil = struct_fit.get("silhouette_info") or {}
    return {
        "X_plot": np.vstack(blocks_x) if blocks_x else np.zeros((0, 2)),
        "labels": np.concatenate(blocks_lab) if blocks_lab else np.zeros(0, dtype=int),
        "dataset_index": np.concatenate(blocks_ds) if blocks_ds else np.zeros(0, dtype=int),
        "dataset_names": [d["label"] for d in datasets_loaded],
        "k": int(struct_fit["k"]),
        "cumulative_variance": float(struct_fit.get("cumulative_variance", np.nan)),
        "silhouette_best": float(sil.get("best_score", np.nan)),
        "plot_axes": tuple(struct_fit.get("plot_axes", ("PC1", "PC2"))),
    }


def save_struct_embeddings_npz(datasets_loaded, struct_fit: dict, path: str) -> None:
    """Cache pooled structure PCA coordinates + per-structure cluster labels."""
    cache = build_struct_embeddings_cache(datasets_loaded, struct_fit)
    np.savez(
        path,
        X_plot=cache["X_plot"],
        labels=cache["labels"],
        dataset_index=cache["dataset_index"],
        dataset_names=np.asarray(cache["dataset_names"], dtype=object),
        k=cache["k"],
        cumulative_variance=cache["cumulative_variance"],
        silhouette_best=cache["silhouette_best"],
        plot_axes=np.asarray(cache["plot_axes"], dtype=object),
    )
    print(f"[save] structure embedding cache -> {path}")


def load_embedding_npz(path: str) -> dict:
    """Load an ACE or GC cache NPZ as a plain dict."""
    data = np.load(path, allow_pickle=True)
    out = {k: data[k] for k in data.files}
    if "dataset_names" in out:
        out["dataset_names"] = [str(x) for x in np.asarray(out["dataset_names"]).tolist()]
    if "plot_axes" in out:
        out["plot_axes"] = tuple(np.asarray(out["plot_axes"]).tolist())
    return out


def atom_fit_from_npz(path: str) -> dict:
    """Reconstruct an atom ``fit`` dict for plotting helpers."""
    d = load_embedding_npz(path)
    return {
        "X_plot": d["X_plot"],
        "labels": d["labels"],
        "k": int(d["k"]),
        "mean_oxidation": d.get("mean_oxidation"),
        "cumulative_variance": float(d.get("cumulative_variance", np.nan)),
        "silhouette_best": float(d.get("silhouette_best", np.nan)),
        "plot_axes": d.get("plot_axes", ("UMAP-1", "UMAP-2")),
    }


def save_features_csv(d, struct_fit, path):
    """Per-structure feature CSV (cluster + composition fractions)."""
    label = d["label"]
    cluster_labels = (
        d["struct_labels"].tolist()
        if struct_fit is not None and "struct_labels" in d
        else [-1] * len(d["structures"])
    )
    ir_frac = d["ir_cluster_fractions"]
    o_frac = d["o_cluster_fractions"]
    k_Ir = ir_frac.shape[1] if ir_frac.ndim == 2 else 0
    k_O = o_frac.shape[1] if o_frac.ndim == 2 else 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        header = ["frame_index", "formula", "n_Ir", "n_O", "cluster"]
        header += [f"Ir-A{c}_frac" for c in range(k_Ir)]
        header += [f"O-A{c}_frac" for c in range(k_O)]
        w.writerow(header)
        for si in range(len(d["structures"])):
            row = [
                si, d["formulas"][si], int(d["n_Ir"][si]), int(d["n_O"][si]),
                int(cluster_labels[si]),
            ]
            row += [f"{v:.6f}" for v in ir_frac[si]]
            row += [f"{v:.6f}" for v in o_frac[si]]
            w.writerow(row)
    print(f"[save] features ({label}) -> {path}")


def _centroid_distance_space(fit):
    """Use UMAP coords for atoms, PCA coords for structures.  Returns (X, centroids)."""
    if "centroids_umap" in fit:
        return fit["X_umap"], fit["centroids_umap"]
    return fit["X_pca"], fit["centroids_pca"]


def _result_for_dataset(d, struct_fit):
    """Wrap pooled ``struct_fit`` with per-dataset structure labels."""
    if struct_fit is None:
        return None
    return {"fit": {**struct_fit, "labels": d["struct_labels"]}}


def save_centroid_traj(d, struct_fit, path):
    result = _result_for_dataset(d, struct_fit)
    if result is None:
        return
    label = d["label"]
    fit = result["fit"]
    Xp = d["struct_x_pca"]
    cp = fit["centroids_pca"]
    representatives = []
    for c in range(fit["k"]):
        members = np.where(d["struct_labels"] == c)[0]
        if members.size == 0:
            continue
        d2 = np.linalg.norm(Xp[members] - cp[c], axis=1)
        best = int(members[np.argmin(d2)])
        atoms = d["structures"][best].copy()
        info = dict(atoms.info)
        info.update({
            "dataset": label,
            "cluster": int(c),
            "rank_in_cluster": 0,
            "distance_to_centroid": float(np.min(d2)),
            "frame_index": int(best),
            "formula": d["formulas"][best],
        })
        atoms.info = info
        representatives.append(atoms)
        print(f"[centroid] {label} C{c}: frame={best} formula={d['formulas'][best]}")
    if representatives:
        write(path, representatives)
        print(f"[save] centroids ({label}) -> {path}")


def save_group_trajs(d, struct_fit, out_dir):
    result = _result_for_dataset(d, struct_fit)
    if result is None:
        return
    label = d["label"]
    os.makedirs(out_dir, exist_ok=True)
    fit = result["fit"]
    Xp = d["struct_x_pca"]
    cp = fit["centroids_pca"]
    for c in range(fit["k"]):
        members = np.where(d["struct_labels"] == c)[0]
        if members.size == 0:
            continue
        dists = np.linalg.norm(Xp[members] - cp[c], axis=1)
        order = np.argsort(dists)
        atoms_list = []
        for rank, j in enumerate(order):
            si = int(members[j])
            atoms = d["structures"][si].copy()
            info = dict(atoms.info)
            info.update({
                "dataset": label,
                "cluster": int(c),
                "rank_in_cluster": int(rank),
                "distance_to_centroid": float(dists[j]),
                "frame_index": si,
                "formula": d["formulas"][si],
            })
            atoms.info = info
            atoms_list.append(atoms)
        path = os.path.join(out_dir, f"C{c}.traj")
        write(path, atoms_list)
        print(f"[group] {label} C{c}: {len(atoms_list)} structures -> {path}")


def save_pooled_group_trajs(datasets_loaded, struct_fit, out_dir):
    """Write ``C*.traj`` with structures from all datasets (global pooled clusters)."""
    if struct_fit is None:
        return
    structures = []
    formulas = []
    for d in datasets_loaded:
        structures.extend(d["structures"])
        formulas.extend(d["formulas"])
    pooled = {
        "label": "pooled",
        "structures": structures,
        "formulas": formulas,
        "struct_labels": np.asarray(struct_fit["labels"], dtype=int),
        "struct_x_pca": struct_fit["X_pca"],
    }
    save_group_trajs(pooled, struct_fit, out_dir)


def save_pooled_centroid_traj(datasets_loaded, struct_fit, path):
    """Centroid representatives from the pooled cluster model (may mix PBE/PBE0)."""
    if struct_fit is None:
        return
    structures = []
    formulas = []
    for d in datasets_loaded:
        structures.extend(d["structures"])
        formulas.extend(d["formulas"])
    pooled = {
        "label": "pooled",
        "structures": structures,
        "formulas": formulas,
        "struct_labels": np.asarray(struct_fit["labels"], dtype=int),
        "struct_x_pca": struct_fit["X_pca"],
    }
    fit = struct_fit
    cp = fit["centroids_pca"]
    Xp = fit["X_pca"]
    representatives = []
    for c in range(fit["k"]):
        members = np.where(fit["labels"] == c)[0]
        if members.size == 0:
            continue
        d2 = np.linalg.norm(Xp[members] - cp[c], axis=1)
        best = int(members[np.argmin(d2)])
        atoms = structures[best].copy()
        info = dict(atoms.info)
        info.update({
            "dataset": atoms.info.get("dataset", "?"),
            "cluster": int(c),
            "rank_in_cluster": 0,
            "distance_to_centroid": float(np.min(d2)),
            "frame_index": int(best),
            "formula": formulas[best],
        })
        atoms.info = info
        representatives.append(atoms)
        print(f"[centroid] pooled C{c}: global_frame={best} formula={formulas[best]}")
    if representatives:
        write(path, representatives)
        print(f"[save] centroids (pooled) -> {path}")


# ---------------------------------------------------------------------------
# Per-atom JSON for the visualizer
# ---------------------------------------------------------------------------


def build_atom_color_dict(k_Ir, k_O, ir_cmap_name, o_cmap_name):
    """Discrete colours for every atom cluster.

    Ir clusters are assumed numbered Ir-A0.. by ascending mean parent-structure
    O/Ir (blue -> red on ``ir_cmap_name``, typically coolwarm).
    """
    ir_palette = discrete_palette(ir_cmap_name, k_Ir)
    o_palette = discrete_palette(o_cmap_name, k_O)
    colors: dict[str, str] = {}
    for c, col in enumerate(ir_palette):
        colors[f"Ir-A{c}"] = to_hex(col)
    for c, col in enumerate(o_palette):
        colors[f"O-A{c}"] = to_hex(col)
    return colors


def load_motif_atoms_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def atom_rgbs_for_motif_tile(
    symbols: list[str],
    motif_labels: list[str],
    motif_json: dict,
    *,
    mode: str,
    struct_cluster: int,
    struct_color,
    neutral: str = "#d8d8d8",
) -> list:
    """RGB per atom for minima tiles: ``ir``, ``o``, ``cluster``, or ``atom`` colouring."""
    from matplotlib.colors import to_rgb

    palette = motif_json.get("colors", {})
    fallback = motif_json.get("fallback_colors", {})
    neutral_rgb = to_rgb(neutral)
    struct_rgb = to_rgb(struct_color)
    mode = mode.lower()
    if mode not in {"ir", "o", "cluster", "atom"}:
        raise ValueError(f"mode must be 'ir', 'o', 'cluster', or 'atom', got {mode!r}")

    if len(motif_labels) != len(symbols):
        motif_labels = ["_"] * len(symbols)

    rgbs = []
    for sym, mot in zip(symbols, motif_labels):
        if mode == "cluster":
            rgbs.append(struct_rgb)
            continue
        if mode == "atom":
            if mot in palette:
                rgbs.append(to_rgb(palette[mot]))
            elif sym in fallback:
                rgbs.append(to_rgb(fallback[sym]))
            else:
                rgbs.append(neutral_rgb)
            continue
        if mode == "ir":
            if sym == "Ir" and mot in palette:
                rgbs.append(to_rgb(palette[mot]))
            elif sym == "Ir" and sym in fallback:
                rgbs.append(to_rgb(fallback[sym]))
            else:
                rgbs.append(neutral_rgb)
            continue
        # mode == "o"
        if sym == "O" and mot in palette:
            rgbs.append(to_rgb(palette[mot]))
        elif sym == "O" and sym in fallback:
            rgbs.append(to_rgb(fallback[sym]))
        else:
            rgbs.append(neutral_rgb)
    return rgbs


def _per_atom_labels_for_dataset(d):
    """Build ``{frame_index: [label_or_'_']}`` for one dataset."""
    ir_per = d["per_struct_ir_cluster"]
    o_per = d["per_struct_o_cluster"]
    frames = {}
    for si, atoms in enumerate(d["structures"]):
        symbols = atoms.get_chemical_symbols()
        ir_labs = ir_per[si]
        o_labs = o_per[si]
        ir_p = 0
        o_p = 0
        per_atom = []
        for sym in symbols:
            if sym == "Ir":
                cid = int(ir_labs[ir_p]) if ir_p < len(ir_labs) else -1
                ir_p += 1
                per_atom.append(ace_atom_label("Ir", cid) if cid >= 0 else "_")
            elif sym == "O":
                cid = int(o_labs[o_p]) if o_p < len(o_labs) else -1
                o_p += 1
                per_atom.append(ace_atom_label("O", cid) if cid >= 0 else "_")
            else:
                per_atom.append("_")
        frames[str(si)] = per_atom
    return frames


def save_motif_atoms_pooled(
    datasets_loaded,
    k_Ir,
    k_O,
    ir_cmap_name,
    o_cmap_name,
    path,
):
    """Write ``motif_atoms_pooled.json`` for pooled trajectories (PBE then PBE0).

    Frame keys are **global pooled indices** (0 … N-1), matching
    ``atoms.info['frame_index']`` on ``ace_centroids_pooled.traj`` and
    ``ace_groups_pooled/C*.traj``.
    """
    colors = build_atom_color_dict(k_Ir, k_O, ir_cmap_name, o_cmap_name)
    frames = {}
    offset = 0
    for d in datasets_loaded:
        local = _per_atom_labels_for_dataset(d)
        for si, labels in local.items():
            frames[str(offset + int(si))] = labels
        offset += len(d["structures"])

    payload = {
        "label": "pooled",
        "colors": colors,
        "ir_motifs": [f"Ir-A{c}" for c in range(k_Ir)],
        "o_motifs": [f"O-A{c}" for c in range(k_O)],
        "fallback_colors": {
            "Ir": "#9d7ad9",
            "O": "#cc4444",
            "unassigned": "#888888",
        },
        "ir_regime_groups": None,
        "o_regime_groups": None,
        "frames": frames,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[save] ace atoms dict (pooled) -> {path}")


def save_ace_atoms_per_dataset(
    datasets_loaded,
    k_Ir,
    k_O,
    ir_cmap_name,
    o_cmap_name,
    path_template,
):
    """Write one ``motif_atoms_<label>.json`` per dataset.

    Same file format as the motif pipeline so the existing
    ``visualize_motifs.py`` renders it without modification.
    """
    colors = build_atom_color_dict(k_Ir, k_O, ir_cmap_name, o_cmap_name)
    ir_motifs = [f"Ir-A{c}" for c in range(k_Ir)]
    o_motifs = [f"O-A{c}" for c in range(k_O)]
    common = {
        "colors": colors,
        "ir_motifs": ir_motifs,
        "o_motifs": o_motifs,
        "fallback_colors": {
            "Ir": "#9d7ad9",
            "O": "#cc4444",
            "unassigned": "#888888",
        },
        "ir_regime_groups": None,
        "o_regime_groups": None,
    }
    written = []
    for d in datasets_loaded:
        label = d["label"]
        frames = _per_atom_labels_for_dataset(d)
        payload = {"label": label, **common, "frames": frames}
        path = path_template.format(label=label)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[save] ace atoms dict ({label}) -> {path}")
        written.append(path)
    return written


def copy_visualizer_script(dest_dir):
    """Copy ``visualize_motifs.py`` from ``aist/ace/`` next to the ACE outputs."""
    src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "visualize_motifs.py")
    dst = os.path.join(dest_dir, "visualize_motifs.py")
    shutil.copyfile(src, dst)
    print(f"[copy] visualize_motifs.py -> {dst}")
