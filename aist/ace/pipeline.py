"""ACE-style atom clustering across PBE and PBE0 datasets.

Pipeline
--------
Per-atom stage (Ir and O, separately, pooled across PBE + PBE0)
    SOAP (radial x angular per-atom descriptor, dscribe)
      -> StandardScaler
      -> PCA (keeps ``pca_variance_target`` of the variance, default 0.95)
      -> UMAP (2D, PCA init) -> KMeans (silhouette ``k`` over ``k_atom_range``)

Per-structure stage (PBE + PBE0 pooled)
    descriptor = per-cluster Ir-fraction + per-cluster O-fraction
      -> StandardScaler -> PCA (95%) -> KMeans (silhouette)
      -> clusters C0..CN ordered by oxidation; same labels on both functionals

Outputs (under ``analysis/ace_results/``):
   plots/
     ace_ir_atom_pca.svg, ace_o_atom_pca.svg     (2D PCA scatter, atoms)
     ace_struct_clustering_2d.svg                 (2D PCA scatter, structures)
     ace_composition_densities.svg                (n_O/n_Ir & n_Ir histograms)
     ace_cluster_composition.svg                  (stacked atom-cluster bars)
     ace_atom_pop_vs_oxidation.svg                (stacked pop % vs n_O/n_Ir)
     ace_struct_pop_vs_oxidation.svg              (PBE/PBE0 rows: C stacks vs n_O/n_Ir)
     ace_struct_cluster_pbe_split.svg             (PBE vs PBE0 % within each C)
   atoms/
     ace_features_<label>.csv
     ace_centroids_<label>.traj
     ace_groups_<label>/C*.traj
     motif_atoms_<label>.json, motif_atoms_pooled.json (for visualize_motifs.py)
     visualize_motifs.py                          (copy of the motif viewer)
"""

from __future__ import annotations

import os
import sys

import numpy as np
from ase.io import read


PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(PACKAGE_DIR))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from aist.ace import io as ace_io
from aist.ace import plotting as ace_plot
from aist.ace.clustering import (
    assign_atom_clusters_to_structures,
    cluster_atoms_umap_kmeans,
    cluster_structures_pca_kmeans,
    distribute_struct_labels_to_datasets,
    reorder_ir_atom_clusters_by_oxidation,
    reorder_struct_clusters_by_oxidation,
)
from aist.ace.common import composition_features_for_struct
from aist.ace.constants import ACE_RANDOM_STATE, SILHOUETTE_SAMPLE
from aist.ace.reproducibility import set_reproducible_seed
from aist.ace.descriptors import (
    compute_per_atom_descriptors,
    split_descriptors_by_species,
)


# ============================================================
# ======================= USER INPUT =========================
# ============================================================

# --- Datasets (atoms and structures both pooled across PBE + PBE0)
datasets = [
    {"label": "PBE", "traj": os.path.join(ROOT_DIR, "data/IryOxPBE.traj")},
    {"label": "PBE0", "traj": os.path.join(ROOT_DIR, "data/IryOxPBE0.traj")},
]

# --- ACE / SOAP descriptor (dscribe.SOAP = radial basis x spherical harmonics)
soap_species = ("Ir", "O")
soap_r_cut = 5.0      # neighbour cutoff (A)
soap_n_max = 8        # radial basis size
soap_l_max = 6        # angular basis size
soap_sigma = 0.5      # Gaussian width of atomic densities (A)

# --- PCA (always 95% variance retained as the user requested)
pca_variance_target = 0.95

# --- UMAP (atoms only) + KMeans silhouette k-selection
umap_n_components = 2
umap_n_neighbors = 30
umap_min_dist = 0.0
umap_metric = "euclidean"
k_atom_range = (2, 8)
k_struct_range = (2, 8)
silhouette_sample = SILHOUETTE_SAMPLE
random_state = ACE_RANDOM_STATE

standardize_atom_features = True
standardize_struct_features = True

# --- Discrete colormaps
ir_atom_cmap = "coolwarm"
o_atom_cmap = "plasma_r"

# --- Atom population vs oxidation stack plot (smoothing)
pop_vs_oxidation_n_bins = 100       # regular grid points along n_O/n_Ir
pop_vs_oxidation_bandwidth = None   # None -> auto from data span / n_bins
pop_vs_oxidation_smooth_sigma = 1.0 # extra 1D Gaussian smooth along x (0 = off)

# --- Output paths
RESULTS_DIR = os.path.join(ROOT_DIR, "analysis", "ace_results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
ATOMS_DIR = os.path.join(RESULTS_DIR, "atoms")

output_ir_atom_pca = os.path.join(PLOTS_DIR, "ace_ir_atom_pca.svg")
output_o_atom_pca = os.path.join(PLOTS_DIR, "ace_o_atom_pca.svg")
output_struct_plot = os.path.join(PLOTS_DIR, "ace_struct_clustering_2d.svg")
output_composition_plot = os.path.join(PLOTS_DIR, "ace_composition_densities.svg")
output_struct_composition_plot = os.path.join(PLOTS_DIR, "ace_cluster_composition.svg")
output_atom_pop_oxidation_plot = os.path.join(PLOTS_DIR, "ace_atom_pop_vs_oxidation.svg")
output_struct_pop_oxidation_plot = os.path.join(PLOTS_DIR, "ace_struct_pop_vs_oxidation.svg")
output_struct_pbe_split_plot = os.path.join(PLOTS_DIR, "ace_struct_cluster_pbe_split.svg")

output_features_csv_template = os.path.join(ATOMS_DIR, "ace_features_{label}.csv")
output_centroid_traj_template = os.path.join(ATOMS_DIR, "ace_centroids_{label}.traj")
output_groups_dir_template = os.path.join(ATOMS_DIR, "ace_groups_{label}")
output_centroid_traj_pooled = os.path.join(ATOMS_DIR, "ace_centroids_pooled.traj")
output_groups_dir_pooled = os.path.join(ATOMS_DIR, "ace_groups_pooled")
# Per-dataset JSON for visualize_motifs; pooled JSON for ace_centroids_pooled.traj / ace_groups_pooled/
output_ace_atoms_json_template = os.path.join(ATOMS_DIR, "motif_atoms_{label}.json")
output_ace_atoms_json_pooled = os.path.join(ATOMS_DIR, "motif_atoms_pooled.json")

# ============================================================
# ===================== END USER INPUT =======================

def set_data_dir(data_dir: str) -> None:
    """Point default datasets at ``<repo>/data``."""
    global datasets
    datasets = [
        {"label": "PBE", "traj": os.path.join(data_dir, "IryOxPBE.traj")},
        {"label": "PBE0", "traj": os.path.join(data_dir, "IryOxPBE0.traj")},
    ]
# ============================================================


def configure_outputs(results_dir: str) -> None:
    """Redirect all ACE pipeline outputs under ``results_dir`` (e.g. figures/cache/ace)."""
    global RESULTS_DIR, PLOTS_DIR, ATOMS_DIR
    global output_ir_atom_pca, output_o_atom_pca, output_struct_plot
    global output_composition_plot, output_struct_composition_plot
    global output_atom_pop_oxidation_plot, output_struct_pop_oxidation_plot
    global output_struct_pbe_split_plot
    global output_features_csv_template, output_centroid_traj_template
    global output_groups_dir_template, output_centroid_traj_pooled
    global output_groups_dir_pooled, output_ace_atoms_json_template
    global output_ace_atoms_json_pooled

    RESULTS_DIR = results_dir
    PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
    ATOMS_DIR = os.path.join(RESULTS_DIR, "atoms")
    output_ir_atom_pca = os.path.join(PLOTS_DIR, "ace_ir_atom_pca.svg")
    output_o_atom_pca = os.path.join(PLOTS_DIR, "ace_o_atom_pca.svg")
    output_struct_plot = os.path.join(PLOTS_DIR, "ace_struct_clustering_2d.svg")
    output_composition_plot = os.path.join(PLOTS_DIR, "ace_composition_densities.svg")
    output_struct_composition_plot = os.path.join(PLOTS_DIR, "ace_cluster_composition.svg")
    output_atom_pop_oxidation_plot = os.path.join(PLOTS_DIR, "ace_atom_pop_vs_oxidation.svg")
    output_struct_pop_oxidation_plot = os.path.join(PLOTS_DIR, "ace_struct_pop_vs_oxidation.svg")
    output_struct_pbe_split_plot = os.path.join(PLOTS_DIR, "ace_struct_cluster_pbe_split.svg")
    output_features_csv_template = os.path.join(ATOMS_DIR, "ace_features_{label}.csv")
    output_centroid_traj_template = os.path.join(ATOMS_DIR, "ace_centroids_{label}.traj")
    output_groups_dir_template = os.path.join(ATOMS_DIR, "ace_groups_{label}")
    output_centroid_traj_pooled = os.path.join(ATOMS_DIR, "ace_centroids_pooled.traj")
    output_groups_dir_pooled = os.path.join(ATOMS_DIR, "ace_groups_pooled")
    output_ace_atoms_json_template = os.path.join(ATOMS_DIR, "motif_atoms_{label}.json")
    output_ace_atoms_json_pooled = os.path.join(ATOMS_DIR, "motif_atoms_pooled.json")


def _count_species(structures, symbol):
    return np.array(
        [int(np.sum(np.asarray(a.get_chemical_symbols()) == symbol)) for a in structures],
        dtype=int,
    )


def load_dataset(label, path):
    structures = list(read(path, ":"))
    return {
        "label": label,
        "structures": structures,
        "formulas": [a.get_chemical_formula() for a in structures],
        "n_Ir": _count_species(structures, "Ir"),
        "n_O": _count_species(structures, "O"),
    }


def _stamp_dataset_in_info(d):
    """Add ``atoms.info['dataset']`` so the visualizer auto-detects the JSON."""
    for atoms in d["structures"]:
        info = dict(atoms.info)
        info.setdefault("dataset", d["label"])
        atoms.info = info


def main(
    results_dir: str | None = None,
    data_dir: str | None = None,
    *,
    datasets_override: list[dict] | None = None,
    write_plots: bool = True,
) -> None:
    global datasets
    if data_dir is not None:
        set_data_dir(data_dir)
    if datasets_override is not None:
        datasets = datasets_override
    if results_dir is not None:
        configure_outputs(results_dir)
    set_reproducible_seed(random_state)
    print(f"[repro] random_state={random_state}  silhouette_sample={silhouette_sample}  threads=1")
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(ATOMS_DIR, exist_ok=True)

    # --- Load datasets
    datasets_loaded = []
    for ds in datasets:
        if not os.path.exists(ds["traj"]):
            print(f"[skip] missing {ds['traj']!r}")
            continue
        d = load_dataset(ds["label"], ds["traj"])
        _stamp_dataset_in_info(d)
        datasets_loaded.append(d)
    if not datasets_loaded:
        raise SystemExit("No datasets loaded.")

    print(
        f"[soap] r_cut={soap_r_cut} A  n_max={soap_n_max}  l_max={soap_l_max}  "
        f"sigma={soap_sigma}  species={soap_species}"
    )

    # --- Per-atom SOAP descriptors
    for d in datasets_loaded:
        d["descriptors"] = compute_per_atom_descriptors(
            d["structures"], soap_species,
            r_cut=soap_r_cut, n_max=soap_n_max, l_max=soap_l_max, sigma=soap_sigma,
        )
        dim = d["descriptors"][0].shape[1] if d["descriptors"] else 0
        n_ir = int(d["n_Ir"].sum())
        n_o = int(d["n_O"].sum())
        print(f"[descriptors] {d['label']}: n_Ir={n_ir}  n_O={n_o}  D={dim}")

    # --- Pool Ir and O atoms across all datasets (with provenance)
    ir_rows = []
    ir_prov = []
    o_rows = []
    o_prov = []
    offset = 0
    pooled_structures = []
    pooled_n_O = []
    pooled_n_Ir = []
    for d in datasets_loaded:
        X_ir_d, prov_ir_d = split_descriptors_by_species(d["structures"], d["descriptors"], "Ir")
        X_o_d, prov_o_d = split_descriptors_by_species(d["structures"], d["descriptors"], "O")
        ir_rows.append(X_ir_d)
        o_rows.append(X_o_d)
        ir_prov.extend((offset + si, ai) for (si, ai) in prov_ir_d)
        o_prov.extend((offset + si, ai) for (si, ai) in prov_o_d)
        pooled_structures.extend(d["structures"])
        pooled_n_O.extend(d["n_O"].tolist())
        pooled_n_Ir.extend(d["n_Ir"].tolist())
        offset += len(d["structures"])
    pooled_n_O = np.asarray(pooled_n_O, dtype=float)
    pooled_n_Ir = np.asarray(pooled_n_Ir, dtype=float)
    X_ir = np.vstack([r for r in ir_rows if r.size]) if any(r.size for r in ir_rows) else np.zeros((0, 0))
    X_o = np.vstack([r for r in o_rows if r.size]) if any(r.size for r in o_rows) else np.zeros((0, 0))

    # --- Atom clustering: PCA -> UMAP -> silhouette-selected KMeans
    ir_fit = cluster_atoms_umap_kmeans(
        X_ir,
        pca_variance_target=pca_variance_target,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        k_min=k_atom_range[0], k_max=k_atom_range[1],
        standardize=standardize_atom_features,
        random_state=random_state,
        sil_sample=silhouette_sample,
        label_for_print="Ir-atom",
    )
    o_fit = cluster_atoms_umap_kmeans(
        X_o,
        pca_variance_target=pca_variance_target,
        umap_n_components=umap_n_components,
        umap_n_neighbors=umap_n_neighbors,
        umap_min_dist=umap_min_dist,
        umap_metric=umap_metric,
        k_min=k_atom_range[0], k_max=k_atom_range[1],
        standardize=standardize_atom_features,
        random_state=random_state,
        sil_sample=silhouette_sample,
        label_for_print="O-atom",
    )

    # --- Distribute pooled atom labels back to per-structure per-atom lists
    reorder_ir_atom_clusters_by_oxidation(
        ir_fit, ir_prov, pooled_n_O, pooled_n_Ir, label_for_print="Ir-atom",
    )
    per_struct_ir_pooled = assign_atom_clusters_to_structures(
        pooled_structures, ir_prov, ir_fit["labels"], "Ir",
    )
    per_struct_o_pooled = assign_atom_clusters_to_structures(
        pooled_structures, o_prov, o_fit["labels"], "O",
    )
    offset = 0
    for d in datasets_loaded:
        n = len(d["structures"])
        d["per_struct_ir_cluster"] = per_struct_ir_pooled[offset: offset + n]
        d["per_struct_o_cluster"] = per_struct_o_pooled[offset: offset + n]
        offset += n

    # --- Per-structure descriptor: per-cluster Ir + O fractions
    for d in datasets_loaded:
        d["ir_cluster_fractions"] = composition_features_for_struct(
            len(d["structures"]), d["per_struct_ir_cluster"], "Ir", ir_fit["k"],
        )
        d["o_cluster_fractions"] = composition_features_for_struct(
            len(d["structures"]), d["per_struct_o_cluster"], "O", o_fit["k"],
        )

    # --- Pooled structure clustering (PBE + PBE0): PCA(95%) + KMeans(silhouette)
    struct_blocks = []
    for d in datasets_loaded:
        struct_blocks.append(np.hstack([d["ir_cluster_fractions"], d["o_cluster_fractions"]]))
    X_struct_pooled = np.vstack(struct_blocks)
    if X_struct_pooled.shape[0] < 5:
        raise SystemExit("Too few structures for pooled structure clustering.")

    struct_fit = cluster_structures_pca_kmeans(
        X_struct_pooled,
        pca_variance_target=pca_variance_target,
        k_min=k_struct_range[0], k_max=k_struct_range[1],
        standardize=standardize_struct_features,
        random_state=random_state,
        sil_sample=silhouette_sample,
        label_for_print="struct-pooled",
    )
    struct_result = {"fit": struct_fit}
    reorder_struct_clusters_by_oxidation(
        struct_result, pooled_n_O, pooled_n_Ir, label_for_print="struct-pooled",
    )
    distribute_struct_labels_to_datasets(datasets_loaded, struct_fit["labels"])
    offset = 0
    for d in datasets_loaded:
        n = len(d["structures"])
        d["struct_x_plot"] = struct_fit["X_plot"][offset: offset + n]
        d["struct_x_pca"] = struct_fit["X_pca"][offset: offset + n]
        offset += n

    # --- Numeric cache (NPZ) for downstream figure plotting
    ace_io.save_atom_embedding_npz(
        ir_fit, os.path.join(RESULTS_DIR, "ace_atom_ir.npz"),
    )
    ace_io.save_atom_embedding_npz(
        o_fit, os.path.join(RESULTS_DIR, "ace_atom_o.npz"),
    )
    ace_io.save_struct_embeddings_npz(
        datasets_loaded, struct_fit,
        os.path.join(RESULTS_DIR, "ace_struct_embeddings.npz"),
    )

    if write_plots:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        ace_plot.plot_atom_scatter(ir_fit, "Ir", ir_atom_cmap, output_ir_atom_pca)
        ace_plot.plot_atom_scatter(o_fit, "O", o_atom_cmap, output_o_atom_pca)
        ace_plot.plot_struct_pca(datasets_loaded, struct_fit, output_struct_plot)
        ace_plot.plot_composition_densities(
            datasets_loaded, struct_fit, output_composition_plot,
        )
        ace_plot.plot_struct_cluster_composition(
            datasets_loaded, struct_fit, ir_fit, o_fit,
            output_struct_composition_plot,
            ir_cmap_name=ir_atom_cmap, o_cmap_name=o_atom_cmap,
        )
        ace_plot.plot_atom_population_vs_oxidation(
            datasets_loaded, ir_fit, o_fit,
            output_atom_pop_oxidation_plot,
            ir_cmap_name=ir_atom_cmap, o_cmap_name=o_atom_cmap,
            n_bins=pop_vs_oxidation_n_bins,
            bandwidth=pop_vs_oxidation_bandwidth,
            smooth_sigma=pop_vs_oxidation_smooth_sigma,
        )
        ace_plot.plot_struct_population_vs_oxidation(
            datasets_loaded, struct_fit,
            output_struct_pop_oxidation_plot,
            n_bins=pop_vs_oxidation_n_bins,
            bandwidth=pop_vs_oxidation_bandwidth,
            smooth_sigma=pop_vs_oxidation_smooth_sigma,
        )
        ace_plot.plot_struct_cluster_dataset_split(
            datasets_loaded, struct_fit, output_struct_pbe_split_plot,
        )

    # --- CSV / trajectory / JSON outputs
    for d in datasets_loaded:
        ace_io.save_features_csv(
            d, struct_fit, output_features_csv_template.format(label=d["label"]),
        )
        ace_io.save_centroid_traj(
            d, struct_fit, output_centroid_traj_template.format(label=d["label"]),
        )
        ace_io.save_group_trajs(
            d, struct_fit, output_groups_dir_template.format(label=d["label"]),
        )
    ace_io.save_pooled_centroid_traj(
        datasets_loaded, struct_fit, output_centroid_traj_pooled,
    )
    ace_io.save_pooled_group_trajs(
        datasets_loaded, struct_fit, output_groups_dir_pooled,
    )

    ace_io.save_ace_atoms_per_dataset(
        datasets_loaded, ir_fit["k"], o_fit["k"],
        ir_atom_cmap, o_atom_cmap,
        output_ace_atoms_json_template,
    )
    ace_io.save_motif_atoms_pooled(
        datasets_loaded, ir_fit["k"], o_fit["k"],
        ir_atom_cmap, o_atom_cmap,
        output_ace_atoms_json_pooled,
    )
    ace_io.copy_visualizer_script(ATOMS_DIR)

    print("\n[done] ACE clustering pipeline complete.")


if __name__ == "__main__":
    import argparse

    _parser = argparse.ArgumentParser(description="ACE clustering pipeline")
    _parser.add_argument(
        "--results-dir",
        default=None,
        help="Output directory (default: analysis/ace_results)",
    )
    _args = _parser.parse_args()
    main(results_dir=_args.results_dir)
