"""PBE | PBE0 comparison views (incl. gas oxides), at the level set by ``GC_COMPARISON_LEVEL``."""
from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from ase.io import read

from figures.config import (
    ACE_STRUCT_NPZ,
    GC_COMPARISON_LEVEL,
    GC_INCLUDE_GAS_RESERVOIR,
    GC_NPZ_BY_DATASET,
    GC_VARIANTS_NPZ_BY_DATASET,
    TRAJ_PBE,
    TRAJ_PBE0,
    gc_gas_reservoir_specs,
)
from figures.utils.homo_lumo import homo_lumo_gaps
from figures.utils.io import load_ace_features, load_gc_npz, load_gc_pbe_novib_view
from figures.utils.join import build_joined_table, ensemble_weighted_rgb_map
from aist.ace.constants import ACE_EXTENT_BLEND, ACE_EXTENT_NORMALIZATION
from aist.ace.join import align_ace_rows_to_gc
from aist.inertia import inertia_shape_extent_features, per_structure_shape_extent_rgb
from aist.thermo import ThermoPartitionFunctions

GAS_CLUSTER_LABEL = "GAS"
GAS_CLUSTER_COLOR = "#ffffff"
GAS_RGB = np.array([1.0, 1.0, 1.0], dtype=float)
GAS_FORMULAS = frozenset({"IrO2", "IrO3"})


@lru_cache(maxsize=None)
def ace_structure_cluster_count() -> int:
    """Number of structure clusters k of the ACE clustering (not the highest label present in a subset)."""
    return int(np.load(ACE_STRUCT_NPZ)["k"])


def gas_structure_mask(formulas) -> np.ndarray:
    """Boolean mask for volatile IrO₂ / IrO₃ GC structures."""
    return np.array(
        [str(f) in GAS_FORMULAS for f in np.asarray(formulas, dtype=object)],
        dtype=bool,
    )


def load_gc_comparison_view(dataset: str, *, light: bool = False) -> dict:
    """PBE or PBE0 grand-canonical view keyed like ``load_gc_npz``: with vibrations ("full", the
    grand-canonical NPZ) or without ("novib", from that dataset's variants NPZ), per GC_COMPARISON_LEVEL."""
    if GC_COMPARISON_LEVEL == "full":
        return load_gc_npz(GC_NPZ_BY_DATASET[dataset], light=light)
    return load_gc_pbe_novib_view(GC_VARIANTS_NPZ_BY_DATASET[dataset], GC_NPZ_BY_DATASET[dataset])


def count_gas_structures(dataset: str) -> int:
    return sum(1 for path, _ in gc_gas_reservoir_specs(dataset) if os.path.exists(path))


@lru_cache(maxsize=None)
def pooled_log_extent_range() -> tuple[float, float]:
    """(min, max) ``log_extent`` over the PBE and PBE0 trajectories: one inertia colour scale for both
    functionals, as in the pooled ACE clustering."""
    tp = ThermoPartitionFunctions()
    vals = [
        inertia_shape_extent_features(a, tp, extent_normalization=ACE_EXTENT_NORMALIZATION,
                                      extent_blend=ACE_EXTENT_BLEND)[2]
        for path in (TRAJ_PBE, TRAJ_PBE0) for a in read(path, ":")
    ]
    return float(min(vals)), float(max(vals))


def inertia_rgb_rows(structures, frame_index) -> np.ndarray:
    """Shape × extent RGB of every trajectory structure on the pooled PBE+PBE0 scale, in joined-table row
    order (``jt["frame_index"]``; gas rows NaN)."""
    rgb = per_structure_shape_extent_rgb(
        structures, ThermoPartitionFunctions(),
        extent_normalization=ACE_EXTENT_NORMALIZATION,
        extent_blend=ACE_EXTENT_BLEND,
        extent_range=pooled_log_extent_range(),
    )
    return per_frame_rows(rgb, frame_index)


def per_frame_rows(values, frame_index) -> np.ndarray:
    """Rows of a per-trajectory-frame array in joined-table order (``jt["frame_index"]``); gas rows -> NaN."""
    values = np.asarray(values, dtype=float)
    frame_index = np.asarray(frame_index, dtype=int)
    out = np.full((frame_index.size,) + values.shape[1:], np.nan)
    ok = frame_index >= 0
    out[ok] = values[frame_index[ok]]
    return out


def homo_lumo_gaps_for_comparison(dataset: str, traj_path: str, frame_index) -> np.ndarray:
    """Per-structure HOMO–LUMO gaps in joined-table row order (``jt["frame_index"]``).

    Structures are picked by trajectory frame; gas rows (-1) take the gas-oxide gaps in order."""
    struct_gaps = list(homo_lumo_gaps(list(read(traj_path, ":"))))
    gas_gaps = []
    for path, _ in gc_gas_reservoir_specs(dataset):
        if not os.path.exists(path):
            continue
        mol = read(path)
        atoms = mol[0] if isinstance(mol, list) else mol
        gas_gaps.append(float(atoms.info.get("HOMO-LUMO", np.nan)))
    gas_iter = iter(gas_gaps)
    return np.array([struct_gaps[f] if f >= 0 else next(gas_iter, np.nan) for f in np.asarray(frame_index, dtype=int)],
                    dtype=float)


def _synthetic_gas_ace_row(
    formula: str,
    template: dict,
    *,
    gas_cluster_id: int,
) -> dict:
    n_Ir = 1
    n_O = 2 if formula == "IrO2" else 3 if formula == "IrO3" else 0
    row: dict = {}
    for key, val in template.items():
        if key.endswith("_frac"):
            row[key] = 0.0
        elif key == "frame_index":
            row[key] = -1
        elif key == "formula":
            row[key] = formula
        elif key == "n_Ir":
            row[key] = n_Ir
        elif key == "n_O":
            row[key] = n_O
        elif key == "cluster":
            row[key] = gas_cluster_id
        else:
            row[key] = val
    return row


def apply_gas_atom_motifs(jt: dict) -> None:
    """Append a white ``GAS`` atom-motif column for volatile IrO₂ / IrO₃ structures."""
    gas_mask = gas_structure_mask(jt["formulas_gc"])
    if not np.any(gas_mask):
        return
    ir = np.asarray(jt["ir_frac"], dtype=float)
    o = np.asarray(jt["o_frac"], dtype=float)
    n = ir.shape[0]
    ir_out = np.zeros((n, ir.shape[1] + 1), dtype=float)
    o_out = np.zeros((n, o.shape[1] + 1), dtype=float)
    ir_out[:, :-1] = np.where(gas_mask[:, None], 0.0, ir)
    o_out[:, :-1] = np.where(gas_mask[:, None], 0.0, o)
    ir_out[gas_mask, -1] = 1.0
    o_out[gas_mask, -1] = 1.0
    jt["ir_frac"] = ir_out
    jt["o_frac"] = o_out
    jt["has_gas_atom_motif"] = True


def atom_motif_colors_labels(jt: dict, species: str) -> tuple[list[str], list[str]]:
    """ACE Ir/O motif palette plus optional trailing white ``GAS`` atom type."""
    from aist.ace.common import ace_atom_display_label
    from figures.utils.plotting import ir_motif_colors, o_motif_colors

    frac_key = "ir_frac" if species == "Ir" else "o_frac"
    k = int(jt[frac_key].shape[1])
    has_gas = bool(jt.get("has_gas_atom_motif"))
    k_ace = k - 1 if has_gas else k
    palette = ir_motif_colors if species == "Ir" else o_motif_colors
    colors = list(palette(k_ace))
    labels = [ace_atom_display_label(species, c) for c in range(k_ace)]
    if has_gas:
        colors.append(GAS_CLUSTER_COLOR)
        labels.append(GAS_CLUSTER_LABEL)
    return colors, labels


def atom_motif_population_stack_unified_gas(
    joined: dict,
    pressure_pa: float,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray, list[str], list[str], int]:
    """Fig. 4 layout: O motifs | Ir motifs | one combined GAS band.

    Gas reservoir structures contribute a single white ``GAS`` row instead of
    separate Ir and O gas columns.
    """
    from aist.ace.join import atom_motif_population_vs_T_at_pressure

    Ts, p_used, Y = atom_motif_population_vs_T_at_pressure(
        joined, pressure_pa, smooth_sigma=smooth_sigma,
    )
    k_ir = int(joined["ir_frac"].shape[1])
    Y_ir = Y[:k_ir]
    Y_o = Y[k_ir:]
    o_colors, o_labels = atom_motif_colors_labels(joined, "O")
    ir_colors, ir_labels = atom_motif_colors_labels(joined, "Ir")

    if joined.get("has_gas_atom_motif"):
        Y_gas = Y_ir[-1:] + Y_o[-1:]
        Y_o = Y_o[:-1]
        Y_ir = Y_ir[:-1]
        o_colors, o_labels = o_colors[:-1], o_labels[:-1]
        ir_colors, ir_labels = ir_colors[:-1], ir_labels[:-1]
        Y_plot = np.vstack([Y_o, Y_ir, Y_gas])
        colors = o_colors + ir_colors + [GAS_CLUSTER_COLOR]
        labels = o_labels + ir_labels + [GAS_CLUSTER_LABEL]
    else:
        Y_plot = np.vstack([Y_o, Y_ir])
        colors = o_colors + ir_colors
        labels = o_labels + ir_labels

    return Ts, p_used, Y_plot, colors, labels, int(Y_o.shape[0])


def atom_motif_population_stack_by_ir_size(
    joined: dict,
    n_ir: int,
    pressure_pa: float,
    *,
    smooth_sigma: float = 0.0,
) -> tuple[np.ndarray, float, np.ndarray, list[str], list[str], int] | None:
    """Fig. 4 per-``n_Ir`` row: O | Ir | GAS atom-motif stack at fixed ``pO₂``."""
    from figures.utils.gc_subset import subset_joined_table_by_n_ir

    jt_sub = subset_joined_table_by_n_ir(joined, n_ir)
    if jt_sub is None:
        return None
    if joined.get("has_gas_atom_motif"):
        jt_sub["has_gas_atom_motif"] = True
    return atom_motif_population_stack_unified_gas(
        jt_sub, pressure_pa, smooth_sigma=smooth_sigma,
    )


def gas_population_fraction(Pgi: np.ndarray, gas_mask: np.ndarray) -> np.ndarray:
    """Ensemble gas-reservoir fraction on the GC grid, shape ``(nP, nT)``."""
    P = np.asarray(Pgi, dtype=float)
    return P[np.asarray(gas_mask, dtype=bool)].sum(axis=0)


def ensemble_shape_rgb_map_with_gas(
    Pgi: np.ndarray,
    per_structure_rgb: np.ndarray,
    gas_mask: np.ndarray,
) -> np.ndarray:
    """Population-weighted shape RGB; gas reservoir pixels mix in as white."""
    gas_mask = np.asarray(gas_mask, dtype=bool)
    rgb = np.asarray(per_structure_rgb, dtype=float).copy()
    rgb[gas_mask] = GAS_RGB
    bad = ~np.all(np.isfinite(rgb), axis=1)
    rgb[bad] = 0.0
    return ensemble_weighted_rgb_map(Pgi, rgb)


def scalar_field_rgb_with_gas(
    field: np.ndarray,
    f_gas: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
) -> np.ndarray:
    """Colormap scalar field with white mixed in proportional to gas population."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    Z = np.asarray(field, dtype=float)
    f = np.clip(np.asarray(f_gas, dtype=float), 0.0, 1.0)
    norm = Normalize(vmin=vmin, vmax=vmax)
    rgb = plt.get_cmap(cmap)(norm(Z))[..., :3]
    return np.clip(rgb * (1.0 - f)[..., None] + f[..., None], 0.0, 1.0)


def structure_cluster_colors_labels(
    joined: dict,
    *,
    n_pop: int | None = None,
) -> tuple[list[str], list[str]]:
    """C0…C* palette plus optional white ``GAS`` entry for volatile oxides."""
    from figures.utils.plotting import cluster_colors

    n_ace = int(joined.get("n_ace_clusters", int(np.max(joined["cluster"])) + 1))
    colors = list(cluster_colors(n_ace))
    labels = [f"C{c}" for c in range(n_ace)]
    if joined.get("gas_cluster_id") is not None:
        colors.append(GAS_CLUSTER_COLOR)
        labels.append(GAS_CLUSTER_LABEL)
    if n_pop is None:
        n_pop = len(colors)
    return colors[:n_pop], labels[:n_pop]


def build_comparison_joined_table(
    gc_data: dict,
    ace_rows: list[dict],
    dataset: str,
    *,
    include_gas_reservoir: bool | None = None,
) -> dict:
    """Join GC comparison data with ACE rows, synthesising rows for GC-only gas oxides."""
    if include_gas_reservoir is None:
        include_gas_reservoir = GC_INCLUDE_GAS_RESERVOIR
    ace_rows = align_ace_rows_to_gc(gc_data, ace_rows)
    n_gc = gc_data["Pgi"].shape[0]
    n_ace = len(ace_rows)
    n_ace_clusters = ace_structure_cluster_count()
    gas_cluster_id = None
    if include_gas_reservoir and n_gc > n_ace:
        gas_cluster_id = n_ace_clusters
        formulas = [str(f) for f in gc_data["formulas"][n_ace:n_gc]]
        extra = [
            _synthetic_gas_ace_row(
                f,
                ace_rows[0],
                gas_cluster_id=gas_cluster_id,
            )
            for f in formulas
        ]
        ace_rows = list(ace_rows) + extra
    jt = build_joined_table(gc_data, ace_rows, dataset)
    jt["n_ace_clusters"] = n_ace_clusters
    jt["gas_cluster_id"] = gas_cluster_id
    if gas_cluster_id is not None:
        apply_gas_atom_motifs(jt)
    return jt


def load_comparison_joined(dataset: str, ace_csv: str) -> dict:
    gc = load_gc_comparison_view(dataset)
    rows = load_ace_features(ace_csv)
    return build_comparison_joined_table(gc, rows, dataset)
