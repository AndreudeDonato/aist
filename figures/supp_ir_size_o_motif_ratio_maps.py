"""Supplementary – Terminal / bridging O per Ir phase maps (PBE | PBE0).

Two outputs (7 rows × 2 columns, Ir₃…Ir₉ × PBE | PBE0), matching fig04 phase-map layout but
restricted to one ``n_Ir`` bucket per row:

  ``supp_ir_size_terminal_o_maps.svg``  — ⟨n_O^{terminal} / n_Ir⟩
  ``supp_ir_size_bridging_o_maps.svg`` — ⟨n_O^{bridging} / n_Ir⟩

Field: ``Σ_i Pgi_i · n_O,i · f_{motif,i} / Σ_i Pgi_i · n_Ir,i`` on the GC grid.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

from figures.config import (
    ACE_FEATURES,
    DPI,
    FIG04_O_BRIDGING_IDX,
    FIG04_O_TERMINAL_IDX,
    GC_NPZ,
    GC_NPZ_PBE0,
    OUTPUT_DIR,
    PHASE_CMAP,
)
from figures.utils.cache import ensure_for_supplementary
from figures.utils.gc_subset import subset_joined_table_by_n_ir
from figures.utils.io import ensure_dirs, load_ace_features, load_gc_npz
from figures.utils.join import build_joined_table, ensemble_o_motif_per_ir_map
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    add_phase_colorbar,
    margin_colorbar_cax,
    save_fig,
    style_ir_size_panel_ax,
)

FIG_W_MM = 180
ROW_H_MM = 32
FIG_H_MM = 7 * ROW_H_MM
IR_COUNTS = list(range(3, 10))
LAYOUT_RECT = [0, 0, 0.84, 1]
CBAR_GAP = 0.01
CBAR_WIDTH = 0.012

DATASETS = [
    {"label": "PBE", "npz": GC_NPZ, "ace_csv": ACE_FEATURES["PBE"]},
    {"label": "PBE0", "npz": GC_NPZ_PBE0, "ace_csv": ACE_FEATURES["PBE0"]},
]

MAP_SPECS = (
    (FIG04_O_TERMINAL_IDX, "supp_ir_size_terminal_o_maps.svg"),
    (FIG04_O_BRIDGING_IDX, "supp_ir_size_bridging_o_maps.svg"),
)


@dataclass
class DatasetBundle:
    label: str
    jt: dict


def _load_dataset(ds: dict) -> DatasetBundle | None:
    if not os.path.exists(ds["npz"]):
        return None
    gc = load_gc_npz(ds["npz"])
    rows = load_ace_features(ds["ace_csv"])
    return DatasetBundle(label=ds["label"], jt=build_joined_table(gc, rows, ds["label"]))


def _field_limits(panels: list[dict]) -> tuple[float, float]:
    vals: list[float] = []
    for panel in panels:
        Z = np.asarray(panel["field"], dtype=float)
        finite = Z[np.isfinite(Z)]
        if finite.size:
            vals.extend([float(finite.min()), float(finite.max())])
    if not vals:
        return 0.0, 1.0
    vmin, vmax = float(min(vals)), float(max(vals))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _panel_payload(bundle: DatasetBundle, n_ir: int, o_idx: int) -> dict | None:
    jt_sub = subset_joined_table_by_n_ir(bundle.jt, n_ir)
    if jt_sub is None:
        return None
    return {
        "Ts": jt_sub["Ts"],
        "Ps": jt_sub["Ps"],
        "field": ensemble_o_motif_per_ir_map(jt_sub, o_idx),
    }


def _render_map(
    o_idx: int,
    filename: str,
    bundles: dict[str, DatasetBundle],
) -> None:
    grid: dict[tuple[int, str], dict] = {}
    for label, bundle in bundles.items():
        for n_ir in IR_COUNTS:
            panel = _panel_payload(bundle, n_ir, o_idx)
            if panel is not None:
                grid[(n_ir, label)] = panel

    if not grid:
        raise RuntimeError(f"[supp] No Ir-size panels for o_idx={o_idx}")

    vmin, vmax = _field_limits(list(grid.values()))

    fig, axes = plt.subplots(
        len(IR_COUNTS), 2,
        figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4),
        dpi=DPI,
        sharex=True,
        sharey=True,
    )

    mesh_ref = None
    last_row = len(IR_COUNTS) - 1
    for row, n_ir in enumerate(IR_COUNTS):
        for col, label in enumerate(("PBE", "PBE0")):
            ax = axes[row, col]
            panel = grid.get((n_ir, label))
            if panel is None:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        fontsize=AXIS_LABEL_FONTSIZE, transform=ax.transAxes)
                ax.set_axis_off()
                continue

            Ts, Ps = panel["Ts"], panel["Ps"]
            Z = np.asarray(panel["field"], dtype=float)
            mesh = ax.pcolormesh(
                Ts, Ps, Z, shading="auto",
                cmap=PHASE_CMAP,
                norm=Normalize(vmin=vmin, vmax=vmax),
                rasterized=True,
            )
            if mesh_ref is None:
                mesh_ref = mesh

            title = label if row == 0 else ""
            style_ir_size_panel_ax(ax, n_ir, col, row, last_row, title=title)

    fig.tight_layout(pad=6 / 25.4, rect=LAYOUT_RECT)
    if mesh_ref is not None:
        add_phase_colorbar(
            fig, mesh_ref,
            cax=margin_colorbar_cax(
                fig, axes[0, 0], axes[-1, 1],
                gap=CBAR_GAP, width=CBAR_WIDTH,
            ),
            label=r"$n_{\mathrm{O}}^{\mathrm{motif}} / n_{\mathrm{Ir}}$",
        )

    out_path = os.path.join(OUTPUT_DIR, filename)
    save_fig(fig, out_path, dpi=DPI)
    print(f"[supp] {out_path}")


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_o_motif_ratio_maps"])
    ensure_dirs(OUTPUT_DIR)

    bundles: dict[str, DatasetBundle] = {}
    for ds in DATASETS:
        bundle = _load_dataset(ds)
        if bundle is not None:
            bundles[bundle.label] = bundle

    if not bundles:
        raise FileNotFoundError("[supp] No GC data found for Ir-size O motif maps.")

    for o_idx, filename in MAP_SPECS:
        _render_map(o_idx, filename, bundles)


if __name__ == "__main__":
    main()
