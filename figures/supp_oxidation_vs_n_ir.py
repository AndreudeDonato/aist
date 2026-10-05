"""Supplementary – Ensemble oxidation ⟨n_O/n_Ir⟩ vs Ir size at the four standard conditions, PBE | PBE0.

1×2 layout (180 mm wide): x = n_Ir (THERMO_IR_COUNTS), y = ⟨n_O/n_Ir⟩ of that size's grand-canonical
ensemble at each (T, pO₂) of ``STANDARD_CONDITIONS`` (colours; nearest grid point). Solid lines: with
vibrations (full), dashed: without (novib). Both functionals share the y axis.

Outputs: ``figures/outputs/supp_oxidation_vs_n_ir.svg`` and the numbers in ``supp_oxidation_vs_n_ir.tsv``.
"""
from __future__ import annotations

import os
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from aist.core.grandcanonical import pressure_grid_index, temperature_grid_index

from figures.config import DPI, OUTPUT_DIR, STANDARD_CONDITIONS, THERMO_IR_COUNTS
from figures.utils.cache import ensure_for_supplementary
from figures.utils.datasets import DATASETS, dataset_inputs
from figures.utils.io import ensure_dirs, load_gc_npz, load_variants_like
from figures.utils.plotting import (
    AXIS_LABEL_FONTSIZE,
    LEGEND_FONTSIZE,
    TICK_FONTSIZE,
    TITLE_FONTSIZE,
    save_fig,
)

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_oxidation_vs_n_ir.svg")
TABLE_PATH = os.path.join(OUTPUT_DIR, "supp_oxidation_vs_n_ir.tsv")
FIG_W_MM = 180
FIG_H_MM = 70
LEVELS = (("full", "-", "with vibrations"), ("novib", "--", "without vibrations"))


def _n_ir(formulas) -> np.ndarray:
    return np.array([int(re.search(r"Ir(\d*)", str(f)).group(1) or 1) for f in formulas])


def _oxidation_by_size(label: str) -> dict:
    """{(level, condition index): [⟨n_O/n_Ir⟩ per n_Ir]} of one functional."""
    inp = dataset_inputs(label)
    gc = load_gc_npz(inp.gc_npz, light=True)
    vd = load_variants_like(inp.variants_npz, gc, ("Ts", "Ps", "stoich", "full_Pgi", "novib_Pgi"))
    n_ir = _n_ir(gc["formulas"])
    stoich = np.asarray(vd["stoich"], dtype=float)
    out = {}
    for level, _, _ in LEVELS:
        pgi = vd[f"{level}_Pgi"]
        for c, cond in enumerate(STANDARD_CONDITIONS):
            p_idx, _ = pressure_grid_index(vd["Ps"], cond["P"])
            t_idx, _ = temperature_grid_index(vd["Ts"], cond["T"])
            vals = []
            for n in THERMO_IR_COUNTS:
                w = np.asarray(pgi[n_ir == n, p_idx, t_idx], dtype=float)
                vals.append(float(np.dot(w, stoich[n_ir == n]) / w.sum()))
            out[(level, c)] = vals
    return out


def _write_table(data: dict) -> None:
    with open(TABLE_PATH, "w") as fh:
        fh.write("functional\tlevel\tT_K\tpO2_Pa\t" + "\t".join(f"Ir{n}" for n in THERMO_IR_COUNTS) + "\n")
        for label, d in data.items():
            for level, _, _ in LEVELS:
                for c, cond in enumerate(STANDARD_CONDITIONS):
                    vals = "\t".join(f"{v:.3f}" for v in d[(level, c)])
                    fh.write(f"{label}\t{level}\t{cond['T']:.0f}\t{cond['P']:.3g}\t{vals}\n")


def main() -> None:
    ensure_for_supplementary(["supp_oxidation_vs_n_ir"])
    ensure_dirs(OUTPUT_DIR)
    data = {label: _oxidation_by_size(label) for label in DATASETS}

    fig, axes = plt.subplots(1, len(DATASETS), figsize=(FIG_W_MM / 25.4, FIG_H_MM / 25.4), dpi=DPI, sharey=True)
    x = np.asarray(THERMO_IR_COUNTS, dtype=float)
    for ax, label in zip(axes, DATASETS):
        for level, ls, _ in LEVELS:
            for c, cond in enumerate(STANDARD_CONDITIONS):
                ax.plot(x, data[label][(level, c)], linestyle=ls, marker="o" if level == "full" else None,
                        markersize=3.5, linewidth=1.2, color=cond["color"])
        ax.set_title(label, fontsize=TITLE_FONTSIZE)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Ir$_{{{n}}}$" for n in THERMO_IR_COUNTS])
        ax.set_xlabel(r"$n_{\mathrm{Ir}}$", fontsize=AXIS_LABEL_FONTSIZE)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
    axes[0].set_ylabel(r"$\langle n_O/n_{\mathrm{Ir}}\rangle$", fontsize=AXIS_LABEL_FONTSIZE)

    handles = [Line2D([], [], color=cond["color"], linewidth=1.5, label=cond["label"]) for cond in STANDARD_CONDITIONS]
    handles += [Line2D([], [], color="0.3", linestyle=ls, linewidth=1.2, label=name) for _, ls, name in LEVELS]
    fig.legend(handles=handles, fontsize=LEGEND_FONTSIZE, frameon=False, ncol=len(handles),
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.17, wspace=0.08)
    save_fig(fig, OUTPUT_PATH, dpi=DPI)
    _write_table(data)
    print(f"[supp] {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
