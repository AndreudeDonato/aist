"""Master driver — build all paper figures.

Data policy: only ``<repo>/data/`` is read as input. Derived artifacts live in
``figures/cache/`` and are built automatically when missing or stale.

Usage:
    python3 figures/build_all.py                  # all figures + supplementary
    python3 figures/build_all.py --only 3 7       # specific figures
    python3 figures/build_all.py --skip 5         # skip figure 5
    python3 figures/build_all.py --rebuild-cache  # force refresh cache from data/

Figure SVGs are written to ``figures/outputs/``.
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIGURES = os.path.dirname(os.path.abspath(__file__))
for _p in [_ROOT, os.path.join(_ROOT, "analysis"), _FIGURES]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from aist.ace.constants import ACE_RANDOM_STATE
from aist.ace.reproducibility import set_reproducible_seed
from figures.utils.cache import ensure_for_figures, ensure_for_supplementary

FIGURE_MODULES = {
    1: ("fig01_minima", "PBE+PBE0 stoichiometry minima tiles (+ supp_minima_PBE/PBE0)"),
    2: ("fig02_motif_clustering_panel", "Motif clustering 3×4 + moment ratios + horizontal centroid strip"),
    3: ("fig03_gc_clustering_maps", "GC phase diagrams + motif overlays (5×2)"),
    4: ("fig04_gc_pressure_slices", "Low-pO₂ motifs + terminal/bridging O maps (3×2)"),
    7: ("fig07_vib_vs_novib_quadrants", "PBE vib vs no-vib phase diagrams (2×2)"),
    8: ("fig08_vib_contrib_tchar", "PBE G-contrib 1D + vib map, ZPE, Tchar (2×2)"),
    9: ("fig09_min_pop_and_fixedpoint_lines", "PBE global-min map + fixed-point pop lines (1×3)"),
    10: ("fig10_homo_lumo_gap_maps", "PBE(novib)|PBE0 ⟨HOMO–LUMO⟩ maps + O-fraction envelope (3×1)"),
    12: ("fig12_ir_size_oxidation_at_po2", "Per-Ir ⟨n_O/n_Ir⟩ & ⟨HOMO–LUMO⟩ vs T at ambient pO₂ (2×3)"),
}

SUPPLEMENTARY_MODULES = {
    "supp_atom_hirshfeld": "Hirshfeld q vs |spin| by atom motif (1×2)",
    "supp_motif_spin_kde": "O | Ir motif |Hirshfeld spin| KDE (PBE | PBE0, 2×2)",
    "supp_ir_size_o_motif_ratio_maps": "Per-Ir terminal | bridging O / n_Ir maps (7×2 each)",
    "supp_gc_g_vs_mu": "Per-structure G vs μ_O₂ at 273 K & 500 K (7×2 each)",
    "supp_ir_size_gc_maps": "Per-Ir GC maps: min pop | oxidation | cluster | Ir | O | inertia | HOMO–LUMO (7×2 each)",
    "supp_ir_size_vib_vs_novib_maps": "Per-Ir PBE vib vs no-vib ⟨n_O/n_Ir⟩ maps (7×2)",
    "supp_moments_of_inertia": "Principal moment-ratio triangle (O | Ir | structures, 3×1)",
}


def run_figure(num: int) -> bool:
    module_name, description = FIGURE_MODULES[num]
    print(f"\n{'='*60}")
    print(f"  Figure {num}: {description}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = importlib.import_module(f"figures.{module_name}")
        mod.main()
        print(f"  [OK] Fig {num} done in {time.time() - t0:.1f}s")
        return True
    except Exception:
        print(f"  [FAIL] Fig {num} after {time.time() - t0:.1f}s:")
        traceback.print_exc()
        return False


def run_supplementary(name: str) -> bool:
    description = SUPPLEMENTARY_MODULES[name]
    print(f"\n{'='*60}")
    print(f"  Supplementary: {description}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        mod = importlib.import_module(f"figures.{name}")
        mod.main()
        print(f"  [OK] {name} done in {time.time() - t0:.1f}s")
        return True
    except Exception:
        print(f"  [FAIL] {name} after {time.time() - t0:.1f}s:")
        traceback.print_exc()
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--only", nargs="+", type=int, metavar="N")
    parser.add_argument("--skip", nargs="+", type=int, metavar="N")
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Recompute figures/cache from data/ before plotting",
    )
    args = parser.parse_args()

    to_build = list(FIGURE_MODULES)
    if args.only:
        to_build = [n for n in to_build if n in args.only]
    if args.skip:
        to_build = [n for n in to_build if n not in args.skip]

    build_supp = not args.only

    set_reproducible_seed(ACE_RANDOM_STATE)
    print(f"Building figures: {to_build}  (ACE_RANDOM_STATE={ACE_RANDOM_STATE})")
    ensure_for_figures(to_build, force=args.rebuild_cache)
    if build_supp:
        ensure_for_supplementary(SUPPLEMENTARY_MODULES.keys(), force=args.rebuild_cache)

    t_start = time.time()
    results = {n: run_figure(n) for n in to_build}
    supp_results: dict[str, bool] = {}
    if build_supp:
        supp_results = {name: run_supplementary(name) for name in SUPPLEMENTARY_MODULES}

    print(f"\n{'='*60}")
    print(f"Summary  ({time.time() - t_start:.0f}s plotting)")
    print(f"{'='*60}")
    for n, ok in results.items():
        print(f"  Fig {n}  [{'OK  ' if ok else 'FAIL'}]  {FIGURE_MODULES[n][1]}")
    for name, ok in supp_results.items():
        print(f"  Supp  [{'OK  ' if ok else 'FAIL'}]  {SUPPLEMENTARY_MODULES[name]}")

    if sum(1 for ok in results.values() if not ok) + sum(1 for ok in supp_results.values() if not ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
