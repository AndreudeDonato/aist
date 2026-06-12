"""Master driver — build all paper figures.

Data policy: only ``<repo>/data/`` is read as input. Derived artifacts live in
``figures/cache/`` and are built automatically when missing or stale.

Usage:
    python3 figures/build_all.py                  # all figures
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

from figures.utils.cache import ensure_for_figures

FIGURE_MODULES = {
    1: ("fig01_minima", "PBE+PBE0 stoichiometry minima tiles"),
    2: ("fig02_atom_motifs", "Atom motif classification (3×2)"),
    3: ("fig03_motif_minima_tiles", "Pooled motif centroid tiles"),
    4: ("fig04_structure_motifs", "Structure motif clustering (1×3)"),
    5: ("fig05_gc_clustering_maps", "GC phase diagrams + motif overlays (4×2)"),
    6: ("fig06_gc_pressure_slices", "Ambient-pO₂ motif/cluster populations vs T (2×2)"),
    7: ("fig07_vib_vs_novib_quadrants", "PBE vib vs no-vib phase diagrams (2×2)"),
    8: ("fig08_vib_contrib_tchar", "PBE G-contrib 1D + vib map, ZPE, Tchar (2×2)"),
    9: ("fig09_tchar_motif_hs", "Hirshfeld parity + O |spin| KDE (2×2)"),
    10: ("fig10_min_pop_and_fixedpoint_lines", "PBE global-min map + fixed-point pop lines (1×3)"),
    11: ("fig11_homo_lumo_gap_maps", "PBE|PBE0 ⟨HOMO–LUMO⟩ phase maps (2×1)"),
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

    print(f"Building figures: {to_build}")
    ensure_for_figures(to_build, force=args.rebuild_cache)

    t_start = time.time()
    results = {n: run_figure(n) for n in to_build}

    print(f"\n{'='*60}")
    print(f"Summary  ({time.time() - t_start:.0f}s plotting)")
    print(f"{'='*60}")
    for n, ok in results.items():
        print(f"  Fig {n}  [{'OK  ' if ok else 'FAIL'}]  {FIGURE_MODULES[n][1]}")

    if sum(1 for ok in results.values() if not ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
