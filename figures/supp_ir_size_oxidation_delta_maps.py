"""Supplementary – Per-Ir Δ(n_O/n_Ir) maps: full − single-min | full − novib ensemble, PBE | PBE0.

With the thermodynamics restricted to Ir₄–Ir₆ this is Figure 9 drawn with shorter rows (it used to
extend Fig. 9 to Ir₃…Ir₉); the panels come from ``fig09_oxidation_delta_maps.build``.

Output: ``figures/outputs/supp_ir_size_oxidation_delta_maps.svg``
"""
from __future__ import annotations

import os

from figures.config import OUTPUT_DIR
from figures.fig09_oxidation_delta_maps import build
from figures.utils.cache import ensure_for_supplementary

OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_ir_size_oxidation_delta_maps.svg")
ROW_H_MM = 38


def main() -> None:
    ensure_for_supplementary(["supp_ir_size_oxidation_delta_maps"])
    build(OUTPUT_PATH, row_h_mm=ROW_H_MM, tag="supp")


if __name__ == "__main__":
    main()
