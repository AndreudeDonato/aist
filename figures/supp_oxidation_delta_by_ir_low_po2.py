"""Supplementary – Thermo-variant oxidation at low pO₂, one figure per functional (PBE, PBE0).

Former Figure 7 (low ``pO₂``). Same layout as ``supp_oxidation_delta_by_ir`` (ambient version), at
``FIG04_LOW_PRESSURE_PA``.

Outputs: figures/outputs/supp_oxidation_delta_by_ir_low_po2_PBE.svg, …_PBE0.svg
"""
from __future__ import annotations

from figures.config import FIG04_LOW_PRESSURE_PA
from figures.supp_oxidation_delta_by_ir import build
from figures.utils.cache import ensure_for_supplementary

OUTPUT_STEM = "supp_oxidation_delta_by_ir_low_po2"


def main() -> None:
    ensure_for_supplementary(["supp_oxidation_delta_by_ir_low_po2"])
    build(FIG04_LOW_PRESSURE_PA, OUTPUT_STEM)


if __name__ == "__main__":
    main()
