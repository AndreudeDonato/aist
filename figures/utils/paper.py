"""Write the figures the manuscript uses into the paper repo (aist_paper/figures/{main,supplementary}/).

Where a figure goes is decided by where the paper already has it: an output ``<name>.svg`` is copied
over ``main/<name>.svg`` or ``supplementary/<name>.svg``, and its PNG is rendered with Inkscape exactly
as ``aist_paper/convert_svgs.sh`` does. Outputs the paper does not use are skipped; to add a new figure
to the paper, put a first copy in the right folder once. A figure is only rewritten when its content
changed (the SVG date and matplotlib's generated ids are ignored), so unchanged figures stay untouched.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess

from figures.config import OUTPUT_DIR, PAPER_EXPORT_DPI, PAPER_FIGURES_DIR

SECTIONS = ("main", "supplementary")


def _content(path: str) -> str:
    """SVG text without the parts matplotlib changes on every save."""
    with open(path, errors="replace") as fh:
        t = fh.read()
    t = re.sub(r"<dc:date>.*?</dc:date>", "", t)
    t = re.sub(r'\b(id|clip-path|mask)="[^"]*"', "", t)
    # references to generated ids only: embedded rasters (href="data:image/png;base64,...") are content
    t = re.sub(r'\b(href|xlink:href)="#[^"]*"', "", t)
    return re.sub(r"url\(#[^)]*\)", "", t)


def publish(output_dir: str = OUTPUT_DIR, paper_dir: str = PAPER_FIGURES_DIR, dpi: int = PAPER_EXPORT_DPI) -> dict:
    """Copy changed figures into the paper and render their PNGs. Returns {updated, unchanged, not_in_paper}."""
    out = {"updated": [], "unchanged": [], "not_in_paper": []}
    if not os.path.isdir(paper_dir):
        print(f"[paper] no paper figures folder at {paper_dir}; skipped")
        return out
    placed = {}
    for sec in SECTIONS:
        for f in glob.glob(os.path.join(paper_dir, sec, "*.svg")):
            placed[os.path.splitext(os.path.basename(f))[0]] = sec
    inkscape = shutil.which("inkscape")
    for svg in sorted(glob.glob(os.path.join(output_dir, "*.svg"))):
        name = os.path.splitext(os.path.basename(svg))[0]
        sec = placed.get(name)
        if sec is None:
            out["not_in_paper"].append(name)
            continue
        dst = os.path.join(paper_dir, sec, name + ".svg")
        png = dst[:-4] + ".png"
        png_current = os.path.exists(png) and os.path.getmtime(png) >= os.path.getmtime(dst)
        if png_current and _content(dst) == _content(svg):
            out["unchanged"].append(name)
            continue
        shutil.copyfile(svg, dst)
        if inkscape:
            subprocess.run([inkscape, dst, "--export-type=png", f"--export-dpi={dpi}", f"--export-filename={png}"],
                           check=True, capture_output=True)
        else:
            print(f"[paper] inkscape not found: {name}.png not refreshed (run aist_paper/convert_svgs.sh)")
        out["updated"].append(f"{sec}/{name}")
    print(f"[paper] {paper_dir}: {len(out['updated'])} updated, {len(out['unchanged'])} unchanged, "
          f"{len(out['not_in_paper'])} outputs the paper does not use")
    for n in out["updated"]:
        print(f"[paper]   updated {n} (.svg + .png)")
    return out
