# aist

Grand-canonical thermodynamics of gas-phase Ir_yO_x clusters (y = 3–9) and the figures of the
IryOx paper (`../aist_paper`). Python package + figure scripts.

## Layout

- `aist/` — the library: grand-canonical ensembles (`core/`), vibrational and rotational thermodynamics
  (`thermo.py`, `composedthermo.py`, `core/vibrational.py`), ACE/SOAP motif clustering (`ace/`)
- `data/` — the only input: `IryOxPBE.traj`, `IryOxPBE0.traj` and the reference structures
  (Ir, O2, IrO2, IrO3, rutile IrO2)
- `figures/` — one script per paper figure (`fig*.py`, `supp_*.py`), `config.py`, `build_all.py`
- `analysis/` — exploratory analysis scripts
- `docs/` — methodology notes
- `scripts/` — older stand-alone scripts

## What enters which figures

- Structure analysis and the ACE/SOAP clustering use every structure of both trajectories (Ir3–Ir9).
- Thermodynamics (everything grand-canonical) uses Ir4–Ir6 only, for both functionals
  (`config.GC_IR_COUNTS`), with vibrations for both (`GC_USE_VIB_ZPE_BY_DATASET`). Structures without a
  spectrum are left out of the ensembles and listed in `figures/cache/gc_excluded_<label>.tsv`; each cache has
  a `<npz>_frames.npy` sidecar mapping its rows to trajectory frames.
- The vibrational figures compare PBE and PBE0, each with its own references
  (`figures/utils/datasets.py`: `dataset_inputs`, `gc_structures`).

## Regenerating everything

Every figure is rebuilt from `data/`; nothing derived is kept in the repository.

```bash
pip install -e .                         # once
python3 figures/build_all.py             # cache (figures/cache/) + all figures (figures/outputs/)
python3 figures/build_all.py --only 3 7  # some figures
python3 figures/build_all.py --rebuild-cache
python3 figures/build_all.py --no-paper      # do not touch ../aist_paper
```

Every figure the manuscript uses is also written into `../aist_paper/figures/{main,supplementary}/`
(SVG + PNG, Inkscape at 150 dpi as `aist_paper/convert_svgs.sh` does; `figures/utils/paper.py`), so
`aist_paper/compile_all.sh` always picks up the latest version. A figure goes where the paper already
has it and is only rewritten when its content changed; to add a new figure to the paper, put a first
copy in the right folder once. `AIST_PAPER_FIGURES=<dir>` points it elsewhere.
