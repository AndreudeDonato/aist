"""The 0.1 file-based API (``compute_gc``, ``run_gc``, ``run_gc_variants``) over the new core.

Every key of the NPZ dictionaries, and the auxiliary trajectories, must equal those of the frozen 0.1 code
(``tests/legacy/aist_v01``) bit for bit; and the functions now accept any single-element reference and
reservoir.
"""
import os
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "legacy"))

import toy  # noqa: E402
from aist.core.gc_variants import run_gc_variants  # noqa: E402
from aist.core.grandcanonical import GCConfig, compute_gc  # noqa: E402


def _assert_same(new: dict, old: dict):
    assert list(new) == list(old)
    for key in old:
        a, b = np.asarray(new[key]), np.asarray(old[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        if a.dtype == object:
            assert list(a) == list(b), key
        else:
            assert np.array_equal(a, b, equal_nan=True), key


@pytest.fixture
def files(tmp_path):
    paths = {name: str(tmp_path / f"{name}.traj") for name in ("ir", "pt", "o2", "h2", "h2_spin", "ir_o", "pt_o", "ir_h")}
    write(paths["ir"], toy.metal_atom())
    write(paths["pt"], toy.metal_atom("Pt"))
    write(paths["o2"], toy.o2())
    write(paths["h2"], toy.h2())
    h2 = toy.h2()
    h2.info["total_spin"] = {"N": 0.0, "S": 0.0, "J": 1.0}
    write(paths["h2_spin"], h2)
    write(paths["ir_o"], toy.metal_oxide_set())
    write(paths["pt_o"], toy.metal_oxide_set(metal="Pt"))
    write(paths["ir_h"], toy.metal_oxide_set(ligand="H"))
    return paths


def _cfg(module, ref, **kw):
    return module.GCConfig(T_min=200.0, T_max=900.0, n_T=6, P_min=1e-8, P_max=1e4, n_P=5, reference_file=ref, **kw)


@pytest.mark.parametrize("decomposition", [True, False])
def test_compute_gc_matches_legacy(files, tmp_path, decomposition):
    import aist_v01.core.grandcanonical as legacy

    outs = {}
    for name, module in (("old", legacy), ("new", sys.modules["aist.core.grandcanonical"])):
        d = tmp_path / name
        outs[name] = module.compute_gc(
            files["ir_o"], files["o2"], _cfg(module, files["ir"], include_decomposition=decomposition),
            output_npz=str(d / "gc.npz"), output_threshold_dir=str(d / "thr"),
            output_minima_traj=str(d / "minima.traj"),
        )
    _assert_same(outs["new"], outs["old"])
    _assert_same(dict(np.load(tmp_path / "new" / "gc.npz", allow_pickle=True)),
                 dict(np.load(tmp_path / "old" / "gc.npz", allow_pickle=True)))
    for rel in ["minima.traj"] + [os.path.join("thr", f) for f in sorted(os.listdir(tmp_path / "old" / "thr"))]:
        old, new = read(tmp_path / "old" / rel, ":"), read(tmp_path / "new" / rel, ":")
        assert [a.get_chemical_formula() for a in old] == [a.get_chemical_formula() for a in new]
        assert all(np.array_equal(a.positions, b.positions) for a, b in zip(old, new))


def test_run_gc_variants_matches_legacy(files, tmp_path):
    import aist_v01.core.gc_variants as legacy_variants
    import aist_v01.core.grandcanonical as legacy

    legacy_variants.run_gc_variants(files["ir_o"], files["o2"], str(tmp_path / "old.npz"),
                                    grid_config=_cfg(legacy, files["ir"]))
    run_gc_variants(files["ir_o"], files["o2"], str(tmp_path / "new.npz"),
                    grid_config=_cfg(sys.modules["aist.core.grandcanonical"], files["ir"]))
    _assert_same(dict(np.load(tmp_path / "new.npz", allow_pickle=True)),
                 dict(np.load(tmp_path / "old.npz", allow_pickle=True)))


def test_compute_gc_takes_other_metals(files):
    ir = compute_gc(files["ir_o"], files["o2"], _cfg(sys.modules["aist.core.grandcanonical"], files["ir"]),
                    save_outputs=False)
    pt = compute_gc(files["pt_o"], files["o2"], _cfg(sys.modules["aist.core.grandcanonical"], files["pt"]),
                    save_outputs=False)
    assert np.array_equal(ir["Pgi"], pt["Pgi"]) and np.array_equal(ir["x_per_non_o"], pt["x_per_non_o"])


def test_compute_gc_needs_the_spin_of_a_non_o2_reservoir(files):
    cfg = _cfg(sys.modules["aist.core.grandcanonical"], files["ir"])
    with pytest.raises(ValueError, match="multiplicity"):
        compute_gc(files["ir_h"], files["h2"], cfg, save_outputs=False)
    out = compute_gc(files["ir_h"], files["h2_spin"], cfg, save_outputs=False)
    structures = read(files["ir_h"], ":")
    assert np.array_equal(out["x_values"], [a.get_chemical_symbols().count("H") for a in structures])
    assert np.array_equal(out["non_o_counts"], [a.get_chemical_symbols().count("Ir") for a in structures])
    assert np.all(out["x_per_non_o"] > 0)


def test_reference_must_be_one_element(files, tmp_path):
    path = str(tmp_path / "iro.traj")
    write(path, toy.with_data(Atoms("IrO", positions=[(0, 0, 0), (1.8, 0, 0)]), -105.0))
    with pytest.raises(ValueError, match="one element"):
        compute_gc(files["ir_o"], files["o2"], _cfg(sys.modules["aist.core.grandcanonical"], path),
                   save_outputs=False)
