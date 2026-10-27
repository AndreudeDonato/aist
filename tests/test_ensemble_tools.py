"""Tests of the ensemble tools: re-weighting, subsets, support shift, reservoir μ, stamps, gaps."""
import os
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.units import kB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aist.cache_stamp import make_stamp, npz_stamp, write_npz_stamp  # noqa: E402
from aist.core.calibration import CalibrationAnchor, g_difference_at_anchor  # noqa: E402
from aist.core.ensemble import lowest_g_index, reensemble_from_g, renormalize_populations  # noqa: E402
from aist.core.gaps import homo_lumo_gap_map, kernel_smoothed_mean_vs_oxidation  # noqa: E402
from aist.core.reference import reservoir_chemical_potential_grid  # noqa: E402
from aist.core.subset import oxidation_vs_t_by_ir_at_pressure, ranked_structure_pops_by_ir_at_fixed_point  # noqa: E402
from aist.core.support import patch_support_stabilization, support_corrections_per_ir  # noqa: E402


def _toy_tensor(seed=0, n=5, n_p=3, n_t=4):
    rng = np.random.default_rng(seed)
    Ts = np.linspace(200.0, 800.0, n_t)
    Ps = np.logspace(-5, 5, n_p)
    Tg, _ = np.meshgrid(Ts, Ps, indexing="xy")
    g = rng.normal(size=(n, 1, 1)) * 0.3 + rng.normal(size=(n, n_p, n_t)) * 0.02
    deg = np.array([1, 2, 1, 1, 2], dtype=float)
    stoich = np.array([0.5, 1.0, 1.5, 2.0, 3.0])
    return g, Tg, Ts, Ps, deg, stoich


def test_reensemble_matches_boltzmann_formula():
    g, Tg, Ts, Ps, deg, stoich = _toy_tensor()
    ens = reensemble_from_g(g, Tg, stoich, deg)
    beta = 1.0 / (kB * Tg)
    w = deg[:, None, None] * np.exp(-beta * (g - g.min(axis=0)))
    P = w / w.sum(axis=0)
    assert np.allclose(ens["Pgi"], P, atol=1e-6)
    assert np.allclose(ens["Pgi"].sum(axis=0), 1.0, atol=1e-6)
    assert np.array_equal(ens["phase_index"], np.argmin(g, axis=0))
    assert np.allclose(ens["x_per_non_o"], np.tensordot(stoich, P, axes=1), atol=1e-5)


def test_lowest_g_index_ignores_chirality_weight():
    # row 1 is chiral (g=2) and 10 meV above row 0: argmax P would pick it at high T, argmin G must not
    g = np.array([[[0.0]], [[0.01]]])
    Tg = np.array([[1000.0]])
    ens = reensemble_from_g(g, Tg, np.array([1.0, 2.0]), np.array([1.0, 2.0]))
    assert np.argmax(ens["Pgi"][:, 0, 0]) == 1
    assert lowest_g_index(ens["Pgi"], np.array([1.0, 2.0]))[0, 0] == 0


def test_renormalized_subset_equals_restricted_boltzmann():
    g, Tg, Ts, Ps, deg, stoich = _toy_tensor()
    ens = reensemble_from_g(g, Tg, stoich, deg)
    mask = np.array([True, False, True, True, False])
    sub = renormalize_populations(ens["Pgi"], mask)
    beta = 1.0 / (kB * Tg)
    w = deg[mask][:, None, None] * np.exp(-beta * (g[mask] - g[mask].min(axis=0)))
    assert np.allclose(sub, w / w.sum(axis=0), atol=1e-6)
    assert np.allclose(sub.sum(axis=0), 1.0, atol=1e-6)


def test_oxidation_vs_t_per_size_and_ranked_populations():
    g, Tg, Ts, Ps, deg, stoich = _toy_tensor()
    ens = reensemble_from_g(g, Tg, stoich, deg)
    n_ir = np.array([3, 3, 3, 4, 4])
    gc = {"Ts": Ts, "Ps": Ps, "Pgi": ens["Pgi"]}
    _, p_used, lines = oxidation_vs_t_by_ir_at_pressure(gc, stoich, n_ir, 1.0)
    assert set(lines) == {3, 4}
    p_idx = int(np.argmin(np.abs(np.log(Ps) - np.log(p_used))))
    for n in (3, 4):
        sub = renormalize_populations(ens["Pgi"], n_ir == n, p_idx=p_idx)
        assert np.allclose(lines[n], stoich[n_ir == n] @ sub, atol=1e-5)
    t_used, _, ranked = ranked_structure_pops_by_ir_at_fixed_point(
        ens["Pgi"], ens["Pgi"], g, n_ir, Ts, Ps, 500.0, 1.0, max_structures=2)
    assert ranked[3][0].shape == (2,) and ranked[3][0][0] >= ranked[3][0][1] - 1e-6  # s0 is the minimum


def test_support_shift_only_tilts_by_oxidation(tmp_path):
    atoms = []
    for n_ir, n_o in ((2, 1), (2, 4), (1, 3)):
        a = Atoms("Ir%dO%d" % (n_ir, n_o), positions=np.random.default_rng(n_o).normal(size=(n_ir + n_o, 3)) * 2)
        a.calc = SinglePointCalculator(a, energy=-1.0)
        atoms.append(a)
    gas = Atoms("IrO3", positions=np.random.default_rng(9).normal(size=(4, 3)) * 2)
    gas.calc = SinglePointCalculator(gas, energy=-1.0)
    traj = tmp_path / "s.traj"
    write(traj, atoms + [gas])
    corr = support_corrections_per_ir(str(traj), [3], ev_per_ir=0.5, ev_per_o=0.1)
    assert np.allclose(corr, [0.5 + 0.1 * 0.5, 0.5 + 0.1 * 2.0, 0.5 + 0.1 * 3.0, 0.0])
    # a per-Ir-only shift leaves the cluster populations unchanged (gas row excluded from the comparison)
    g, Tg, Ts, Ps, deg, stoich = _toy_tensor(n=4)
    stoich = np.array([0.5, 2.0, 3.0, 3.0]); deg = np.ones(4)
    ens0 = reensemble_from_g(g, Tg, stoich, deg)
    npz = tmp_path / "gc.npz"
    np.savez(npz, G_tensor=g.astype(np.float32), Tg=Tg, stoich=stoich, degeneracy=deg, **{k: v for k, v in ens0.items()})
    patch_support_stabilization(str(npz), str(traj), [3], ev_per_ir=0.5, ev_per_o=0.0)
    d = dict(np.load(npz, allow_pickle=True))
    P1 = d["Pgi"][:3] / d["Pgi"][:3].sum(axis=0)
    P0 = ens0["Pgi"][:3] / ens0["Pgi"][:3].sum(axis=0)
    assert np.allclose(P1, P0, atol=1e-5)
    assert float(d["gc_support_ev_per_ir"]) == 0.5


def test_reservoir_mu_has_ideal_gas_pressure_dependence(tmp_path):
    o2 = Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.2]])
    o2.calc = SinglePointCalculator(o2, energy=-10.0)
    o2.info["frequencies"] = [0.0] * 5 + [1580.0]
    o2.info["zpe"] = 0.5 * 1580.0 * 1.23981e-4
    path = tmp_path / "O2.xyz"
    write(path, o2)
    Ts = np.array([300.0, 600.0]); Ps = np.array([1.0, 1000.0])
    mu = reservoir_chemical_potential_grid(str(path), Ts, Ps, novib=True)
    assert mu.shape == (2, 2)
    # mu(p2) - mu(p1) = kT ln(p2/p1) per molecule, at each T
    assert np.allclose(mu[1] - mu[0], kB * Ts * np.log(1000.0), atol=1e-6)
    assert np.all(np.diff(mu, axis=1) < 0)  # entropy lowers mu with T
    mu_vib = reservoir_chemical_potential_grid(str(path), Ts, Ps, novib=False)
    assert np.all(mu_vib > mu)  # ZPE dominates the vibrational term at these T


def test_cache_stamp_tracks_inputs_and_settings(tmp_path):
    f = tmp_path / "in.txt"; f.write_text("a")
    s1 = make_stamp("gc", cache_version=2, inputs=[str(f)], n_T=150)
    assert s1 == make_stamp("gc", cache_version=2, inputs=[str(f)], n_T=150)
    assert s1 != make_stamp("gc", cache_version=3, inputs=[str(f)], n_T=150)
    assert s1 != make_stamp("gc", cache_version=2, inputs=[str(f)], n_T=151)
    f.write_text("b"); os.utime(f, (1, 1))
    assert s1 != make_stamp("gc", cache_version=2, inputs=[str(f)], n_T=150)
    npz = tmp_path / "x.npz"; np.savez(npz, a=np.arange(3))
    assert npz_stamp(str(npz)) is None
    write_npz_stamp(str(npz), s1)
    assert npz_stamp(str(npz)) == s1 and np.array_equal(np.load(npz)["a"], np.arange(3))


def test_gap_tools():
    Pgi = np.full((3, 2, 2), 1 / 3)
    assert np.allclose(homo_lumo_gap_map(Pgi, np.array([1.0, 2.0, 3.0])), 2.0)
    with pytest.raises(ValueError):
        homo_lumo_gap_map(Pgi, np.array([1.0, 2.0]))
    x = np.linspace(0, 3, 20)
    y = kernel_smoothed_mean_vs_oxidation(np.random.default_rng(1).uniform(0, 3, 50), np.full(50, 1.5), x)
    assert np.allclose(y, 1.5, atol=1e-9)


def test_g_difference_at_anchor():
    ps = np.array([1.0, 10.0, 100.0]); ts = np.array([100.0, 200.0])
    g = np.zeros((2, 3, 2)); g[0] = 1.0; g[1] = np.arange(6).reshape(3, 2) * 0.1
    d = g_difference_at_anchor(g, ps, ts, ["A", "B"], "A", "B", anchor=CalibrationAnchor(T_K=150.0, P_O2_Pa=10.0))
    assert np.isclose(d, 1.0 - 0.25)
