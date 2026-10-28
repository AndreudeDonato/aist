"""Ensemble building blocks and the generalised grand-canonical construction (``aist.ensembles``).

Generality (other metals, other reservoirs, two reservoirs at once), the exact balance, and continuity: on toy
data the construction reproduces the frozen 0.1 code (``tests/legacy/aist_v01``) bit for bit.
"""
import os
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.io import write
from ase.thermochemistry import IdealGasThermo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "legacy"))

import toy  # noqa: E402
from aist import (  # noqa: E402
    BalanceError, GrandCanonicalEnsemble, Reference, Reservoir, ThermoTerms, boltzmann_populations, free_energy,
)

TS = np.linspace(200.0, 900.0, 5)
PS = np.logspace(-8, 4, 4)
VIB = ThermoTerms(vibrations=True, zpe=True)  # electronic + vibrations + ZPE: the 0.1 defaults


def _ir_o2(structures=None, **kw):
    return GrandCanonicalEnsemble(
        structures if structures is not None else toy.metal_oxide_set(),
        references=[Reference(toy.metal_atom())],
        reservoirs=[Reservoir(toy.o2(), multiplicity=3)],
        **kw,
    )


def _iro2_gas():
    return toy.with_data(Atoms("IrO2", positions=[(0, 0, 0), (1.2, 1.1, 0), (-1.2, 1.1, 0)]), -111.5, J=2.0)


# --------------------------------------------------------------------------- building blocks
def test_boltzmann_populations_formula():
    G = np.array([[0.0, 0.1], [0.05, 0.0], [0.2, 0.3]])
    T = np.array([300.0, 900.0])
    g = np.array([1.0, 2.0, 1.0])
    w = g[:, None] * np.exp(-(G - G.min(axis=0)) / (8.617333262e-5 * T))
    assert np.allclose(boltzmann_populations(G, T, g), w / w.sum(axis=0))


def test_reservoir_mu_is_its_ideal_gas_free_energy():
    res = Reservoir(toy.o2(), multiplicity=3)
    T, p = np.array([300.0, 800.0]), np.array([1e5, 1e-3])
    ase = [IdealGasThermo(vib_energies=[1580.0 * toy.CM_TO_EV], potentialenergy=-9.86, atoms=toy.o2(),
                          geometry="linear", symmetrynumber=2, spin=1).get_gibbs_energy(t, q, verbose=False)
           for t, q in zip(T, p)]
    assert np.allclose(res.chemical_potential(T, p), ase, atol=1e-4)


# --------------------------------------------------------------------------- balance
def test_balance_matches_balance_reaction_exactly():
    from aist.stoichiometry import balance_reaction

    ens = _ir_o2()
    for i, atoms in enumerate(ens.structures):
        r, p = balance_reaction([toy.metal_atom(), toy.o2()], [atoms])
        assert ens._p[i] == p[0] and ens._c[i, 0] == r[0] and ens._c[i, 1] == r[1]


def test_dependent_reservoirs_raise():
    res = [Reservoir(toy.o2(), multiplicity=3), Reservoir(toy.h2(), multiplicity=1),
           Reservoir(toy.h2o(), multiplicity=1)]
    with pytest.raises(BalanceError, match="linearly dependent"):
        GrandCanonicalEnsemble(toy.hydroxide_set(), references=[Reference(toy.metal_atom())], reservoirs=res)


def test_unsupplied_element_raises():
    with pytest.raises(BalanceError, match="H"):
        _ir_o2(toy.hydroxide_set())


def test_two_reservoir_balance_is_unique():
    ens = GrandCanonicalEnsemble(toy.hydroxide_set(), references=[Reference(toy.metal_atom())],
                                 reservoirs=[Reservoir(toy.o2(), multiplicity=3), Reservoir(toy.h2o(), multiplicity=1)])
    i = list(ens.formulas).index("H2Ir2O3")
    assert ens.coefficients["H2O"][i] == 1.0 and ens.coefficients["O2"][i] == 1.0 and ens.coefficients["Ir"][i] == 2.0


# --------------------------------------------------------------------------- generality
def test_h2_reservoir_uses_its_own_spin():
    ens = GrandCanonicalEnsemble(toy.metal_oxide_set(ligand="H"), references=[Reference(toy.metal_atom())],
                                 reservoirs=[Reservoir(toy.h2(), multiplicity=1)])
    res = ens.evaluate(T=TS, p={"H2": PS})
    assert np.array_equal(ens.normalization, ens.count("Ir").astype(float))
    assert np.all(res.composition("H") > 0)


def test_platinum_equals_iridium_relabelled():
    ir = _ir_o2(terms=VIB).evaluate(T=TS, p={"O2": PS})
    pt = GrandCanonicalEnsemble(toy.metal_oxide_set(metal="Pt"), references=[Reference(toy.metal_atom("Pt"))],
                                reservoirs=[Reservoir(toy.o2(), multiplicity=3)], terms=VIB).evaluate(T=TS, p={"O2": PS})
    assert np.array_equal(ir.G, pt.G) and np.array_equal(ir.populations, pt.populations)


def test_fixed_temperature_map_with_two_reservoirs():
    ens = GrandCanonicalEnsemble(toy.hydroxide_set(), references=[Reference(toy.metal_atom())],
                                 reservoirs=[Reservoir(toy.o2(), multiplicity=3), Reservoir(toy.h2o(), multiplicity=1)])
    p_o2, p_h2o = np.logspace(-10, 5, 7), np.logspace(-10, 5, 6)
    res = ens.evaluate(T=500.0, p={"O2": p_o2, "H2O": p_h2o})
    assert res.G.shape[1:] == (7, 6) and list(res.axes) == ["O2", "H2O"]
    assert np.allclose(res.populations.sum(axis=0), 1.0, atol=1e-5)
    assert np.all(np.diff(res.composition("H"), axis=1) >= -1e-7)  # more water, at least as much H
    flipped = ens.evaluate(T=500.0, p={"O2": p_o2, "H2O": p_h2o}, axes=["H2O", "O2"])
    assert np.array_equal(flipped.populations, np.transpose(res.populations, (0, 2, 1)))


def test_reservoir_basis_change_gives_same_populations():
    o2, h2o, h2 = (Reservoir(toy.o2(), multiplicity=3), Reservoir(toy.h2o(), multiplicity=1),
                   Reservoir(toy.h2(), multiplicity=1))
    ref = [Reference(toy.metal_atom())]
    T, p_o2, p_h2o = 600.0, 1e2, np.logspace(-8, 4, 9)
    a = GrandCanonicalEnsemble(toy.hydroxide_set(), references=ref, reservoirs=[o2, h2o])
    b = GrandCanonicalEnsemble(toy.hydroxide_set(), references=ref, reservoirs=[o2, h2])
    dmu_h2 = h2o.chemical_potential(T, p_h2o) - 0.5 * o2.chemical_potential(T, p_o2) - h2.energy  # H2O = H2 + ½O2
    pa = a.evaluate(T=T, p={"O2": p_o2, "H2O": p_h2o}, dtype=np.float64).populations
    pb = b.evaluate(T=T, p={"O2": p_o2}, dmu={"H2": dmu_h2}, dtype=np.float64).populations
    assert np.allclose(pa, pb, atol=1e-6)


def test_dmu_axis_equals_pressure_axis():
    ens = _ir_o2()
    res_p = ens.evaluate(T=700.0, p={"O2": PS}, dtype=np.float64)
    mu = ens.reservoirs[0].chemical_potential(700.0, PS)
    res_mu = ens.evaluate(T=700.0, dmu={"O2": mu - ens.reservoirs[0].energy}, dtype=np.float64)
    assert np.allclose(res_p.G, res_mu.G, atol=1e-12)


def test_conditions_are_validated():
    ens = _ir_o2()
    with pytest.raises(ValueError, match="exactly one"):
        ens.evaluate(T=TS)
    with pytest.raises(KeyError, match="No reservoir"):
        ens.evaluate(T=TS, p={"O2": PS, "H2": 1.0})
    with pytest.raises(ValueError, match="axes"):
        ens.evaluate(T=TS, p={"O2": PS}, axes=["T"])


# --------------------------------------------------------------------------- terms, variants, views
def test_terms_per_row():
    """Gas-phase molecules in the pool: rows with their own terms (translation and rotation at a pressure)."""
    rows = toy.metal_oxide_set() + [_iro2_gas()]
    gas = VIB.replace(rotation=True, translation=True, pressure=1.0)
    res = _ir_o2(rows, terms=[VIB] * (len(rows) - 1) + [gas]).evaluate(T=TS, p={"O2": PS}, dtype=np.float64)
    mu = Reservoir(toy.o2(), multiplicity=3).chemical_potential(res.grid.T, res.grid.full["O2"])
    g = free_energy(rows[-1], res.grid.T, terms=gas) - toy.metal_atom().get_potential_energy() - mu
    assert np.allclose(res.G[-1], g, atol=1e-10)
    with pytest.raises(ValueError, match="one per structure"):
        _ir_o2(rows, terms=[VIB, VIB])


def test_variant_equals_fresh_ensemble():
    novib = _ir_o2(terms=VIB).variant(vibrations=False, zpe=False).evaluate(T=TS, p={"O2": PS})
    fresh = GrandCanonicalEnsemble(
        toy.metal_oxide_set(), references=[Reference(toy.metal_atom())],
        reservoirs=[Reservoir(toy.o2(), multiplicity=3, terms=ThermoTerms.gas(vibrations=False, zpe=False))],
    ).evaluate(T=TS, p={"O2": PS})
    assert np.array_equal(novib.G, fresh.G) and np.array_equal(novib.populations, fresh.populations)


def test_with_structures_uses_the_new_data():
    ens = _ir_o2(terms=VIB)
    stiffer = toy.metal_oxide_set()
    for a in stiffer:
        a.info["frequencies"] = [f * 1.1 for f in a.info["frequencies"]]
        a.info["zpe"] *= 1.1
    swapped = ens.with_structures(stiffer).evaluate(T=TS, p={"O2": PS})
    assert np.array_equal(swapped.G, _ir_o2(stiffer, terms=VIB).evaluate(T=TS, p={"O2": PS}).G)
    assert not np.array_equal(swapped.G, ens.evaluate(T=TS, p={"O2": PS}).G)
    with pytest.raises(ValueError, match="formulas"):
        ens.with_structures(toy.metal_oxide_set()[::-1])


def test_subset_renormalises_and_shifted_reweights():
    ens = _ir_o2()
    res = ens.evaluate(T=TS, p={"O2": PS})
    sub = res.subset(counts={"Ir": 3})
    assert np.allclose(sub.populations.sum(axis=0), 1.0, atol=1e-6) and set(sub.formulas) == {"Ir3O2", "Ir3O4"}
    delta = -(0.65 + 0.1 * ens.count("O") / ens.normalization)  # e.g. a support proxy
    shifted = res.shifted(delta)
    assert np.allclose((shifted.G.astype(float) - res.G.astype(float))[:, 0, 0], delta, atol=1e-5)
    assert np.allclose(shifted.populations,
                       boltzmann_populations(shifted.G.astype(float), shifted.grid.T, ens.degeneracy), atol=1e-6)


def test_degeneracy_counts_both_enantiomers():
    ens = _ir_o2()
    assert set(np.unique(ens.degeneracy)) <= {1.0, 2.0} and 2.0 in ens.degeneracy
    flat = _ir_o2(degeneracy=None).evaluate(T=TS, p={"O2": PS})
    w = flat.populations.astype(float) * ens.degeneracy[:, None, None]
    assert np.allclose(ens.evaluate(T=TS, p={"O2": PS}).populations, w / w.sum(axis=0), atol=1e-6)


# --------------------------------------------------------------------------- continuity with the 0.1 code
@pytest.fixture
def toy_files(tmp_path):
    clusters, iro2 = toy.metal_oxide_set(), _iro2_gas()
    paths = {name: str(tmp_path / f"{name}.traj") for name in ("ir", "o2", "clusters", "merged")}
    write(paths["ir"], toy.metal_atom())
    write(paths["o2"], toy.o2())
    write(paths["clusters"], clusters)
    write(paths["merged"], clusters + [iro2])
    return paths, clusters, iro2


def _legacy_config(ir_path, **kw):
    from aist_v01.core.grandcanonical import GCConfig

    return GCConfig(T_min=200.0, T_max=900.0, n_T=6, P_min=1e-8, P_max=1e4, n_P=5, reference_file=ir_path, **kw)


def test_matches_legacy_compute_gc(toy_files):
    from aist_v01.core.grandcanonical import compute_gc

    paths, clusters, _ = toy_files
    cfg = _legacy_config(paths["ir"], include_decomposition=True)
    old = compute_gc(paths["clusters"], paths["o2"], cfg, save_outputs=False)
    ens = _ir_o2(clusters, terms=VIB.replace(pressure=cfg.P_dilute))
    new = ens.evaluate(T=old["Ts"], p={"O2": old["Ps"]})
    assert np.array_equal(new.G, old["G_tensor"])
    assert np.array_equal(new.populations, old["Pgi"])
    assert np.array_equal(new.phase_index, old["phase_index"])
    assert np.array_equal(new.composition("O"), old["x_per_non_o"])
    assert np.array_equal(ens.degeneracy, old["degeneracy"])
    for key, value in new.decomposition().items():
        assert np.array_equal(value, old[key]), key


@pytest.mark.parametrize("structure_vib", [True, False])
def test_gas_rows_and_support_match_legacy_patches(toy_files, tmp_path, structure_vib):
    """Gas rows with their own terms equal the 0.1 patches to float32 rounding, which 0.1 applied twice:
    translation + rotation on top of the structure terms (main maps), or full gas terms over a no-vibration
    baseline (the rutile maps). The support shift through ``shifted`` is exact."""
    from aist_v01.core.grandcanonical import run_gc
    from aist_v01.core.reservoir import patch_gas_reservoir_thermo, patch_gas_rot_trans
    from aist_v01.core.support import patch_support_stabilization

    paths, clusters, iro2 = toy_files
    npz = str(tmp_path / "legacy.npz")
    cfg = _legacy_config(paths["ir"], include_decomposition=False, include_v_product=structure_vib,
                         include_zpe_product=structure_vib, include_v_mu=structure_vib, include_zpe_mu=structure_vib)
    run_gc(paths["merged"], paths["o2"], npz, config=cfg)
    patch = patch_gas_rot_trans if structure_vib else patch_gas_reservoir_thermo
    patch(npz, [(len(clusters), iro2, 1.0)], use_vib_zpe=True)
    patch_support_stabilization(npz, paths["merged"], [len(clusters)], ev_per_ir=0.65, ev_per_o=0.1)
    old = dict(np.load(npz, allow_pickle=True))

    terms = ThermoTerms(vibrations=structure_vib, zpe=structure_vib, pressure=cfg.P_dilute)
    gas = (terms.replace(rotation=True, translation=True, pressure=1.0) if structure_vib
           else ThermoTerms.gas(pressure=1.0))
    ens = GrandCanonicalEnsemble(
        clusters + [iro2], references=[Reference(toy.metal_atom())],
        reservoirs=[Reservoir(toy.o2(), multiplicity=3,
                              terms=ThermoTerms.gas(vibrations=structure_vib, zpe=structure_vib))],
        terms=[terms] * len(clusters) + [gas],
    )
    support = -np.array([0.65 + 0.1 * (int(o) / int(i)) for o, i in zip(ens.count("O"), ens.count("Ir"))])
    support[-1] = 0.0
    new = ens.evaluate(T=old["Ts"], p={"O2": old["Ps"]}).shifted(support)
    assert np.allclose(new.G, old["G_tensor"], atol=2e-6)
    assert np.allclose(new.populations, old["Pgi"], atol=1e-5)
    assert np.array_equal(new.phase_index, old["phase_index"])


def test_variants_match_legacy_run_gc_variants(toy_files, tmp_path):
    from aist_v01.core.gc_variants import run_gc_variants

    paths, clusters, _ = toy_files
    npz = str(tmp_path / "variants.npz")
    run_gc_variants(paths["clusters"], paths["o2"], npz, grid_config=_legacy_config(paths["ir"]))
    old = np.load(npz, allow_pickle=True)
    ens = _ir_o2(clusters, terms=VIB.replace(pressure=0.01))
    for name, v, z in (("full", True, True), ("novib", False, False), ("vib_nozpe", True, False),
                       ("zpe_only", False, True)):
        new = ens.variant(vibrations=v, zpe=z).evaluate(T=old["Ts"], p={"O2": old["Ps"]})
        assert np.array_equal(new.G, old[f"{name}_G_tensor"]), name
        assert np.array_equal(new.populations, old[f"{name}_Pgi"]), name
        assert np.array_equal(new.phase_index, old[f"{name}_phase_index"]), name
