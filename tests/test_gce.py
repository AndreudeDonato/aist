"""Tests of the species-agnostic grand-canonical core (``aist.gce``).

Physics checks against ASE and closed forms, generality (other metals, other reservoirs, two reservoirs), and
continuity: on toy data the new core must reproduce the frozen 0.1 code (``tests/legacy/aist_v01``) bit for bit.
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
    BalanceError, GasSpecies, GrandCanonicalEnsemble, Reference, Reservoir, SupportShift, ThermoTerms,
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


# --------------------------------------------------------------------------- balance
def test_balance_matches_balance_reaction_exactly():
    from aist.stoichiometry import balance_reaction

    ens = _ir_o2()
    for i, atoms in enumerate(ens.structures):
        r, p = balance_reaction([toy.metal_atom(), toy.o2()], [atoms])
        assert ens._p[i] == p[0] and ens._c_ref[i, 0] == r[0] and ens._c_res[i, 0] == r[1]


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


# --------------------------------------------------------------------------- reservoirs
def test_reservoir_multiplicity_only_needed_for_the_electronic_term():
    with pytest.raises(ValueError, match="multiplicity"):
        Reservoir(toy.o2())
    assert Reservoir(toy.o2(with_spin=True)).electronic_states == [(0.0, 3.0)]
    no_elec = Reservoir(toy.h2(), terms=ThermoTerms.gas(electronic=False))
    singlet = Reservoir(toy.h2(), multiplicity=1)
    T, p = np.array([300.0, 800.0]), np.array([1e5, 1e-3])
    assert np.array_equal(no_elec.chemical_potential(T, p), singlet.chemical_potential(T, p))  # ln 1 = 0
    with pytest.raises(ValueError, match="electronic term is on"):
        no_elec.with_terms(electronic=True)


def test_default_terms_are_electronic_only():
    from scipy.constants import Boltzmann, electron_volt

    ens = _ir_o2()
    assert ens.terms == ThermoTerms(vibrations=False, zpe=False, electronic=True, rotation=False, translation=False)
    T = 600.0
    res = ens.evaluate(T=T, p={"O2": 1e3}, dtype=np.float64)
    mu = ens.reservoirs[0].chemical_potential(np.array([T]), np.array([1e3]))[0]
    for i, a in enumerate(ens.structures):
        n_ir, n_o = a.get_chemical_symbols().count("Ir"), a.get_chemical_symbols().count("O")
        g = a.get_potential_energy() - Boltzmann / electron_volt * T * np.log(a.info["total_spin"]["J"])
        assert res.G[i] == pytest.approx(g / n_ir - toy.metal_atom().get_potential_energy() - n_o / (2 * n_ir) * mu,
                                         abs=1e-9)
    with pytest.raises(ValueError, match="pressure"):
        _ir_o2(terms=ThermoTerms(translation=True)).evaluate(T=TS, p={"O2": PS})


def test_every_term_is_a_switch():
    """Each contribution can be left out on structures, reservoirs and gas species alike."""
    full = _ir_o2(terms=VIB).evaluate(T=TS, p={"O2": PS}, dtype=np.float64)
    for flag in ("vibrations", "zpe", "electronic"):
        off = _ir_o2(terms=VIB.replace(**{flag: False})).evaluate(T=TS, p={"O2": PS}, dtype=np.float64)
        assert not np.allclose(off.G, full.G), flag
    res_off = GrandCanonicalEnsemble(
        toy.metal_oxide_set(), references=[Reference(toy.metal_atom())],
        reservoirs=[Reservoir(toy.o2(), terms=ThermoTerms.gas(electronic=False, rotation=False))], terms=VIB,
    ).evaluate(T=TS, p={"O2": PS}, dtype=np.float64)
    assert not np.allclose(res_off.G, full.G)
    iro2 = toy.with_data(Atoms("IrO2", positions=[(0, 0, 0), (1.2, 1.1, 0), (-1.2, 1.1, 0)]), -111.5, J=2.0)
    static = _ir_o2(gas=[GasSpecies(iro2, pressure=1.0, rotation=False, translation=False)])
    plain = _ir_o2(toy.metal_oxide_set() + [iro2])
    assert np.array_equal(static.evaluate(T=TS, p={"O2": PS}).G, plain.evaluate(T=TS, p={"O2": PS}).G)


@pytest.mark.parametrize("make,geometry,sigma,spin,mult", [
    (toy.o2, "linear", 2, 1, 3), (toy.h2, "linear", 2, 0, 1), (toy.h2o, "nonlinear", 2, 0, 1),
])
def test_reservoir_mu_matches_ase(make, geometry, sigma, spin, mult):
    mol = make()
    res = Reservoir(mol, multiplicity=mult)
    nu = np.asarray(mol.info["frequencies"])
    nu = nu[5:] if geometry == "linear" else nu[6:]
    for T, p in ((300.0, 1e5), (800.0, 1e-3)):
        ase = IdealGasThermo(vib_energies=list(nu * toy.CM_TO_EV), potentialenergy=mol.get_potential_energy(),
                             atoms=mol, geometry=geometry, symmetrynumber=sigma, spin=spin)
        mu = res.chemical_potential(np.array([T]), np.array([p]))[0]
        assert mu == pytest.approx(ase.get_gibbs_energy(T, p, verbose=False), abs=1e-4)


def test_h2_reservoir_has_no_o2_triplet():
    ens = GrandCanonicalEnsemble(toy.metal_oxide_set(ligand="H"), references=[Reference(toy.metal_atom())],
                                 reservoirs=[Reservoir(toy.h2(), multiplicity=1)])
    res = ens.evaluate(T=TS, p={"H2": PS})
    assert np.array_equal(ens.normalization, ens.count("Ir").astype(float))
    assert np.all(res.composition("H") > 0)
    mu = ens.reservoirs[0].chemical_potential(np.array([500.0]), np.array([1e5]))[0]
    ase = IdealGasThermo(vib_energies=[4401.0 * toy.CM_TO_EV], potentialenergy=-6.77, atoms=toy.h2(),
                         geometry="linear", symmetrynumber=2, spin=0).get_gibbs_energy(500.0, 1e5, verbose=False)
    assert mu == pytest.approx(ase, abs=1e-4)


# --------------------------------------------------------------------------- generality
def test_platinum_equals_iridium_relabelled():
    ir = _ir_o2().evaluate(T=TS, p={"O2": PS})
    pt = GrandCanonicalEnsemble(toy.metal_oxide_set(metal="Pt"), references=[Reference(toy.metal_atom("Pt"))],
                                reservoirs=[Reservoir(toy.o2(), multiplicity=3)]).evaluate(T=TS, p={"O2": PS})
    assert np.array_equal(ir.G, pt.G) and np.array_equal(ir.populations, pt.populations)


def test_fixed_temperature_map_with_two_reservoirs():
    ens = GrandCanonicalEnsemble(toy.hydroxide_set(), references=[Reference(toy.metal_atom())],
                                 reservoirs=[Reservoir(toy.o2(), multiplicity=3), Reservoir(toy.h2o(), multiplicity=1)])
    p_o2, p_h2o = np.logspace(-10, 5, 7), np.logspace(-10, 5, 6)
    res = ens.evaluate(T=500.0, p={"O2": p_o2, "H2O": p_h2o})
    assert res.shape == (7, 6) and list(res.axes) == ["O2", "H2O"]
    assert np.allclose(res.populations.sum(axis=0), 1.0, atol=1e-5)
    h = res.composition("H")
    assert np.all(np.diff(h, axis=1) >= -1e-7)  # more water, at least as much H
    flipped = ens.evaluate(T=500.0, p={"O2": p_o2, "H2O": p_h2o}, axes=["H2O", "O2"])
    assert np.array_equal(flipped.populations, np.transpose(res.populations, (0, 2, 1)))


def test_reservoir_basis_change_gives_same_populations():
    o2, h2o, h2 = (Reservoir(toy.o2(), multiplicity=3), Reservoir(toy.h2o(), multiplicity=1),
                   Reservoir(toy.h2(), multiplicity=1))
    ref = [Reference(toy.metal_atom())]
    T, p_o2, p_h2o = 600.0, 1e2, np.logspace(-8, 4, 9)
    a = GrandCanonicalEnsemble(toy.hydroxide_set(), references=ref, reservoirs=[o2, h2o])
    b = GrandCanonicalEnsemble(toy.hydroxide_set(), references=ref, reservoirs=[o2, h2])
    mu_h2o = h2o.chemical_potential(np.full(9, T), p_h2o)
    mu_o2 = o2.chemical_potential(np.array([T]), np.array([p_o2]))[0]
    dmu_h2 = mu_h2o - 0.5 * mu_o2 - h2.energy  # equilibrium H2O = H2 + 1/2 O2
    pa = a.evaluate(T=T, p={"O2": p_o2, "H2O": p_h2o}, dtype=np.float64).populations
    pb = b.evaluate(T=T, p={"O2": p_o2}, dmu={"H2": dmu_h2}, dtype=np.float64).populations
    assert np.allclose(pa, pb, atol=1e-9)


def test_dmu_axis_equals_pressure_axis():
    ens = _ir_o2()
    res_p = ens.evaluate(T=700.0, p={"O2": PS}, dtype=np.float64)
    mu = ens.reservoirs[0].chemical_potential(np.full(len(PS), 700.0), PS)
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


# --------------------------------------------------------------------------- variants, replacements, views
def test_variant_equals_fresh_ensemble():
    novib = _ir_o2(terms=VIB).variant(vibrations=False, zpe=False).evaluate(T=TS, p={"O2": PS})
    fresh = GrandCanonicalEnsemble(
        toy.metal_oxide_set(), references=[Reference(toy.metal_atom())],
        reservoirs=[Reservoir(toy.o2(), multiplicity=3, terms=ThermoTerms.gas(vibrations=False, zpe=False))],
    ).evaluate(T=TS, p={"O2": PS})
    assert np.array_equal(novib.G, fresh.G) and np.array_equal(novib.populations, fresh.populations)


def test_with_structures_recomputes_only_the_new_data():
    ens = _ir_o2(terms=VIB)
    stiffer = toy.metal_oxide_set()
    for a in stiffer:
        a.info["frequencies"] = [f * 1.1 for f in a.info["frequencies"]]
        a.info["zpe"] *= 1.1
    swapped = ens.with_structures(stiffer).evaluate(T=TS, p={"O2": PS})
    fresh = _ir_o2(stiffer, terms=VIB).evaluate(T=TS, p={"O2": PS})
    assert np.array_equal(swapped.G, fresh.G)
    assert not np.array_equal(swapped.G, ens.evaluate(T=TS, p={"O2": PS}).G)
    with pytest.raises(ValueError, match="formula"):
        ens.with_structures(toy.metal_oxide_set()[::-1])


def test_subset_renormalises_and_support_shift_spares_gas():
    iro2 = toy.with_data(Atoms("IrO2", positions=[(0, 0, 0), (1.2, 1.1, 0), (-1.2, 1.1, 0)]), -111.5, J=2.0)
    ens = _ir_o2(gas=[GasSpecies(iro2, pressure=1.0)])
    res = ens.evaluate(T=TS, p={"O2": PS})
    sub = res.subset(counts={"Ir": 3})
    assert np.allclose(sub.populations.sum(axis=0), 1.0, atol=1e-6) and set(sub.formulas) == {"Ir3O2", "Ir3O4"}
    shifted = res.shifted(SupportShift(0.65, {"O": 0.1}))
    d = (shifted.G.astype(float) - res.G.astype(float))[:, 0, 0]
    expected = -(0.65 + 0.1 * ens.count("O") / ens.normalization)
    expected[ens.is_gas] = 0.0
    assert np.allclose(d, expected, atol=1e-5)


def test_degeneracy_counts_both_enantiomers():
    ens = _ir_o2()
    assert set(np.unique(ens.degeneracy)) <= {1.0, 2.0} and 2.0 in ens.degeneracy
    flat = _ir_o2(degeneracy=None).evaluate(T=TS, p={"O2": PS})
    chiral = ens.evaluate(T=TS, p={"O2": PS})
    w = flat.populations.astype(float) * ens.degeneracy[:, None, None]
    assert np.allclose(chiral.populations, w / w.sum(axis=0), atol=1e-6)


# --------------------------------------------------------------------------- continuity with the 0.1 code
@pytest.fixture
def toy_files(tmp_path):
    clusters = toy.metal_oxide_set()
    iro2 = toy.with_data(Atoms("IrO2", positions=[(0, 0, 0), (1.2, 1.1, 0), (-1.2, 1.1, 0)]), -111.5, J=2.0)
    paths = {name: str(tmp_path / f"{name}.traj") for name in ("ir", "o2", "clusters", "merged", "iro2")}
    write(paths["ir"], toy.metal_atom())
    write(paths["o2"], toy.o2())
    write(paths["clusters"], clusters)
    write(paths["iro2"], iro2)
    write(paths["merged"], clusters + [iro2])
    return paths, clusters, iro2


def _legacy_config(ir_path, **kw):
    from aist_v01.core.grandcanonical import GCConfig

    return GCConfig(T_min=200.0, T_max=900.0, n_T=6, P_min=1e-8, P_max=1e4, n_P=5, reference_file=ir_path, **kw)


def test_core_matches_legacy_compute_gc(toy_files, tmp_path):
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


@pytest.mark.parametrize("gas_vib", [True, False])
def test_gas_species_and_support_match_legacy_patches(toy_files, tmp_path, gas_vib):
    from aist_v01.core.grandcanonical import run_gc
    from aist_v01.core.reservoir import patch_gas_rot_trans, patch_gas_strip_vib_zpe
    from aist_v01.core.support import patch_support_stabilization

    paths, clusters, iro2 = toy_files
    npz = str(tmp_path / "legacy.npz")
    cfg = _legacy_config(paths["ir"], include_decomposition=False)
    run_gc(paths["merged"], paths["o2"], npz, config=cfg)
    gas_rows = [(len(clusters), iro2, 1.0)]
    if not gas_vib:
        patch_gas_strip_vib_zpe(npz, gas_rows)
    patch_gas_rot_trans(npz, gas_rows, use_vib_zpe=True)
    patch_support_stabilization(npz, paths["merged"], [len(clusters)], ev_per_ir=0.65, ev_per_o=0.1)
    old = dict(np.load(npz, allow_pickle=True))

    ens = _ir_o2(clusters, terms=VIB.replace(pressure=cfg.P_dilute),
                 gas=[GasSpecies(iro2, pressure=1.0, vibrations=gas_vib)])
    new = ens.evaluate(T=old["Ts"], p={"O2": old["Ps"]}).shifted(SupportShift(0.65, {"O": 0.1}))
    assert np.array_equal(new.G, old["G_tensor"])
    assert np.array_equal(new.populations, old["Pgi"])
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
