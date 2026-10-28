"""Ensemble building blocks and the generalised grand-canonical construction (``aist.ensembles``).

Generality (other metals, other reservoirs, two reservoirs at once), the exact balance, and the parts of ΔG.
"""
import os
import sys

from fractions import Fraction

import numpy as np
import pytest
from ase import Atoms
from ase.thermochemistry import IdealGasThermo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import toy  # noqa: E402
from aist import (  # noqa: E402
    BalanceError, GrandCanonicalEnsemble, Reference, Reservoir, ThermoTerms, boltzmann_populations, free_energy,
    vibrational, zero_point_energy,
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
def test_balance_coefficients_are_exact():
    ens = _ir_o2()
    for i, atoms in enumerate(ens.structures):
        x, y = atoms.get_chemical_symbols().count("Ir"), atoms.get_chemical_symbols().count("O")
        assert ens._p[i] == float(Fraction(1, x)) and ens._c[i, 0] == -1.0
        assert ens._c[i, 1] == float(Fraction(-y, 2 * x))


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


# --------------------------------------------------------------------------- parts of ΔG
def test_contributions_per_row_add_up_and_are_assembled_from_the_core():
    clusters = toy.metal_oxide_set()
    ens = _ir_o2(clusters, terms=VIB)
    T, p = np.linspace(200.0, 900.0, 6), 3e2
    res = ens.evaluate(T=T, p={"O2": p}, dtype=np.float64)
    parts = res.contributions()
    assert np.allclose(sum(parts.values()), res.G, atol=1e-9)
    assert set(res.contributions(names=["vibrational"])) == {"vibrational"}
    o2 = ens.reservoirs[0].atoms
    ev2, sv2 = vibrational(o2, T)
    for i, atoms in enumerate(clusters):
        ev, sv = vibrational(atoms, T)
        x, y = atoms.get_chemical_symbols().count("Ir"), atoms.get_chemical_symbols().count("O")
        expected = (ev + zero_point_energy(atoms) - T * sv) / x - y / (2 * x) * (ev2 + zero_point_energy(o2) - T * sv2)
        assert np.allclose(parts["vibrational"][i], expected, atol=1e-12)
    w = res.populations.astype(float)
    for name, key in (("dft", "E_dft_avg"), ("vibrational", "E_vib_avg"), ("electronic", "E_elec_avg")):
        assert np.allclose(res.decomposition()[key], np.sum(w * parts[name], axis=0), atol=1e-12)
