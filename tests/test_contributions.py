"""The free-energy contributions of one Atoms object (``aist.contributions``) against ASE and closed forms."""
import os
import sys

import numpy as np
import pytest
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.units import kB

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import toy  # noqa: E402
from aist import ThermoTerms, contributions, free_energy, vibrational, zero_point_energy  # noqa: E402


@pytest.mark.parametrize("make,geometry,sigma,spin,mult", [
    (toy.o2, "linear", 2, 1, 3), (toy.h2, "linear", 2, 0, 1), (toy.h2o, "nonlinear", 2, 0, 1),
])
def test_ideal_gas_free_energy_matches_ase(make, geometry, sigma, spin, mult):
    mol = make()
    nu = np.asarray(mol.info["frequencies"])[5 if geometry == "linear" else 6:]
    for T, p in ((300.0, 1e5), (800.0, 1e-3)):
        ase = IdealGasThermo(vib_energies=list(nu * toy.CM_TO_EV), potentialenergy=mol.get_potential_energy(),
                             atoms=mol, geometry=geometry, symmetrynumber=sigma, spin=spin)
        g = free_energy(mol, T, p, ThermoTerms.gas(), multiplicity=mult)
        assert g == pytest.approx(ase.get_gibbs_energy(T, p, verbose=False), abs=1e-4)


def test_vibrational_matches_ase_harmonic():
    cluster = toy.metal_oxide_set()[3]
    nu = np.asarray(cluster.info["frequencies"])[6:]
    for T in (150.0, 700.0):
        U, S = vibrational(cluster, T)
        ase = HarmonicThermo(vib_energies=list(nu * toy.CM_TO_EV), potentialenergy=0.0)
        assert U - T * S + zero_point_energy(cluster) == pytest.approx(ase.get_helmholtz_energy(T, verbose=False),
                                                                       abs=1e-6)


def test_default_is_the_electronic_contribution_only():
    cluster = toy.metal_oxide_set()[4]
    T = np.array([300.0, 900.0])
    J = cluster.info["total_spin"]["J"]
    assert np.allclose(free_energy(cluster, T), cluster.get_potential_energy() - kB * T * np.log(J))


def test_contributions_add_up_to_the_free_energy():
    o2 = toy.o2()
    T, p = np.array([250.0, 600.0, 1000.0]), np.array([1e5, 1.0, 1e-6])
    parts = contributions(o2, T, p, ThermoTerms.gas(), multiplicity=3)
    assert set(parts) == {"energy", "zpe", "vibrational", "rotational", "translational", "electronic"}
    assert np.allclose(sum(parts.values()), free_energy(o2, T, p, ThermoTerms.gas(), multiplicity=3), atol=1e-12)
    assert np.allclose(parts["electronic"], -kB * T * np.log(3))


def test_terms_are_switches():
    o2 = toy.o2()
    T, p = 500.0, 1e3
    full = free_energy(o2, T, p, ThermoTerms.gas(), multiplicity=3)
    for flag in ("vibrations", "zpe", "electronic", "rotation", "translation"):
        assert free_energy(o2, T, p, ThermoTerms.gas(**{flag: False}), multiplicity=3) != full, flag
    with pytest.raises(ValueError, match="pressure"):
        free_energy(o2, T, terms=ThermoTerms(translation=True))
    assert free_energy(o2, T, terms=ThermoTerms(translation=True, pressure=p), multiplicity=3) == pytest.approx(
        free_energy(o2, T, p, ThermoTerms(translation=True), multiplicity=3))


def test_missing_spin_is_degeneracy_one_with_a_warning():
    h2 = toy.h2()
    with pytest.warns(UserWarning, match="degeneracy 1"):
        g = free_energy(h2, 400.0)
    assert g == h2.get_potential_energy()


def test_zpe_falls_back_to_the_frequencies():
    cluster = toy.metal_oxide_set()[2]
    stored = cluster.info.pop("zpe")
    assert zero_point_energy(cluster) == pytest.approx(stored, rel=1e-6)
