"""Checks of the thermodynamics core against ASE and closed-form expressions.

    python3 -m pytest tests/
"""
import os
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
from ase.thermochemistry import HarmonicThermo, IdealGasThermo
from ase.units import kB

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aist.composedthermo import compute_thermo_contributions  # noqa: E402
from aist.core.grandcanonical import GCConfig, compute_gc  # noqa: E402
from aist.stoichiometry import balance_reaction  # noqa: E402
from aist.thermo import ThermoPartitionFunctions  # noqa: E402

CM_TO_EV = 1.239841984e-4
NU_O2 = 1580.0


def _with_energy(atoms, energy, **info):
    atoms.calc = SinglePointCalculator(atoms, energy=energy)
    atoms.info.update(info)
    return atoms


def _o2(energy=-9.86):
    o2 = Atoms("O2", positions=[(0, 0, -0.605), (0, 0, 0.605)])
    return _with_energy(o2, energy, frequencies=[0.0] * 5 + [NU_O2], zpe=0.5 * NU_O2 * CM_TO_EV)


def _ch4():
    ch4 = Atoms("CH4", positions=[(0, 0, 0), (0.6276, 0.6276, 0.6276), (-0.6276, -0.6276, 0.6276),
                                  (-0.6276, 0.6276, -0.6276), (0.6276, -0.6276, -0.6276)])
    nu = sorted([1306, 1534, 2917, 3019, 1311, 2914, 1534, 3018, 1534])
    return _with_energy(ch4, -24.0, frequencies=[0.0] * 6 + nu, zpe=0.5 * sum(nu) * CM_TO_EV), nu


@pytest.mark.parametrize("T,p", [(150.0, 1e8), (300.0, 2.0265e4), (1000.0, 1e-5)])
def test_o2_chemical_potential_matches_ase(T, p):
    o2 = _o2()
    g, _ = compute_thermo_contributions(o2, np.array([T]), np.array([p]), elec_states=[(0.0, 3)])
    ase = IdealGasThermo(vib_energies=[NU_O2 * CM_TO_EV], potentialenergy=o2.get_potential_energy(),
                         atoms=o2, geometry="linear", symmetrynumber=2, spin=1)
    assert g[0] == pytest.approx(ase.get_gibbs_energy(T, p, verbose=False), abs=1e-4)


def test_nonlinear_molecule_matches_ase():
    ch4, nu = _ch4()
    T, p = 500.0, 1e5
    g, _ = compute_thermo_contributions(ch4, np.array([T]), np.array([p]), elec_states=[(0.0, 1)])
    ase = IdealGasThermo(vib_energies=[v * CM_TO_EV for v in nu], potentialenergy=-24.0,
                         atoms=ch4, geometry="nonlinear", symmetrynumber=12, spin=0)
    assert g[0] == pytest.approx(ase.get_gibbs_energy(T, p, verbose=False), abs=1e-4)


def test_rotation_does_not_depend_on_vibrations():
    o2 = _o2()
    T, p = np.array([300.0, 1000.0]), np.array([1e5, 1e5])

    def g(**flags):
        return compute_thermo_contributions(o2, T, p, elec_states=[(0.0, 3)], include_t=True, **flags)[0]

    rot_novib = g(include_v=False, include_zpe=False, include_r=True) - g(include_v=False, include_zpe=False, include_r=False)
    rot_vib = g(include_v=True, include_zpe=True, include_r=True) - g(include_v=True, include_zpe=True, include_r=False)
    np.testing.assert_allclose(rot_novib, rot_vib, atol=1e-10)
    inertia = o2.get_moments_of_inertia()[-1] * 1.66053906660e-27 * 1e-20  # kg m^2
    theta_rot = 1.054571817e-34 ** 2 / (2 * inertia * 1.380649e-23)  # K
    np.testing.assert_allclose(rot_novib, -kB * T * np.log(T / (2 * theta_rot)), rtol=1e-5)


def test_vibrational_matches_ase_harmonic():
    nu = np.array([25.0, 120.0, 480.0, 950.0])
    T = 700.0
    e, s, _ = ThermoPartitionFunctions().get_vibrational(nu, np.array([T]), zpe=True)
    ase = HarmonicThermo(vib_energies=list(nu * CM_TO_EV), potentialenergy=0.0)
    assert (e - T * s)[0] == pytest.approx(ase.get_helmholtz_energy(T, verbose=False), abs=1e-6)


def test_electronic_and_nuclear_spin_entropy():
    tp = ThermoPartitionFunctions()
    T = 300.0
    states = [(0.0, 1), (0.05, 3)]
    e, s, q = tp.get_electronic(states, np.array([T]))
    w = np.array([g * np.exp(-E / (kB * T)) for E, g in states])
    u = sum(E * wi for (E, _), wi in zip(states, w)) / w.sum()
    assert q[0] == pytest.approx(w.sum(), rel=1e-6)
    assert e[0] == pytest.approx(u, rel=1e-6)
    assert s[0] == pytest.approx(kB * np.log(w.sum()) + u / T, rel=1e-5)
    s_spin, q_spin = tp.get_nuclear_spin([0.5, 0.5])
    assert q_spin == 4
    assert s_spin == pytest.approx(kB * np.log(4), rel=1e-5)


def test_balance_reaction_is_per_ir():
    r, p = balance_reaction([Atoms("Ir"), Atoms("O2")], [Atoms("Ir4O6")])
    assert r == pytest.approx([-1.0, -0.75])
    assert p == pytest.approx([0.25])


def test_compute_gc_toy_ensemble(tmp_path):
    e_ir = -100.0
    ir = _with_energy(Atoms("Ir"), e_ir)
    o2 = _o2()
    structures = []
    for formula, pos, energy, J in (
        ("Ir2", [(0, 0, 0), (2.2, 0, 0)], -205.0, 3.0),
        ("Ir2O", [(0, 0, 0), (2.4, 0, 0), (1.2, 1.3, 0)], -212.0, 1.0),
        ("Ir2O2", [(0, 0, 0), (2.4, 0, 0), (1.2, 1.3, 0), (1.2, -1.3, 0)], -218.5, 1.0),
    ):
        a = Atoms(formula, positions=pos)
        n_rigid = 5 if len(a) == 2 else 6
        nu = sorted(100.0 + 150.0 * k for k in range(3 * len(a) - n_rigid))
        structures.append(_with_energy(a, energy, frequencies=[0.0] * n_rigid + nu, zpe=0.5 * sum(nu) * CM_TO_EV,
                                       total_spin={"N": J - 1, "S": (J - 1) / 2, "J": J},
                                       hirshfeld={"charges": [0.1] * len(a)}))
    write(tmp_path / "ir.traj", ir)
    write(tmp_path / "o2.traj", o2)
    write(tmp_path / "structures.traj", structures)

    cfg = GCConfig(T_min=300.0, T_max=900.0, n_T=4, P_min=1e-8, P_max=1e4, n_P=3,
                   reference_file=str(tmp_path / "ir.traj"), include_decomposition=False)
    gc = compute_gc(str(tmp_path / "structures.traj"), str(tmp_path / "o2.traj"), cfg, save_outputs=False)

    assert gc["G_tensor"].shape == (3, 3, 4)
    np.testing.assert_allclose(gc["Pgi"].sum(axis=0), 1.0, atol=1e-5)
    np.testing.assert_array_equal(gc["phase_index"], np.argmin(gc["G_tensor"], axis=0))

    # Ir-normalized formation free energy (Methods): G/y - E_Ir - x/(2y) mu_O2, mu_O2 from ASE
    j, k = 2, 1  # T index, p index
    T, p = gc["Ts"][j], gc["Ps"][k]
    mu = IdealGasThermo(vib_energies=[NU_O2 * CM_TO_EV], potentialenergy=o2.get_potential_energy(), atoms=o2,
                        geometry="linear", symmetrynumber=2, spin=1).get_gibbs_energy(T, p, verbose=False)
    for i, a in enumerate(structures):
        nu = np.asarray(a.info["frequencies"][5 if len(a) == 2 else 6:])
        f_vib = HarmonicThermo(vib_energies=list(nu * CM_TO_EV), potentialenergy=0.0).get_helmholtz_energy(T, verbose=False)
        g_i = a.get_potential_energy() + f_vib - kB * T * np.log(a.info["total_spin"]["J"])
        x = a.get_chemical_symbols().count("O")
        assert gc["G_tensor"][i, k, j] == pytest.approx(g_i / 2 - e_ir - x / 4 * mu, abs=2e-4)


def test_stoichiometry_weights_survive_underflow():
    from aist.core.vibrational import stoichiometry_weights

    Ts = np.array([300.0])
    # two isomers of one stoichiometry, 3 eV per Ir above the rest of the ensemble: their global
    # populations underflow in float32, the bucket weights must not
    G = np.array([0.0, 3.0, 3.02]).reshape(3, 1, 1)
    deg = np.array([1.0, 1.0, 2.0])
    w = np.exp(-(G - G.min(axis=0)) / (kB * Ts)) * deg[:, None, None]
    pgi = (w / w.sum(axis=0)).astype(np.float32)
    assert pgi[1:, 0, 0].sum() == 0.0
    gc = {"Ts": Ts, "G_tensor": G, "degeneracy": deg}
    got = stoichiometry_weights(gc, [1, 2], 0, 0)
    expect = np.array([1.0, 2.0 * np.exp(-0.02 / (kB * 300.0))])
    np.testing.assert_allclose(got, expect / expect.sum(), rtol=1e-6)
    # where nothing underflows the bucket weights equal the renormalized populations
    np.testing.assert_allclose(stoichiometry_weights(gc, [0], 0, 0), [1.0])
