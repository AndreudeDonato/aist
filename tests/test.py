from ase import Atoms
from ase.calculators.emt import EMT
from ase.thermochemistry import IdealGasThermo
from aist.thermo import ThermoFromAtoms  # Your custom class
import numpy as np
from ase.units import kB

# Define molecules
methane = Atoms('CH4', positions=[
    (0.0000, 0.0000, 0.0000),
    (0.6276, 0.6276, 0.6276),
    (-0.6276, -0.6276, 0.6276),
    (-0.6276, 0.6276, -0.6276),
    (0.6276, -0.6276, -0.6276)
])

oxygen = Atoms('O2', positions=[
    (0.000, 0.000, -0.605),
    (0.000, 0.000, 0.605)
])

# Assign calculator
methane.calc = EMT()
oxygen.calc = EMT()

# Temperature and pressure
T = 298.15  # K
P = 101325  # Pa

# Vibrational frequencies in cm^-1
vib_CH4 = [1306, 1534, 2917, 3019, 1311, 2914, 1534, 3018, 1534]
vib_O2 = [1556]

# Electronic states
elec_CH4 = [(0.0, 1)]
elec_O2 = [(0.0, 3)]

# Nuclear spins
nuclear_spin_CH4 = [0, 0.5, 0.5, 0.5, 0.5]
nuclear_spin_O2 = [0, 0]

# === YOUR ThermoFromAtoms ===
thermo_CH4 = ThermoFromAtoms(methane)
thermo_O2 = ThermoFromAtoms(oxygen)

# === ASE IdealGasThermo ===
ase_thermo_CH4 = IdealGasThermo(vib_energies=[v * 1.23981e-4 for v in vib_CH4],  # cm^-1 to eV
                                potentialenergy=0.0,
                                atoms=methane,
                                geometry='nonlinear',
                                symmetrynumber=12,
                                spin=0)

ase_thermo_O2 = IdealGasThermo(vib_energies=[v * 1.23981e-4 for v in vib_O2],
                               potentialenergy=0.0,
                               atoms=oxygen,
                               geometry='linear',
                               symmetrynumber=2,
                               spin=1)


def compare_results(label, thermo, ase_thermo, vib, elec, spin, sigma, linear, nuclear_spin):
    print(f"\n=== {label} thermodynamic functions at {T} K ===")
    print(f"  Is linear: {linear}")
    print(f"  Rotational symmetry number (σ): {sigma}")

    # Your model
    etrans, strans, _ = thermo.get_translational(T, P)
    evib, svib, _ = thermo.get_vibrational(vib, T)
    erot, srot, _ = thermo.get_rotational(T, islinear=linear, sigma=sigma)
    eelec, selec, _ = thermo.get_electronic(elec, T)
    sspin, _ = thermo.get_nuclear_spin(nuclear_spin)

    # ASE model
    h_ase = ase_thermo.get_enthalpy(T)
    s_ase = ase_thermo.get_entropy(T, P)
    u_ase = h_ase - kB * T  # U = H - RT

    # Print header
    print(f"\n{'Component':<22} {'Your Energy (eV/mol)':>25} {'Your Entropy (eV/K/mol)':>30}")
    print("-" * 80)
    print(f"{'Translational':<22} {etrans:>25.5f} {strans:>30.5f}")
    print(f"{'Vibrational':<22} {evib:>25.5f} {svib:>30.5f}")
    print(f"{'Rotational':<22} {erot:>25.5f} {srot:>30.5f}")
    print(f"{'Electronic':<22} {eelec:>25.5f} {selec:>30.5f}")
    print(f"{'Nuclear Spin':<22} {'—':>25} {sspin:>30.5f}")

    print(f"\n{'ASE IdealGasThermo Results':<22}")
    print("-" * 80)
    print(f"{'Internal Energy (U)':<22} {u_ase:>25.5f} {'—':>30}")
    print(f"{'Entropy (S)':<22} {'—':>25} {s_ase:>30.5f}")
    print()



# Compare both molecules
compare_results("Methane (CH4)", thermo_CH4, ase_thermo_CH4, vib_CH4, elec_CH4, spin=0, sigma=12,
                linear=False, nuclear_spin=nuclear_spin_CH4)

compare_results("Oxygen (O2)", thermo_O2, ase_thermo_O2, vib_O2, elec_O2, spin=1, sigma=2,
                linear=True, nuclear_spin=nuclear_spin_O2)
