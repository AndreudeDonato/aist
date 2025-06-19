import numpy as np
from aist.thermo import ThermoPartitionFunctions

def load_parameters(filepath):
    """Load fitted Taylor parameters from .npz file."""
    data = np.load(filepath)
    return [data[key] for key in ['a0', 'a1', 'a2', 'b0', 'b1', 'b2']]

def compute_thermo_US(T, atoms, params):
    """
    Compute vibrational entropy and internal energy at temperature T
    using fitted Taylor parameters.

    Parameters:
        T (float): Temperature in K
        atoms (ase.Atoms): ASE Atoms object
        params (list): [a0, a1, a2, b0, b1, b2]

    Returns:
        (U_vib, S_vib): entropy (eV/K), internal energy (eV)
    """
    a0, a1, a2, b0, b1, b2 = params
    thermo = ThermoPartitionFunctions()
    Nv = thermo.get_num_vibrational_modes(atoms)
    U = (a0 + a1*T + a2*T**2) * Nv
    S = (b0 + b1*T + b2*T**2) * Nv
    return U, S