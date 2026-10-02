import numpy as np
from scipy.constants import Planck, Boltzmann, R, speed_of_light, Avogadro, electron_volt, physical_constants
from pymatgen.core.structure import Molecule
from pymatgen.symmetry.analyzer import PointGroupAnalyzer

# Rotational symmetry number lookup table for common point groups
POINT_GROUP_SIGMA = {
    "C1": 1, "Ci": 1, "Cs": 1,
    "C2": 2, "C3": 3, "C4": 4, "C5": 5, "C6": 6,
    "C2v": 2, "C3v": 3, "C4v": 4, "C5v": 5, "C6v": 6,
    "C2h": 2, "C3h": 3, "C4h": 4, "C6h": 6,
    "S4": 2, "S6": 3, "S8": 4,
    "D2": 4, "D3": 6, "D4": 8, "D5": 10, "D6": 12,
    "D2h": 4, "D3h": 6, "D4h": 8, "D5h": 10, "D6h": 12,
    "D2d": 4, "D3d": 6, "D4d": 8, "D5d": 10, "D6d": 12,
    "T": 12, "Td": 12, "Th": 12,
    "O": 24, "Oh": 24,
    "I": 60, "Ih": 60,
    "C∞v": 1, "D∞h": 2, "C*v": 1, "D*h": 2
}

POINT_GROUP_DEGEN = {
    # chiral groups
    "C1": 2, "C2": 2, "C3": 2, "C4": 2, "C5": 2, "C6": 2,
    "D2": 2, "D3": 2, "D4": 2, "D5": 2, "D6": 2,
    "T": 2, "O": 2, "I": 2,
    
    # achiral groups
    "Ci": 1, "Cs": 1,
    "C2v": 1, "C3v": 1, "C4v": 1, "C5v": 1, "C6v": 1,
    "C2h": 1, "C3h": 1, "C4h": 1, "C6h": 1,
    "S4": 1, "S6": 1, "S8": 1,
    "D2h": 1, "D3h": 1, "D4h": 1, "D5h": 1, "D6h": 1,
    "D2d": 1, "D3d": 1, "D4d": 1, "D5d": 1, "D6d": 1,
    "Td": 1, "Th": 1, "Oh": 1, "Ih": 1,
    "C∞v": 1, "D∞h": 1, "C*v": 1, "D*h": 1
}

class ThermoFromAtoms:
    def __init__(self, atoms):
        self.atoms = atoms
        self.tp = ThermoPartitionFunctions()

    def is_linear(self, tol=1e-3):
        return self.tp.is_linear(self.atoms, tol=tol)

    def get_symmetry_number_from_point_group(self):
        return self.tp.get_symmetry_number_from_point_group(self.atoms)

    def get_translational(self, T, P):
        return self.tp.get_translational(T, P, self.atoms)

    def get_vibrational(self, vib_freqs, T, zpe=False):
        return self.tp.get_vibrational(vib_freqs, T, zpe=zpe)

    def get_rotational(self, T, islinear=None, sigma=None):
        return self.tp.get_rotational(self.atoms, T, islinear=islinear, sigma=sigma)

    def get_electronic(self, states, T):
        return self.tp.get_electronic(states, T)

    def get_nuclear_spin(self, nuclear_spins):
        return self.tp.get_nuclear_spin(nuclear_spins)

    def get_total_thermo(self, T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=False):
        return self.tp.get_total_thermo(
            self.atoms, T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=include_zpe
        )
    
    def get_enthalpy(self, T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=False):
        """Compute H = U + RT"""
        etot, _, _ = self.get_total_thermo(T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=include_zpe)
        H = etot + self.tp.r_cons * T * self.tp.Jmol_to_eV  # RT term
        return H

    def get_gibbs(self, T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=False):
        """Compute G = H - T*S"""
        etot, stot, _ = self.get_total_thermo(T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=include_zpe)
        H = etot + self.tp.r_cons * T * self.tp.Jmol_to_eV
        G = H - T * stot
        return G

class ThermoPartitionFunctions:
    def __init__(self):
        self.h_planck = Planck
        self.kb = Boltzmann
        self.r_cons = R
        self.c = speed_of_light
        self.Na = Avogadro

        self.eV_to_J = electron_volt
        self.J_to_eV = 1 / electron_volt
        self.amu_to_kg = physical_constants["atomic mass constant"][0]
        self.angstrom2_to_m2 = 1e-20
        self.Jmol_to_eV = self.J_to_eV / self.Na

    def get_inertia_tensor(self, atoms):
        positions = atoms.get_positions() - atoms.get_center_of_mass()
        masses = atoms.get_masses()
        I = np.zeros((3, 3))
        for m, (x, y, z) in zip(masses, positions):
            I[0, 0] += m * (y**2 + z**2)
            I[1, 1] += m * (x**2 + z**2)
            I[2, 2] += m * (x**2 + y**2)
            I[0, 1] += -m * x * y
            I[1, 0] += -m * x * y
            I[0, 2] += -m * x * z
            I[2, 0] += -m * x * z
            I[1, 2] += -m * y * z
            I[2, 1] += -m * y * z
        return I

    def is_linear(self, atoms, tol=1e-3):
        I = self.get_inertia_tensor(atoms)
        eigvals = np.linalg.eigvalsh(I)
        return np.count_nonzero(eigvals < tol) == 1
    
    def get_num_vibrational_modes(self, atoms):
        n_atoms = len(atoms)
        if n_atoms < 2:
            return 0
        return 3 * n_atoms - (5 if self.is_linear(atoms) else 6)

    def get_symmetry_number_from_point_group(self, atoms):
        mol = Molecule(species=atoms.get_chemical_symbols(), coords=atoms.get_positions())
        pga = PointGroupAnalyzer(mol)
        point_group = pga.sch_symbol
        sigma = POINT_GROUP_SIGMA.get(point_group, None)
        if sigma is None:
            raise ValueError(f"Unknown or unsupported point group: '{point_group}'")
        return sigma

    def get_translational(self, T, P, atoms):
        T = np.asarray(T)
        P = np.asarray(P)
        T, P = np.broadcast_arrays(T, P)
        total_mass_kg = atoms.get_masses().sum() * self.amu_to_kg

        qtrans = ((2.0 * np.pi * total_mass_kg * self.kb * T / (self.h_planck ** 2)) ** (3.0 / 2.0)) * (self.kb * T / P)
        strans = self.r_cons * (np.log(qtrans) + 2.5)
        etrans = 1.5 * self.r_cons * T
        return etrans * self.Jmol_to_eV, strans * self.Jmol_to_eV, qtrans

    def get_vibrational(self, vib_freqs, T, zpe=False):
        Vib = np.asarray(vib_freqs).reshape(-1, 1)
        T = np.asarray(T)
        T_shape = T.shape
        T_flat = T.ravel()[np.newaxis, :]

        vib_temp = Vib * 100.0 * self.c * self.h_planck / self.kb
        if zpe:
            q_ind_vib = np.exp(-vib_temp / (2.0 * T_flat)) / (1 - np.exp(-vib_temp / T_flat))
            e_ind_vib = vib_temp * (0.5 + 1.0 / (np.exp(vib_temp / T_flat) - 1))
        else:
            q_ind_vib = 1 / (1 - np.exp(-vib_temp / T_flat))
            e_ind_vib = vib_temp / (np.exp(vib_temp / T_flat) - 1)

        s_ind_vib = (vib_temp / T_flat) / (np.exp(vib_temp / T_flat) - 1) - np.log(1 - np.exp(-vib_temp / T_flat))
        
        qvib = np.prod(q_ind_vib, axis=0)
        svib = self.r_cons * np.sum(s_ind_vib, axis=0)
        evib = self.r_cons * np.sum(e_ind_vib, axis=0)

        return evib.reshape(T_shape) * self.Jmol_to_eV, svib.reshape(T_shape) * self.Jmol_to_eV, qvib.reshape(T_shape)

    def get_rotational(self, atoms, T, islinear=None, sigma=None):
        T = np.asarray(T)
        T_shape = T.shape
        T_flat = T.ravel()
        if len(atoms) == 1:
            zeros = np.zeros_like(T_flat)
            ones = np.ones_like(T_flat)
            return zeros.reshape(T_shape), zeros.reshape(T_shape), ones.reshape(T_shape)
        if islinear is None:
            islinear = self.is_linear(atoms)
        if sigma is None:
            sigma = self.get_symmetry_number_from_point_group(atoms)
        m_I = atoms.get_moments_of_inertia() * self.amu_to_kg * self.angstrom2_to_m2
        if islinear:
            I = m_I[-1]
            R_temp = (self.h_planck ** 2) / (8 * np.pi ** 2 * self.kb * I)
            qrot = (1 / sigma) * (T_flat / R_temp)
            srot = self.r_cons * (np.log(qrot) + 1)
            erot = self.r_cons * T_flat
        else:
            R_temps = (self.h_planck ** 2) / (8 * np.pi ** 2 * self.kb * m_I)
            qrot = (np.sqrt(np.pi) / sigma) * (T_flat ** 1.5) / np.sqrt(np.prod(R_temps))
            srot = self.r_cons * (np.log(qrot) + 1.5)
            erot = 1.5 * self.r_cons * T_flat
        return erot.reshape(T_shape) * self.Jmol_to_eV, srot.reshape(T_shape) * self.Jmol_to_eV, qrot.reshape(T_shape)

    def get_electronic(self, states, T):
        T = np.asarray(T)
        T_shape = T.shape
        T_flat = T.ravel()

        qelec = np.zeros_like(T_flat)
        eelec = np.zeros_like(T_flat)
        selec = np.zeros_like(T_flat)

        for energy, mult in states:
            contrib = mult * np.exp(-energy * self.eV_to_J / (self.kb * T_flat))
            qelec += contrib
            eelec += energy * contrib
            selec += contrib * (energy * self.eV_to_J / (self.kb * T_flat))

        eelec /= qelec
        selec = np.log(qelec) + selec / qelec
        selec = self.r_cons * selec
        selec_ev = selec * self.Jmol_to_eV

        return eelec.reshape(T_shape), selec_ev.reshape(T_shape), qelec.reshape(T_shape)

    def get_nuclear_spin(self, nuclear_spins):
        qspin = 1
        for I in nuclear_spins:
            qspin *= int(2 * I + 1)
        sspin = self.r_cons * np.log(qspin)
        return sspin * self.Jmol_to_eV, qspin

    def get_total_thermo(self, atoms, T, P, vib_freqs, elec_states, nuclear_spins, include_zpe=False):
        # Translational
        etrans, strans, qtrans = self.get_translational(T, P, atoms)
        
        # Vibrational (optionally include ZPE)
        evib, svib, qvib = self.get_vibrational(vib_freqs, T, zpe=include_zpe)
        
        # Rotational
        erot, srot, qrot = self.get_rotational(atoms, T)
        
        # Electronic
        eelec, selec, qelec = self.get_electronic(elec_states, T)
        
        # Nuclear spin
        sspin, qspin = self.get_nuclear_spin(nuclear_spins)
        
        # Total energy and entropy
        etot = etrans + evib + erot + eelec
        stot = strans + svib + srot + selec + sspin
        
        # Total partition function
        qtot = qtrans * qvib * qrot * qelec * qspin
        
        return etot, stot, qtot
