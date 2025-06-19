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

    def get_vibrational(self, vib_freqs, T):
        return self.tp.get_vibrational(vib_freqs, T)

    def get_rotational(self, T, islinear=None, sigma=None):
        return self.tp.get_rotational(self.atoms, T, islinear=islinear, sigma=sigma)

    def get_electronic(self, states, T):
        return self.tp.get_electronic(states, T)

    def get_nuclear_spin(self, nuclear_spins):
        return self.tp.get_nuclear_spin(nuclear_spins)

    def get_total_thermo(self, T, P, vib_freqs, elec_states, nuclear_spins):
        return self.tp.get_total_thermo(
            self.atoms, T, P, vib_freqs, elec_states, nuclear_spins
        )

class ThermoPartitionFunctions:
    def __init__(self):
        self.h_planck = Planck  # J·s
        self.kb = Boltzmann     # J/K
        self.r_cons = R         # J/mol/K
        self.c = speed_of_light # m/s
        self.Na = Avogadro      # mol⁻¹

        self.eV_to_J = 1 / electron_volt  # eV⁻¹ × J = J/eV
        self.J_to_eV = electron_volt      # J/eV = eV

        self.amu_to_kg = physical_constants["atomic mass constant"][0]  # kg
        self.angstrom2_to_m2 = 1e-20  # m²
        self.Jmol_to_eV = self.eV_to_J / self.Na  # J/mol → eV

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
        if isinstance(P, float) or isinstance(P, int):
            P = np.asarray([P])
        else:
            P = np.asarray(P).reshape(-1, 1)
        if isinstance(T, float) or isinstance(T, int):
            T = np.asarray([T])
        else:
            T = np.asarray(T).reshape(1, -1)

        total_mass_kg = atoms.get_masses().sum() * self.amu_to_kg
        qtrans = ((2.0 * np.pi * total_mass_kg * self.kb * T / (self.h_planck ** 2)) ** (3.0 / 2.0)) * (self.kb * T / P)
        strans = self.r_cons * (np.log(qtrans) + 2.5)
        etrans = 1.5 * self.r_cons * T
        etrans_ev = etrans * self.Jmol_to_eV
        strans_ev = strans * self.Jmol_to_eV
        return etrans_ev.squeeze(), strans_ev.squeeze(), qtrans.squeeze()

    def get_vibrational(self, vib_freqs, T):
        Vib = np.asarray(vib_freqs).reshape(-1, 1)
        if isinstance(T, float) or isinstance(T, int):
            T = np.asarray([T])
        else:
            T = np.asarray(T).reshape(1, -1)
        
        vib_temp = Vib * 100.0 * self.c * self.h_planck / self.kb  # Convert cm^-1 to K
        
        q_ind_vib = (np.exp(-vib_temp / (2.0 * T))) / (1 - np.exp(-vib_temp / T))
        qvib = np.prod(q_ind_vib, axis=0)
        s_ind_vib = (vib_temp / T) / (np.exp(vib_temp / T) - 1) - np.log(1 - np.exp(-vib_temp / T))
        e_ind_vib = vib_temp * (0.5 + 1.0 / (np.exp(vib_temp / T) - 1))
        svib = self.r_cons * np.sum(s_ind_vib, axis=0)
        evib = self.r_cons * np.sum(e_ind_vib, axis=0)
        evib_ev = evib * self.Jmol_to_eV
        svib_ev = svib * self.Jmol_to_eV
        return evib_ev.squeeze(), svib_ev.squeeze(), qvib.squeeze()

    def get_rotational(self, atoms, T, islinear=None, sigma=None):
        T = np.atleast_1d(T)
        if len(atoms) == 1:
            return np.zeros_like(T), np.zeros_like(T), np.ones_like(T)
        if islinear is None:
            islinear = self.is_linear(atoms)
        if sigma is None:
            sigma = self.get_symmetry_number_from_point_group(atoms)
        m_I = atoms.get_moments_of_inertia()
        m_I = m_I * self.amu_to_kg * self.angstrom2_to_m2
        if islinear:
            I = m_I[-1]
            R_temp = (self.h_planck ** 2) / (8 * np.pi ** 2 * self.kb * I)
            qrot = (1 / sigma) * (T / R_temp)
            srot = self.r_cons * (np.log(qrot) + 1)
        else:
            R_temps = (self.h_planck ** 2) / (8 * np.pi ** 2 * self.kb * m_I)
            qrot = (np.sqrt(np.pi) / sigma) * (T ** 1.5) / np.sqrt(np.prod(R_temps))
            srot = self.r_cons * (np.log(qrot) + 1.5)
        erot = 1.5 * self.r_cons * T
        erot_ev = erot * self.Jmol_to_eV
        srot_ev = srot * self.Jmol_to_eV
        return erot_ev.squeeze(), srot_ev.squeeze(), qrot.squeeze()

    def get_electronic(self, states, T):
        T = np.atleast_1d(T)
        qelec = np.zeros_like(T)
        eelec = np.zeros_like(T)
        selec = np.zeros_like(T)
        for energy, mult in states:
            contrib = mult * np.exp(-energy * self.eV_to_J / (self.kb * T))
            qelec += contrib
            eelec += energy * contrib
            selec += contrib * (energy * self.eV_to_J / (self.kb * T))
        eelec /= qelec
        selec = np.log(qelec) + selec / qelec
        selec = self.r_cons * selec
        selec_ev = selec * self.Jmol_to_eV
        return eelec.squeeze(), selec_ev.squeeze(), qelec.squeeze()

    def get_nuclear_spin(self, nuclear_spins):
        qspin = 1
        for I in nuclear_spins:
            qspin *= int(2 * I + 1)
        sspin = self.kb * np.log(qspin) / 1e3
        sspin_ev = sspin * self.Jmol_to_eV
        return sspin_ev.squeeze(), qspin

    def get_total_thermo(self, atoms, T, P, vib_freqs, elec_states, nuclear_spins):
        etrans, strans, qtrans = self.get_translational(T, P, atoms)
        evib, svib, qvib = self.get_vibrational(vib_freqs, T)
        erot, srot, qrot = self.get_rotational(atoms, T)
        eelec, selec, qelec = self.get_electronic(elec_states, T)
        sspin, qspin = self.get_nuclear_spin(nuclear_spins)
        etot = etrans + evib + erot + eelec
        stot = strans + svib + srot + selec + sspin
        qtot = qtrans * qvib * qrot * qelec * qspin
        return etot, stot, qtot
