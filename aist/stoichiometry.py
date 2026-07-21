from sympy import Matrix
from collections import Counter


def get_element_counts(atoms_list):
    """Return list of dicts with element counts for each Atoms object."""
    counts = []
    for atoms in atoms_list:
        count = Counter(atoms.get_chemical_symbols())
        counts.append(dict(count))
    return counts


def build_element_matrix(reactants, products):
    """Builds the matrix for solving stoichiometry."""
    all_species = get_element_counts(reactants + products)
    elements = sorted({el for d in all_species for el in d})
    
    matrix = []
    for el in elements:
        row = []
        for species in get_element_counts(reactants):
            row.append(species.get(el, 0))
        for species in get_element_counts(products):
            row.append(-species.get(el, 0))
        matrix.append(row)
    
    return Matrix(matrix)


def balance_reaction(reactants, products):
    """Return stoichiometric coefficients with the first reactant normalised to 1.0.

    For Ir + O2 -> Ir_n O_m this yields r = [-1, -m/(2n)], p = [1/n], i.e.
    formation free energy per Ir atom (G_i / n_Ir).
    """
    M = build_element_matrix(reactants, products)
    nullspace = M.nullspace()

    if not nullspace:
        raise ValueError("No solution found for stoichiometry.")

    vec = nullspace[0]
    vec = vec / vec[0]  # normalize first reactant to 1.0

    coeffs = [float(term.evalf()) for term in vec]  # convert to floats

    product_coeffs = coeffs[len(reactants):]
    reactant_coeffs = [-c for c in coeffs[:len(reactants)]]  # make products positive

    return reactant_coeffs, product_coeffs


def format_reaction(reactants, products, r_coeffs, p_coeffs):
    def fmt(species, coeff):
        return f"{coeff if coeff > 1 else ''}{species.get_chemical_formula()}"

    r_str = " + ".join(fmt(a, c) for a, c in zip(reactants, r_coeffs))
    p_str = " + ".join(fmt(a, c) for a, c in zip(products, p_coeffs))
    return f"{r_str} -> {p_str}"
