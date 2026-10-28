"""aist: free-energy contributions of ASE ``Atoms`` objects, and building blocks for ensembles of them.

    from aist import ThermoTerms, free_energy, contributions

    G = free_energy(atoms, T, p, ThermoTerms(vibrations=True, zpe=True))      # E + chosen terms, eV
    parts = contributions(atoms, T, p, ThermoTerms.gas())                     # each term separately

How the free energies are combined into an ensemble is the user's choice: ``aist.ensembles`` has the general
pieces (``boltzmann_populations``, ``Reference``, ``Reservoir``) and a grand-canonical construction built on them
(``GrandCanonicalEnsemble``); ``examples/`` shows complete setups.
"""

__version__ = "0.2.0"

from aist.balance import BalanceError  # noqa: E402
from aist.contributions import (  # noqa: E402
    ThermoTerms,
    contributions,
    electronic,
    free_energy,
    rotational,
    translational,
    vibrational,
    zero_point_energy,
)
from aist.ensembles import (  # noqa: E402
    EnsembleResult,
    GrandCanonicalEnsemble,
    Reference,
    Reservoir,
    boltzmann_populations,
)

__all__ = [
    "BalanceError",
    "EnsembleResult",
    "GrandCanonicalEnsemble",
    "Reference",
    "Reservoir",
    "ThermoTerms",
    "boltzmann_populations",
    "contributions",
    "electronic",
    "free_energy",
    "rotational",
    "translational",
    "vibrational",
    "zero_point_energy",
    "__version__",
]
