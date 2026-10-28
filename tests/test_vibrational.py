"""Vibrational analysis helpers (``aist.core.vibrational``)."""
import os
import sys

import numpy as np
import pytest
from ase.thermochemistry import HarmonicThermo
from ase.units import kB

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import toy  # noqa: E402
from aist.core.vibrational import compute_vib_gibbs_contribution, stoichiometry_weights  # noqa: E402


def test_vib_gibbs_contribution_matches_ase_harmonic():
    cluster = toy.metal_oxide_set()[3]
    nu = np.asarray(cluster.info["frequencies"])[6:]
    for T in (200.0, 800.0):
        ase = HarmonicThermo(vib_energies=list(nu * toy.CM_TO_EV), potentialenergy=0.0)
        assert compute_vib_gibbs_contribution(cluster, T) == pytest.approx(ase.get_helmholtz_energy(T, verbose=False),
                                                                           abs=1e-6)


def test_stoichiometry_weights_survive_underflow():
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
    np.testing.assert_allclose(stoichiometry_weights(gc, [0], 0, 0), [1.0])
