"""Energy feature."""

import numpy as np

from .core import Simulation


def apply_energy(sim: Simulation, upkeep=(0, 1, 1, 3, 2)):
    """
    Apply energy calculations to the simulation.

    This function deducts energy based on cell roles and handles cell death
    when energy is depleted.
    """
    if sim.energy is None:
        sim.energy = np.full_like(sim.board, 255, dtype=np.uint8)

    cost = np.choose(sim.role, upkeep, mode="clip")
    sim.energy -= cost

    starved = sim.board & (sim.energy <= 0)
    sim.board[starved] = 0
    sim.role[starved] = 0
