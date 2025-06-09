"""DNA mutation feature."""

import numpy as np

from .core import Simulation


def init_dna(sim: Simulation):
    """Initialise DNA attributes for the simulation."""
    sim.dna = np.random.randint(0, 256, size=sim.board.shape, dtype=np.uint8)


def apply_dna_drift(sim: Simulation):
    """
    Apply genetic drift to the DNA.

    This function is a placeholder and is not yet implemented.
    """
    pass


def mutate_region(sim: Simulation, mask: np.ndarray, p: float = 0.001):
    """
    Mutate a region of the DNA.

    Args:
        sim: The simulation object.
        mask: A boolean array indicating the region to mutate.
        p: The probability of mutation for each cell.
    """
    if sim.dna is None:
        return
    sub = sim.dna[mask]
    jitter = np.random.randint(-3, 4, sub.size, dtype=np.int8)
    mutate_mask = np.random.random(sub.size) < p
    sub[mutate_mask] = (sub[mutate_mask].astype(np.int16) + jitter[mutate_mask]) & 0xFF
    sim.dna[mask] = sub
