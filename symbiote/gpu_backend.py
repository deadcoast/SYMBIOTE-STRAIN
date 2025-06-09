"""CuPy-based Simulation drop-in."""

import cupy as cp

from symbiote.core import Simulation as CPUSimulation


class Simulation(CPUSimulation):
    """
    A CuPy-based simulation that inherits from the CPU version.

    This class moves the simulation tensors to the GPU for accelerated computation.
    """

    def __init__(self, *args, **kwargs):
        """Initialise the simulation and move tensors to the GPU."""
        super().__init__(*args, **kwargs)
        self.board = cp.asarray(self.board)
        self.role = cp.asarray(self.role)
        self.mito_clk = cp.asarray(self.mito_clk)
        self.idle_def = cp.asarray(self.idle_def)
        if self.dna is not None:
            self.dna = cp.asarray(self.dna)
        if self.energy is not None:
            self.energy = cp.asarray(self.energy)

    def _neigh(self, mat):
        """Count neighbours for each cell in a matrix using CuPy."""
        tot = cp.zeros_like(mat, cp.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                tot += cp.roll(cp.roll(mat, dy, 0), dx, 1)
        return tot

    def step(self):
        """Placeholder for a simulation step. Currently does nothing."""
        pass

    def __iter__(self):
        return self

    def __next__(self):
        self.step()
        return self.board
