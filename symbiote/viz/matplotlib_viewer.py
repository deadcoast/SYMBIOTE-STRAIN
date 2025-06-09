"""A matplotlib-based viewer for the simulation."""

import matplotlib.pyplot as plt

from ..core import Simulation


class MatplotlibViewer:
    """A viewer for the simulation using matplotlib."""

    def __init__(self, sim: Simulation):
        """Initialise the viewer."""
        self.sim = sim
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.im = self.ax.imshow(sim.board)

    def update(self):
        """Update the viewer with the current simulation state."""
        self.im.set_data(self.sim.board)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
