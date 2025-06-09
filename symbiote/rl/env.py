"""A gymnasium environment for the simulation."""

import gymnasium as gym
import numpy as np

from symbiote.core import Simulation


class SymbioteEnv(gym.Env):
    """A gymnasium environment for the symbiote simulation."""

    def __init__(self):
        """Initialise the environment."""
        super().__init__()
        self.sim = Simulation()
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.sim.board.shape, dtype=np.uint8
        )

    def reset(self, *, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        self.sim = Simulation(seed=seed)
        return self.sim.board, {}

    def step(self, action):
        """Execute one time step within the environment."""
        _ = action  # unused
        self.sim.step()
        obs = self.sim.board
        terminated = not np.any(self.sim.board)
        truncated = False
        reward = 0.0 if terminated else 1.0
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        """Render the environment."""
        _ = mode  # unused
