"""A gym environment for the simulation."""

import gym
import numpy as np

from symbiote.core import Simulation


class SymbioteEnv(gym.Env):
    """A gym environment for the symbiote simulation."""

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
        super().reset(seed=seed, options=options)
        self.sim = Simulation(seed=seed)
        return self.sim.board, {}

    def step(self, action):
        """Execute one time step within the environment."""
        _ = action  # unused
        self.sim.step()
        obs = self.sim.board
        reward = 1.0
        done = not np.any(self.sim.board)
        info = {}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        """Render the environment."""
        _ = mode  # unused
        # not implemented
        pass
