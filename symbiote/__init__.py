"""Top‑level convenience API."""
from .core import Simulation, step
from .config import load_config
__all__ = ['Simulation', 'step', 'load_config']
