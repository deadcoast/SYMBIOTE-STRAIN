#!/usr/bin/env python
import sys
import pathlib

# Add project root to the path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.absolute()))

from symbiote.core import Simulation
from symbiote.viz.pygame_viewer import Viewer
from symbiote.dna import init_dna       # optional

sim = Simulation(h=512, w=512, strains=5)
init_dna(sim)                           # comment-out if not using DNA hues
Viewer(sim, scale=2).run()
