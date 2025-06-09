#!/usr/bin/env python3
import argparse
from symbiote.core import Simulation
from symbiote.dna import init_dna
from symbiote.viz.pygame_viewer import Viewer

def main():
    parser = argparse.ArgumentParser(description="Run the Symbiote Strains simulation.")
    parser.add_argument('--width', type=int, default=256, help='Width of the simulation grid.')
    parser.add_argument('--height', type=int, default=256, help='Height of the simulation grid.')
    parser.add_argument('--strains', type=int, default=8, help='Number of strains.')
    parser.add_argument('--scale', type=int, default=4, help='Screen scaling factor.')
    args = parser.parse_args()

    print("Initializing simulation...")
    sim = Simulation(h=args.height, w=args.width, strains=args.strains)
    init_dna(sim)

    print("Starting viewer...")
    viewer = Viewer(sim, scale=args.scale)
    viewer.run()
    print("Viewer closed.")

if __name__ == '__main__':
    main() 