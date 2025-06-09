"""CLI façade wrapping core + optional plugins."""

import argparse
import importlib

import numpy as np

from . import dna, energy


def main():
    """
    CLI façade wrapping core + optional plugins.
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ticks", type=int, default=512)
    p.add_argument("--gpu", choices=["off", "cupy", "torch"], default="off")
    p.add_argument("--features", default="", help="comma list: energy,dna")
    p.add_argument("--render", action="store_true", help="Enable matplotlib viewer")
    args = p.parse_args()

    sim_mod = "symbiote.core"
    if args.gpu != "off":
        try:
            if args.gpu == "cupy":
                import cupy  # check for install
            sim_mod = "symbiote.gpu_backend"
        except ImportError:
            print(
                f"ERROR: --gpu={args.gpu} requires the '{args.gpu}' package to be installed."
            )
            return

    sim = importlib.import_module(sim_mod).Simulation()

    viewer = None
    if args.render:
        # from .viz.matplotlib_viewer import MatplotlibViewer
        # viewer = MatplotlibViewer(sim)
        print("--render flag is currently disabled. Run scripts/run_pygame.py instead.")

    feats = args.features.split(",")
    if "dna" in feats:
        dna.init_dna(sim)
    for t in range(1, args.ticks + 1):
        sim.step()
        if "energy" in feats:
            energy.apply_energy(sim)
        # rudimentary text output every 100 ticks
        if t % 100 == 0:
            print(f"t={t}  masses={sim.mass[1:]}")

        # early stop when one strain dominates
        live_strains = np.flatnonzero(sim.mass[1:] > 0)
        if live_strains.size <= 1:
            if live_strains.size == 1:
                print(f"Strain {live_strains[0]+1} wins at t={t}")
            else:
                print(f"All strains died out at t={t}")
            break


if __name__ == "__main__":
    main()
