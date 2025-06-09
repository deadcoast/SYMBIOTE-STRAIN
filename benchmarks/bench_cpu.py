"""Benchmark for the core CPU simulation."""

import time

from symbiote.core import Simulation


def run_benchmark():
    """Runs the benchmark and prints the elapsed time."""
    sim = Simulation()
    t0 = time.perf_counter()
    for _ in range(128):
        sim.step()
    elapsed = time.perf_counter() - t0
    print(f"Elapsed: {elapsed:.4f}s")


if __name__ == "__main__":
    run_benchmark()
