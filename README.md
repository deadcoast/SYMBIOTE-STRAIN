# SYMBIOTE STRAINS

Evolutionary Conway‑variant where five competing cell cultures fight, absorb, and mutate.

- **CPU & GPU kernels** (NumPy / CuPy)
- Energy economy, genetic drift, async scheduler
- Optional RL meta‑brains via Gym/PPO
- Pluggable renderers (matplotlib, OpenGL/WebGL)

```bash
# quick‑start (CPU)
pip install -r requirements.txt
python scripts/run_game.py --ticks 2000

# GPU (CuPy CUDA)
python scripts/run_game.py --gpu cupy --features energy,dna
```

**Note on GPU support**: The `cupy` backend requires an NVIDIA GPU and the CUDA toolkit. As `cupy` is not available on all platforms (e.g., macOS), it is not included in `requirements.txt`. To use the GPU-accelerated version, please install `cupy` manually by following the instructions on their official website.
