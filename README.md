# SYMBIOTE STRAINS

Evolutionary Conway‑variant where five competing cell cultures fight, absorb, and mutate.

* **CPU & GPU kernels** (NumPy / CuPy)
* Energy economy, genetic drift, async scheduler
* Optional RL meta‑brains via Gym/PPO
* Pluggable renderers (matplotlib, OpenGL/WebGL)

```bash
# quick‑start (CPU)
pip install -r requirements.txt
python scripts/run_game.py --ticks 2000

# GPU (CuPy CUDA)
python scripts/run_game.py --gpu cupy --features energy,dna
```
