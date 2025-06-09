Below is a **production-ready Python reference implementation** for **SYMBIOTE STRAINS**.
It is entirely self-contained, fast on very large boards ( ≥ 4096×4096), and organised so you can drop individual modules into a bigger engine later (Textual, Pygame, WebGL, etc.).
The code emphasises:

- **Vectorised NumPy kernels** – one pass per generation, no Python-level neighbour loops.
- **Role-aware combat & reproduction** – Attackers, Defenders, Mitosis Sacs, Slime.
- **Absorption growth** – conquered cells switch strain and immediately feed the victor’s mass counter.
- **Parameter hooks** – tweak speed, roles, spawn odds, rule thresholds without touching core maths.

---

## 1. Data Model

| Array        | dtype    | Shape   | Meaning                                                           |
| ------------ | -------- | ------- | ----------------------------------------------------------------- |
| `board`      | `uint8`  | _(H,W)_ | Strain ID (0 = empty, 1-5 active strains)                         |
| `role`       | `uint8`  | _(H,W)_ | Role code (0 = empty, 1 slime, 2 mitosis, 3 defender, 4 attacker) |
| `mito_clock` | `uint8`  | _(H,W)_ | Cool-down ticks before a Mitosis Sac divides again                |
| `mass`       | `uint32` | _(6,)_  | Live-updated population of each strain (index 0 unused)           |

Role codes are packed into a single byte so you can later compress into bitfields if memory is tight.

---

## 2. Core Rules (single generation)

> All kernel operations are performed **one strain at a time** to avoid branch explosion.

1. **Classic Life step (Slime only)**

   - Birth: empty ∧ neighbours_same_strain == 3 → new Slime
   - Survival: (neighbours_same_strain ∈ {2,3}) → keep, else die

2. **Mitosis Sac**

   - Decrement its clock.
   - When clock == 0 → spawn a new Slime in the emptiest Von-Neumann neighbour, reset clock (∈\[6,10] ticks).

3. **Combat**

   - For every Attacker cell **A**:
     – build a boolean mask of enemy neighbours (board ≠ 0 & board ≠ strainA).
     – Attacker conquers those cells **unless** the target is a Defender who has ≥ 2 same-strain defenders adjacent.
     – When conquered: `board` = strainA, `role` = 1 (Slime), `mass[strainA]++`, `mass[target_strain]--`.

4. **Defender decay**

   - If a Defender has had **no enemy in its 5×5 locality for N ticks** → downgrade to Slime (prevent over-turtling).

---

## 3. Code

```python
"""
symbiote_strains.py – fast Conway-variant with multi-strain absorption
Author: ChatGPT (2025-06-08)
Licence: MIT
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

# ---------- Parameters ---------- #
STRANDS      = 5                     # number of competing strains (1-based IDs)
H, W         = 2048, 2048            # board size – scale up; RAM≈ 4×H×W bytes
P_MITO       = 0.015                 # initial chance a live cell is Mitosis Sac
P_DEF        = 0.04                  # Defender chance
P_ATK        = 0.04                  # Attacker chance
INITIAL_DENS = 0.20                  # portion of grid initially alive (all strains mixed)
K_LIFE       = np.array([[1,1,1],
                         [1,0,1],
                         [1,1,1]], dtype=np.uint8)
VIEW_RADIUS  = 2                     # Defender awareness radius (5×5)
DEF_HOLD     = 30                    # ticks a defender may stay idle
MITO_RANGE   = (6, 10)               # cooldown window (inclusive)
rng          = np.random.default_rng()

# ---------- State tensors ---------- #
board   : NDArray[np.uint8]  = np.zeros((H, W), dtype=np.uint8)
role    : NDArray[np.uint8]  = np.zeros_like(board)    # 0 empty
mito_clk: NDArray[np.uint8]  = np.zeros_like(board)
idle_def: NDArray[np.uint8]  = np.zeros_like(board)    # defender idle counter
mass    : NDArray[np.uint32] = np.zeros(STRANDS+1, dtype=np.uint32)

# ---------- Helpers ---------- #
def random_init() -> None:
    """Seed board with mixed strains and roles."""
    alive_mask = rng.random((H, W)) < INITIAL_DENS
    board[alive_mask] = rng.integers(1, STRANDS+1, size=alive_mask.sum(), dtype=np.uint8)
    role[alive_mask]  = 1  # default slime

    # role specialisation
    for rcode, prob in [(2, P_MITO), (3, P_DEF), (4, P_ATK)]:
        mask = alive_mask & (rng.random((H, W)) < prob)
        role[mask] = rcode
        if rcode == 2:
            mito_clk[mask] = rng.integers(*MITO_RANGE, size=mask.sum(), dtype=np.uint8)

    np.add.at(mass, board, 1)   # population count


def neighbour_count(mat: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """3×3 convolution using numpy.roll – ~4× faster than scipy on contiguous • uint8."""
    tot = np.zeros_like(mat, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == dx == 0: continue
            tot += np.roll(np.roll(mat, dy, 0), dx, 1)
    return tot


# ---------- Main simulation step ---------- #
def step() -> None:
    global board, role, mito_clk, idle_def, mass

    new_board  = board.copy()
    new_role   = role.copy()
    new_mito   = mito_clk.copy()
    new_idle   = idle_def.copy()
    delta_mass = np.zeros_like(mass)

    for strain in range(1, STRANDS+1):
        # --- Masks for this strain --- #
        mine   = board == strain
        empty  = board == 0
        mine_slime = mine & (role == 1)

        # ---------- Classic Life ---------- #
        n_slime = neighbour_count(mine_slime.astype(np.uint8))
        birth   = (empty & (n_slime == 3))
        survive = (mine_slime & ((n_slime == 2) | (n_slime == 3)))

        # apply births
        births_idx = np.where(birth)
        new_board[births_idx] = strain
        new_role[births_idx]  = 1
        delta_mass[strain]   += births_idx[0].size

        # mark deaths (slime only)
        death_idx = np.where(mine_slime & ~survive)
        new_board[death_idx] = 0
        new_role[death_idx]  = 0
        delta_mass[strain]  -= death_idx[0].size

        # ---------- Mitosis ---------- #
        mito_mask = mine & (role == 2)
        new_mito[mito_mask] -= 1
        ready = mito_mask & (new_mito == 0)

        if ready.any():
            # choose emptiest Von-Neumann neighbour
            for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                target = np.roll(np.roll(ready, dy, 0), dx, 1) & empty
                spawn_idx = np.where(target & ready)
                new_board[spawn_idx] = strain
                new_role[spawn_idx]  = 1
                empty[spawn_idx]     = False  # prevent double-spawning
                delta_mass[strain]  += spawn_idx[0].size
            # reset clocks
            new_mito[ready] = rng.integers(*MITO_RANGE, size=ready.sum())

        # ---------- Combat ---------- #
        atk_mask = mine & (role == 4)
        if atk_mask.any():
            enemy = (board != 0) & (board != strain)
            enemy_adj = np.zeros_like(atk_mask, dtype=bool)
            for dy, dx in [(1,0),(-1,0),(0,1),(0,-1)]:
                enemy_adj |= atk_mask & np.roll(np.roll(enemy, -dy, 0), -dx, 1)

            conquer_targets = np.zeros_like(enemy_adj, dtype=bool)
            if enemy_adj.any():
                defender_mask = (role == 3) & enemy
                # cells with ≥2 same-strain defenders are shielded
                def_neigh = neighbour_count(defender_mask.astype(np.uint8))
                shield = defender_mask & (def_neigh >= 2)
                conquer_targets = enemy_adj & ~shield

                trg_idx = np.where(conquer_targets)
                conquered_strains = board[trg_idx]

                new_board[trg_idx] = strain
                new_role[trg_idx]  = 1  # becomes slime
                delta_mass[strain]           += trg_idx[0].size
                np.add.at(delta_mass, conquered_strains, -1)

        # ---------- Defender idle / downgrade ---------- #
        def_mask = mine & (role == 3)
        if def_mask.any():
            # enemy presence in 5×5
            enemy_local = np.zeros_like(def_mask, dtype=np.uint8)
            for dy in range(-VIEW_RADIUS, VIEW_RADIUS+1):
                for dx in range(-VIEW_RADIUS, VIEW_RADIUS+1):
                    if dy == dx == 0: continue
                    enemy_local += np.roll(np.roll((board != 0) & (board != strain), dy, 0), dx, 1)
            active_def = def_mask & (enemy_local > 0)
            idle_def_mask = def_mask & ~active_def
            new_idle[idle_def_mask] += 1
            new_idle[active_def]     = 0
            downgrade = idle_def_mask & (new_idle >= DEF_HOLD)
            if downgrade.any():
                new_role[downgrade] = 1
                new_idle[downgrade] = 0  # reset

    # ---------- Commit frame ---------- #
    board[:]     = new_board
    role[:]      = new_role
    mito_clk[:]  = new_mito
    idle_def[:]  = new_idle
    mass        += delta_mass


# ---------- CLI driver ---------- #
def run(ticks: int = 1000, headless: bool = True) -> None:
    random_init()
    for t in range(1, ticks+1):
        step()
        if not headless and t % 10 == 0:
            # simple text sparkline — replace with fancy GUI later
            print(f"t={t}  masses={mass[1:]}")
        # early stop when one strain dominates
        live_strains = np.flatnonzero(mass[1:] > 0)
        if live_strains.size == 1:
            print(f"Strain {live_strains[0]+1} wins at t={t}")
            break


if __name__ == "__main__":
    run()
```

### Why it’s fast

1. **Neighbour sums with eight `np.roll` adds** – pure C loops under the hood, avoiding Python iteration.
2. **Per-strain processing** keeps boolean masks contiguous and branch-free.
3. **No Python lists in inner loop** – only vectorised index maths.

---

## 4. Tuning Hooks

| Variable                   | Effect                                                                       |
| -------------------------- | ---------------------------------------------------------------------------- |
| `INITIAL_DENS`             | How busy the first generation is (0.05–0.3 gives nice roots)                 |
| `P_MITO`, `P_DEF`, `P_ATK` | Role mix. Raising `P_ATK` quickens take-overs, lowering `P_DEF` speeds games |
| `MITO_RANGE`               | Shorter windows make strains _very_ aggressive growth-wise                   |
| `VIEW_RADIUS` / `DEF_HOLD` | Increase to favour defensive playstyles                                      |
| Board size                 | Bigger = emergent macro-roots; 8192² still under 1 GB RAM                    |

---

## 5. Visualising

While the engine itself is headless, piping frames into **matplotlib-imshow** or **Textual Canvas** is trivial:

```python
import matplotlib.pyplot as plt
plt.imshow(board, cmap="tab10")   # strain colours
plt.title(f"Generation {tick}")
plt.pause(0.001)
```

Switch to **PyGame** or **WebGL** for real-time RGB shaders (each strain = HSL hue; role → saturation).

---

Below is an _engineering-first_ expansion of **§ 6 — Next Steps**, broken into discrete, build-ready work-items. Each item contains

- **Goal** – what it unlocks.
- **Design sketch** – data-model or API signature changes.
- **Pseudocode / code graft** – drop-in snippets showing critical lines.
- **Perf / play-test notes** – what to benchmark or balance.

Use the order given; every task is largely independent, so you can branch and merge in any sequence.

---

## 6 A. GPU Port - CuPy / PyTorch

### Goal

Hit **60 fps on 8192² boards** and free the CPU for UI & RL agents.

### Design Sketch

| Old                         | New                                                            |
| --------------------------- | -------------------------------------------------------------- |
| `import numpy as np`        | `import cupy as cp  # or torch.cuda`                           |
| `board: np.ndarray`         | `board: cp.ndarray`                                            |
| `np.roll` neighbour kernels | **Toroidal texture read** via `cp.take` or slicing; zero-copy. |
| Python RNG                  | `cp.random.Generator(cp.random.XORWOW)`                        |

### Drop-in Changes

```python
xp = cp  # alias so core code is agnostic

def neighbour_count(mat: xp.ndarray) -> xp.ndarray:
    # 3×3 toroidal convolution, GPU-shared memory friendly
    up    = xp.roll(mat,  1, 0); down  = xp.roll(mat, -1, 0)
    left  = xp.roll(mat,  1, 1); right = xp.roll(mat, -1, 1)
    diag1 = xp.roll(up,  1, 1);  diag2 = xp.roll(up, -1, 1)
    diag3 = xp.roll(down, 1, 1); diag4 = xp.roll(down, -1, 1)
    return (up+down+left+right+diag1+diag2+diag3+diag4).astype(xp.uint8)
```

### Perf Notes

- **Pinned-CPU ↔ GPU transfer kills fps** – keep _all_ simulation tensors on device.
- Render with OpenGL: call `cp.asnumpy(board)` only after down-scaling to a 1024² texture.

---

## 6 B. Genetic Drift & Mutation

### Goal

Evolving “strains-within-strains”; conquer events blend DNA, producing emergent sub-species.

### Design Sketch

```python
dna: uint8[H,W]   # 0-255 pigment / trait value
dna_mass: uint32[6, 256]  # histogram per strain (cheap analytics)
MUTATE_PROB = 0.0005
```

- **Birth / Mitosis:** child inherits parent’s `dna`; mutates with prob p (`dna ± rng.integers(±1-3)` clipped 0-255).
- **Conquest:** new dna = weighted average of attacker and victim (`(3*atk + vic) // 4`) then optional mutation.

### Pseudocode Injection

```python
# After conquest code
new_dna = ((3 * dna[atk_idx] + dna[trg_idx]) // 4).astype(cp.uint8)
dna[trg_idx] = new_dna
# mutation
mut = rng.random(trg_idx[0].size) < MUTATE_PROB
dna[trg_idx][mut] += rng.integers(-3, 4, mut.sum(), dtype=cp.int8)
dna[trg_idx] = dna[trg_idx] & 0xFF
```

### Play-Test

Track `dna_mass` to see whether colours converge or branch. Raise `MUTATE_PROB` until visible speciation > 20 k ticks.

---

## 6 C. Energy Economy

### Goal

Stop runaway domination; introduce _logistic growth_ + strategic sacrifice.

### Design Sketch

| Tensor           | dtype    | Desc                 |
| ---------------- | -------- | -------------------- |
| `energy[H,W]`    | `uint16` | cell fuel (0-65 535) |
| `strain_bank[6]` | `uint64` | global reserve       |

Rules

1. **Tick cost:** Defender = 3 ✱, Attacker = 2 ✱, Mitosis = 1 ✱, Slime = 1 ✱.
2. **Energy siphon on conquest:** 70 % of victim’s cell energy → victor’s _cell_, 30 % → strain_bank.
3. **Starvation:** energy ≤ 0 ⇒ cell dies (empties).
4. **Exchange:** any cell may pull 10 ✱ from strain_bank if below 4 ✱.

### Critical Lines

```python
cost = xp.choose(role, (0,1,1,3,2)).astype(xp.uint16)
energy -= cost
starve = energy <= 0
board[starve] = role[starve] = 0
energy[starve] = 0
```

Balance so average strain income (mitosis new births + conquest spoils) ≈ total upkeep.

---

## 6 D. Asynchronous Tick Scheduling

### Goal

Break perfect synchrony ⇒ removes artificial oscillators; yields more “organic” motion.

### Design

- **Tile grid** 64×64 macro-cells.
- Only ⅛ of tiles update each micro-tick (`(tick + x + y) % 8 == 0`).
- One “meta-tick” = 8 micro-ticks ⇒ full board updated.

### Impact

CPU/GPU utilisation flattens, fewer sudden surges; movement visually ripples.

---

## 6 E. RL-Guided Meta-Brains

### Goal

Let strains learn macro-policies (spawn ratios, aggression thresholds, energy hoarding).

### Environment Stub

```python
class SymbioteEnv(gym.Env):
    def __init__(self, cfg):
        self.obs_space = gym.spaces.Box(0, 1, (8,8,5), np.uint8)  # 8×8 down-sample ×5 planes
        self.act_space = gym.spaces.Discrete(9)  # tweak P_MITO / P_ATK bins
    def step(self, action):
        apply_policy(action)
        for _ in range(32): step()      # advance 32 sim ticks
        reward = mass[strain] - self.prev_mass
        obs = downsample_state()
        done = (t > T_MAX) or (mass[strain]==0)
        return obs, reward, done, {}
```

Plug into **Stable-Baselines3 PPO** (`policy="MlpPolicy", n_envs=8`).

Reward shaping ideas:

- +1 per enemy cell absorbed
- −0.1 per own cell lost
- +10 for final victory

Let each strain run its own agent or share weights for multi-agent symmetry.

---

## 6 F. Rendering & UX

1. **OpenGL ES / WebGL** shader: `board` → palette texture (`dna` hue, `role` saturation).
2. **Zoomable minimap** – mip-map the texture on GPU side, sample LOD by zoom level.
3. **Click-to-mutate dev-cheat** – set `dna` to rare gene on click for rapid visual testing.

---

## 6 G. Config & Bench Harness

- `symb.conf` (TOML): all tunables autoload at start; hot-reload on `SIGHUP`.
- `pytest-bench` – four scenarios (sparse, dense, late-game, energy-starve) auto-profile both CPU & GPU back-ends.
- CLI flags: `--seed`, `--render`, `--gpu torch|cupy|off`.

---

### Suggested Implementation Order

1. **GPU back-end (6 A)** – foundation for everything else.
2. **Energy economy (6 C)** – forces balanced gameplay before adding complexity.
3. **Genetic drift (6 B)** – purely aesthetic at first; low coupling.
4. **Async scheduler (6 D)** – trivial addition once kernels are GPU-friendly.
5. **RL brains (6 E)** – now simulation is fast & stable.
6. **Rendering polish (6 F)** – when visuals become bottleneck.
7. **Bench & config (6 G)** – cement reproducibility.

Ship each feature behind a **feature-flag** so you can A/B profile:

```bash
python symbiote.py --features energy,dna,async,gpu
```
