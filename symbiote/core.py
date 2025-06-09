"""Core CPU simulation – vectorised NumPy version."""

from __future__ import annotations

import numpy as np

from . import config

K_LIFE = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)


class Simulation:
    """
    Manages the state and progression of the simulation.

    This class encapsulates the simulation board, cell roles, and all the logic
    for updating the game state at each tick.
    """

    def __init__(self, seed: int | None = None):
        cfg = config.load_config()
        sim_cfg = cfg.get("simulation", {})
        role_cfg = cfg.get("roles", {})

        self.rng = np.random.default_rng(seed)
        self.height = sim_cfg.get("h", 1024)
        self.width = sim_cfg.get("w", 1024)
        self.strains = sim_cfg.get("strands", 5)

        self.board = np.zeros((self.height, self.width), np.uint8)
        self.role = np.zeros_like(self.board)
        self.mito_clk = np.zeros_like(self.board)
        self.idle_def = np.zeros_like(self.board)
        self.mass = np.zeros(self.strains + 1, np.uint32)
        self.dna = None
        self.energy = None
        self._init_random(
            p_alive=sim_cfg.get("initial_density", 0.15), role_cfg=role_cfg
        )

    # --- initialisation ------------------------------------------------
    def _init_random(self, p_alive=0.15, role_cfg=None):
        """Initialise the board with random cell placements."""
        if role_cfg is None:
            role_cfg = {}
        alive = self.rng.random((self.height, self.width)) < p_alive
        self.board[alive] = self.rng.integers(
            1, self.strains + 1, alive.sum(), dtype=np.uint8
        )
        self.role[alive] = 1

        # role specialisation
        p_mito = role_cfg.get("p_mito", 0.015)
        p_def = role_cfg.get("p_def", 0.04)
        p_atk = role_cfg.get("p_atk", 0.04)
        mito_range = role_cfg.get("mito_range", [6, 10])

        for rcode, prob in [(2, p_mito), (3, p_def), (4, p_atk)]:
            mask = alive & (self.rng.random((self.height, self.width)) < prob)
            self.role[mask] = rcode
            if rcode == 2:
                self.mito_clk[mask] = self.rng.integers(
                    *mito_range, size=mask.sum(), dtype=np.uint8
                )

        np.add.at(self.mass, self.board, 1)

    # --- utilities -----------------------------------------------------
    def _neigh(self, mat: np.ndarray) -> np.ndarray:
        """Count neighbours for each cell in a matrix."""
        tot = np.zeros_like(mat, np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == dx == 0:
                    continue
                tot += np.roll(np.roll(mat, dy, 0), dx, 1)
        return tot

    # --- external API ---------------------------------------------------
    def step(self):
        """
        Advance the simulation by one time step.

        This method applies all the rules of the simulation, including life,
        death, mitosis, and combat.
        """
        new_board = self.board.copy()
        new_role = self.role.copy()
        new_mito = self.mito_clk.copy()
        new_idle = self.idle_def.copy()
        delta_mass = np.zeros_like(self.mass)

        cfg = config.load_config()
        role_cfg = cfg.get("roles", {})
        mito_range = role_cfg.get("mito_range", [6, 10])
        view_radius = role_cfg.get("view_radius", 2)
        def_hold = role_cfg.get("def_hold", 30)

        for s in range(1, self.strains + 1):
            mine = self.board == s
            slime = mine & (self.role == 1)
            empty = self.board == 0

            # --- Classic life ---
            nbs = self._neigh(slime.astype(np.uint8))
            birth = empty & (nbs == 3)
            survive = slime & ((nbs == 2) | (nbs == 3))

            birth_idx = np.where(birth)
            new_board[birth_idx] = s
            new_role[birth_idx] = 1
            delta_mass[s] += birth_idx[0].size

            death_idx = np.where(slime & ~survive)
            new_board[death_idx] = 0
            new_role[death_idx] = 0
            delta_mass[s] -= death_idx[0].size

            # --- Mitosis ---
            mito_mask = mine & (self.role == 2)
            new_mito[mito_mask] -= 1
            ready = mito_mask & (new_mito == 0)

            if ready.any():
                # choose emptiest Von-Neumann neighbour
                for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    target = np.roll(np.roll(ready, dy, 0), dx, 1) & empty
                    spawn_idx = np.where(target & ready)
                    new_board[spawn_idx] = s
                    new_role[spawn_idx] = 1
                    empty[spawn_idx] = False  # prevent double-spawning
                    delta_mass[s] += spawn_idx[0].size
                # reset clocks
                new_mito[ready] = self.rng.integers(*mito_range, size=ready.sum())

            # --- Combat ---
            atk_mask = mine & (self.role == 4)
            if atk_mask.any():
                enemy = (self.board != 0) & (self.board != s)
                enemy_adj = np.zeros_like(atk_mask, dtype=bool)
                for dy, dx in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    enemy_adj |= atk_mask & np.roll(np.roll(enemy, -dy, 0), -dx, 1)

                conquer_targets = np.zeros_like(enemy_adj, dtype=bool)
                if enemy_adj.any():
                    defender_mask = (self.role == 3) & enemy
                    # cells with ≥2 same-strain defenders are shielded
                    def_neigh = self._neigh(defender_mask.astype(np.uint8))
                    shield = defender_mask & (def_neigh >= 2)
                    conquer_targets = enemy_adj & ~shield

                    trg_idx = np.where(conquer_targets)
                    conquered_strains = self.board[trg_idx]

                    new_board[trg_idx] = s
                    new_role[trg_idx] = 1  # becomes slime
                    delta_mass[s] += trg_idx[0].size
                    np.add.at(delta_mass, conquered_strains, -1)

            # --- Defender idle / downgrade ---
            def_mask = mine & (self.role == 3)
            if def_mask.any():
                # enemy presence in 5×5
                enemy_local = np.zeros_like(def_mask, dtype=np.uint8)
                for dy in range(-view_radius, view_radius + 1):
                    for dx in range(-view_radius, view_radius + 1):
                        if dy == dx == 0:
                            continue
                        enemy_local += np.roll(
                            np.roll((self.board != 0) & (self.board != s), dy, 0), dx, 1
                        )
                active_def = def_mask & (enemy_local > 0)
                idle_def_mask = def_mask & ~active_def
                new_idle[idle_def_mask] += 1
                new_idle[active_def] = 0
                downgrade = idle_def_mask & (new_idle >= def_hold)
                if downgrade.any():
                    new_role[downgrade] = 1
                    new_idle[downgrade] = 0  # reset

        self.board[:] = new_board
        self.role[:] = new_role
        self.mito_clk[:] = new_mito
        self.idle_def[:] = new_idle
        self.mass += delta_mass

    # iterable convenience
    def __iter__(self):
        return self

    def __next__(self):
        self.step()
        return self.board


# shorthand
def step(sim: "Simulation"):
    """Functional-style step function."""
    sim.step()
