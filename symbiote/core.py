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

    def __init__(self, h: int | None = None, w: int | None = None, strains: int | None = None, seed: int | None = None):
        cfg = config.load_config()
        sim_cfg = cfg.get("simulation", {})
        role_cfg = cfg.get("roles", {})

        self.rng = np.random.default_rng(seed)
        self.height = h if h is not None else sim_cfg.get("h", 1024)
        self.width = w if w is not None else sim_cfg.get("w", 1024)
        self.strains = strains if strains is not None else sim_cfg.get("strands", 5)

        self.board = np.zeros((self.height, self.width), np.uint8)
        self.role = np.zeros_like(self.board)
        self.mito_clk = np.zeros_like(self.board)
        self.idle_def = np.zeros_like(self.board)
        self.mass = np.zeros(self.strains + 1, np.uint32)
        self.last_conquer_mask = np.zeros_like(self.board, dtype=bool)
        self.dna = np.zeros_like(self.board, dtype=np.uint16)
        self.idle_counter = np.zeros_like(self.board, dtype=np.uint16)
        self.energy = None
        self._init_clustered()

    # --- initialisation ------------------------------------------------
    def _init_clustered(self):
        """Seed the board with dense, circular clusters for each strain."""
        num_strains = self.strains
        radius = min(self.height, self.width) / (num_strains * 1.5) # Cluster radius
        
        # Create coordinates grid
        Y, X = np.ogrid[:self.height, :self.width]

        for i in range(1, num_strains + 1):
            # Arrange spawn points in a circle
            angle = 2 * np.pi * (i-1) / num_strains
            center_h = self.height / 2 + (self.height / 3) * np.sin(angle)
            center_w = self.width / 2 + (self.width / 3) * np.cos(angle)

            # Create a circular mask for the cluster
            dist_from_center = np.sqrt((X - center_w)**2 + (Y - center_h)**2)
            
            # Use a probability falloff from the center to make clusters look more organic
            prob_mask = np.clip(1.0 - (dist_from_center / radius), 0, 1)
            alive_mask = self.rng.random(self.board.shape) < (prob_mask * 0.9)

            # Assign strain and base role
            self.board[alive_mask] = i
            self.role[alive_mask] = 1

            # Role specialisation
            cluster_cells = np.where(alive_mask)
            num_cluster_cells = cluster_cells[0].size
            p_mito = 0.03
            p_def = 0.10
            p_atk = 0.08

            # Assign Mitosis Sacs
            mito_indices = self.rng.choice(num_cluster_cells, int(num_cluster_cells * p_mito), replace=False)
            self.role[cluster_cells[0][mito_indices], cluster_cells[1][mito_indices]] = 2
            
            # Assign Defenders
            def_indices = self.rng.choice(np.setdiff1d(np.arange(num_cluster_cells), mito_indices), int(num_cluster_cells * p_def), replace=False)
            self.role[cluster_cells[0][def_indices], cluster_cells[1][def_indices]] = 3

            # Assign Attackers
            atk_indices = np.setdiff1d(np.arange(num_cluster_cells), np.concatenate([mito_indices, def_indices]))
            atk_indices = self.rng.choice(atk_indices, int(num_cluster_cells * p_atk), replace=False)
            self.role[cluster_cells[0][atk_indices], cluster_cells[1][atk_indices]] = 4

            # Initialize DNA with strain-specific characteristics
            # Genes: [Mobility, Growth, Aggression, Defense]
            base_dna = np.array([
                self.rng.integers(5, 15), # Mobility
                self.rng.integers(5, 15), # Growth
                self.rng.integers(5, 15), # Aggression
                self.rng.integers(5, 15)  # Defense
            ], dtype=np.uint16)
            self.dna[alive_mask] = (base_dna[0] << 12) | (base_dna[1] << 8) | (base_dna[2] << 4) | base_dna[3]

        # Initialize mass count
        np.add.at(self.mass, self.board, 1)

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
        delta_mass = np.zeros_like(self.mass, dtype=np.int32)

        cfg = config.load_config()
        role_cfg = cfg.get("roles", {})
        mito_range = role_cfg.get("mito_range", [6, 10])
        view_radius = role_cfg.get("view_radius", 2)
        def_hold = role_cfg.get("def_hold", 30)

        # --- Pre-computation of Role Auras ---
        defender_aura = self._neigh(self.role == 3) > 0
        attacker_aura = self._neigh(self.role == 4) > 0
        mitosis_aura = self._neigh(self.role == 2) > 0

        for s in range(1, self.strains + 1):
            mine = self.board == s
            slime = mine & (self.role == 1)
            empty = self.board == 0
            
            conquer_targets = np.zeros_like(self.board, dtype=bool)

            # --- Classic Life with Aura Effects ---
            n_slime = self._neigh(mine & (self.role == 1))
            
            # Mitosis aura makes birth easier
            birth_threshold = np.full_like(self.board, 3, dtype=np.uint8)
            birth_threshold[mine & mitosis_aura] = 2 # Need only 2 neighbors to make a baby in the nursery
            birth = empty & (n_slime >= birth_threshold)
            
            # Fortified cells are harder to kill
            survive_threshold_2 = (n_slime == 2)
            survive_threshold_3 = (n_slime == 3)
            survive = (mine & (self.role == 1)) & (survive_threshold_2 | survive_threshold_3)

            birth_idx = np.where(birth)
            new_board[birth_idx] = s
            new_role[birth_idx] = 1
            delta_mass[s] += birth_idx[0].size

            # DNA inheritance for new slimes
            if self.dna is not None:
                parent_mask = (n_slime == 3) & birth
                parent_dna_sum = np.zeros_like(self.dna, dtype=np.uint32)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == dx == 0: continue
                        parent_dna_sum += np.roll(np.roll(self.dna * (self.board == s), dy, 0), dx, 1)
                
                # Average the DNA of the 3 parents, avoiding division by zero
                avg_dna = np.divide(parent_dna_sum, n_slime, where=n_slime>0, out=np.zeros_like(parent_dna_sum, dtype=np.float32)).astype(np.uint8)
                self.dna[birth_idx] = avg_dna[birth_idx]

            death_idx = np.where(slime & ~survive)
            new_board[death_idx] = 0
            new_role[death_idx] = 0
            delta_mass[s] -= death_idx[0].size

            # --- Mitosis (now just a cooldown) ---
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

            # --- Combat with Aura Effects ---
            atk_mask = mine & (self.role == 4)
            if atk_mask.any():
                enemy = (self.board != 0) & (self.board != s)
                enemy_adj = self._neigh(atk_mask) > 0
                
                # Defenders project an aura that makes surrounding friendly cells tougher
                fortified_enemy = enemy & (self._neigh(self.role == 3) > 1)
                
                conquer_targets = enemy_adj & enemy & ~fortified_enemy
                
                if conquer_targets.any():
                    trg_idx = np.where(conquer_targets)
                    conquered_strains = self.board[trg_idx]

                    new_board[trg_idx] = s
                    new_role[trg_idx] = 1  # becomes slime
                    
                    # DNA blending on conquest
                    if self.dna is not None:
                        # Create a map of DNA for attackers of the current strain
                        attacker_dna_map = np.zeros_like(self.dna)
                        s_attackers = mine & (self.role == 4)
                        attacker_dna_map[s_attackers] = self.dna[s_attackers]

                        # For each conquered cell, find the sum of neighboring attacker DNA
                        sum_neighbor_attacker_dna = np.zeros_like(self.dna, dtype=np.uint32)
                        num_neighbor_attackers = np.zeros_like(self.dna, dtype=np.uint8)

                        for dy in (-1, 0, 1):
                            for dx in (-1, 0, 1):
                                if dy == 0 and dx == 0:
                                    continue
                                
                                rolled_dna = np.roll(np.roll(attacker_dna_map, dy, 0), dx, 1)
                                rolled_is_attacker = np.roll(np.roll(s_attackers, dy, 0), dx, 1)

                                sum_neighbor_attacker_dna += rolled_dna
                                num_neighbor_attackers += rolled_is_attacker
                                
                        # Get victim DNA
                        victim_dna = self.dna[trg_idx].astype(np.uint32)

                        # Calculate average attacker DNA, only for conquered cells that have attacker neighbors
                        avg_attacker_dna = np.divide(
                            sum_neighbor_attacker_dna[trg_idx], 
                            num_neighbor_attackers[trg_idx], 
                            where=num_neighbor_attackers[trg_idx] > 0,
                            out=np.zeros_like(victim_dna, dtype=np.float32)
                        ).astype(np.uint32)

                        # Blend the DNA (average of victim's original DNA and the average of surrounding attackers)
                        # Only blend where there was at least one attacker
                        has_attackers_mask = avg_attacker_dna > 0
                        
                        # Use a temporary array to store the blended results
                        blended_dna_results = victim_dna.copy()
                        
                        # Perform the blend only on the elements that have attackers
                        if np.any(has_attackers_mask):
                            blended_values = (avg_attacker_dna[has_attackers_mask] + victim_dna[has_attackers_mask]) // 2
                            blended_dna_results[has_attackers_mask] = blended_values

                        self.dna[trg_idx] = blended_dna_results.astype(np.uint16)

                    delta_mass[s] += trg_idx[0].size
                    np.add.at(delta_mass, conquered_strains, -1)

                    # On conquest, high chance to spawn a Pioneer
                    is_pioneer = self.rng.random(trg_idx[0].size) < 0.75
                    new_role[trg_idx[0][is_pioneer], trg_idx[1][is_pioneer]] = 1 # Becomes slime, but will move next turn

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
            
            # --- Pioneer Slime Movement (Smarter) ---
            is_frontier = (self.role == 1) & mine & (self._neigh(empty) > 0)
            pioneer_chance = 0.3 + ((self.dna[mine] >> 12) & 0xF) / 50.0 # DNA-based mobility
            pioneers = is_frontier & (self.rng.random(self.board.shape) < pioneer_chance[0])
            
            if pioneers.any():
                # Prefer moves that maintain contact with the colony
                colony_contact_count = self._neigh(mine)
                best_move_quality = np.zeros_like(self.board, dtype=np.uint8)
                best_move_dir = np.zeros_like(self.board, dtype=np.int8)

                # Evaluate 8 directions
                for i, (dy, dx) in enumerate([(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]):
                    # Quality of the target cell (how many empty neighbors it has)
                    move_quality = np.roll(colony_contact_count, (dy, dx))
                    # Is the target cell empty?
                    is_valid_target = np.roll(pioneers, (dy, dx)) & empty
                    
                    # Update best move if this one is better and valid
                    is_better = move_quality > best_move_quality
                    update_mask = is_better & is_valid_target
                    best_move_quality[update_mask] = move_quality[update_mask]
                    best_move_dir[update_mask] = i + 1 # Store direction 1-8

                # Execute the best moves
                for i, (dy, dx) in enumerate([(-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1)]):
                    move_mask = pioneers & (best_move_dir == i + 1)
                    if move_mask.any():
                        # Target position for these movers
                        target_pos = np.roll(move_mask, (dy, dx))
                        
                        # Move the pioneer
                        new_board[target_pos] = s
                        new_role[target_pos] = 1
                        if self.dna is not None:
                            self.dna[target_pos] = self.dna[move_mask]
                        
                        # Clear the old position
                        new_board[move_mask] = 0
                        new_role[move_mask] = 0
                        
                        # Prevent conquered cells in the same tick from being pioneer targets
                        empty[target_pos] = False

            self.last_conquer_mask |= conquer_targets

        # --- Post-step mutations with Aura Effects ---
        # Slime near recent combat mutates
        combat_zone = np.zeros_like(self.board, dtype=bool)
        if self.last_conquer_mask.any():
            from scipy.ndimage import binary_dilation
            combat_zone = binary_dilation(self.last_conquer_mask, iterations=3)

        # Slime in dense areas mutates
        slime_board = (self.role == 1)
        slime_neigh = self._neigh(slime_board)
        dense_area = (slime_neigh > 5)

        # Apply mutations
        for s in range(1, self.strains + 1):
            my_slime = (self.board == s) & (self.role == 1)
            
            # Combat mutation
            eligible_combat = my_slime & combat_zone
            if eligible_combat.any():
                mutating = np.where(eligible_combat)
                new_roles = self.rng.choice([3, 4], size=mutating[0].size)
                new_role[mutating] = new_roles

            # Density mutation
            eligible_density = my_slime & dense_area & ~eligible_combat # prevent double mutation
            if eligible_density.any():
                # Apply with a higher probability
                mutate_prob = self.rng.random(eligible_density.sum()) < 0.10 # ACCELERATED EVOLUTION
                mutating_indices = np.where(eligible_density)[0][mutate_prob]
                mutating_coords = (np.where(eligible_density)[0][mutate_prob], np.where(eligible_density)[1][mutate_prob])
                if mutating_coords[0].size > 0:
                    new_role[mutating_coords] = 2

            # Attacker aura inspires more attackers
            attacker_rally = my_slime & attacker_aura
            aggression_gene = (self.dna[attacker_rally] >> 4) & 0xF
            if attacker_rally.any():
                 if self.rng.random() < 0.10 + aggression_gene.mean()/200.0:
                    new_role[np.where(attacker_rally)] = 4

        # --- Anti-Stalemate Protocol (Toned Down) ---
        changed_mask = (self.board != new_board) | (self.role != new_role)
        self.idle_counter[~changed_mask] += 1
        self.idle_counter[changed_mask] = 0

        stale_mask = self.idle_counter > 300 # Much longer fuse
        if stale_mask.any():
            # 50% chance to just die, otherwise revert to slime to try something new
            dies = stale_mask & (self.rng.random(self.board.shape) < 0.5)
            reverts = stale_mask & ~dies
            
            new_board[dies] = 0
            new_role[dies] = 0
            self.idle_counter[dies] = 0

            new_role[reverts] = 1 # Revert to slime
            self.idle_counter[reverts] = 0

        self.last_conquer_mask.fill(False)
        self.board[:] = new_board
        self.role[:] = new_role
        self.mito_clk[:] = new_mito
        self.mass = (self.mass + delta_mass).clip(0).astype(np.uint32)

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
