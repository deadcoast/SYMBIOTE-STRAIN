"""
High-clarity visualiser for Symbiote Strains.
Focuses on readability and sharp, distinct graphics.

Keys:
  P       Pause/Unpause the simulation.
  R       Reset the simulation to a new random state.
  M       Toggle the DNA color variation effect.
  Q/Esc   Quit the application.
"""
from __future__ import annotations
import numpy as np
import pygame as pg
from pygame import surfarray
import colorsys
from scipy.ndimage import distance_transform_edt

class Viewer:
    def __init__(self, sim, scale=4, dna_view=True):
        self.sim = sim
        self.scale = max(scale, 4) # Sprites need a minimum scale
        self.h, self.w = sim.height, sim.width
        self.dna_view = dna_view

        pg.init()
        self.screen = pg.display.set_mode((self.w * self.scale, self.h * self.scale))
        pg.display.set_caption("SYMBIOTE STRAINS")
        self.clock = pg.time.Clock()
        self.paused = False

        self._sprites = self._create_role_sprites()
        self._aura_cache = {}

    def _create_role_sprites(self) -> dict:
        """Create high-contrast, readable sprites for each role."""
        s = self.scale
        sprites = {}

        # Attacker: Aggressive Chevron
        attacker_surf = pg.Surface((s, s), pg.SRCALPHA)
        pg.draw.polygon(attacker_surf, (255, 255, 255), 
                        [(s//2, s//4), (s*3//4, s//2), (s//2, s*3//4), (s//4, s//2)])
        sprites[4] = attacker_surf
        
        # Defender: Reinforced Cross
        defender_surf = pg.Surface((s, s), pg.SRCALPHA)
        pg.draw.rect(defender_surf, (255, 255, 255), (s//4, s*3//8, s//2, s//4))
        pg.draw.rect(defender_surf, (255, 255, 255), (s*3//8, s//4, s//4, s//2))
        sprites[3] = defender_surf

        # Mitosis Sac: Glowing Nucleus
        mitosis_surf = pg.Surface((s, s), pg.SRCALPHA)
        pg.draw.circle(mitosis_surf, (255, 255, 220, 150), (s//2, s//2), s//3)
        pg.draw.circle(mitosis_surf, (255, 255, 255), (s//2, s//2), s//5)
        sprites[2] = mitosis_surf

        return sprites

    def _get_aura_surface(self, role_id: int, color: tuple) -> pg.Surface:
        """Create or retrieve a cached surface for a tactical aura border."""
        if (role_id, color) in self._aura_cache:
            return self._aura_cache[(role_id, color)]

        s = self.scale
        aura_surf = pg.Surface((s, s), pg.SRCALPHA)
        pg.draw.rect(aura_surf, color + (180,), (0, 0, s, s), 1) # 1 pixel border
        self._aura_cache[(role_id, color)] = aura_surf
        return aura_surf

    def _frame_surface(self) -> pg.Surface:
        """Render the complete game state to a surface."""
        # 1. Create the base color layer from simulation state
        board = self.sim.board
        role = self.sim.role
        dna = self.sim.dna if hasattr(self.sim, 'dna') and self.dna_view else np.zeros_like(board, dtype=np.uint16)
        
        # Base hue by strain, with slight DNA variation
        hue = (board / self.sim.strains + ((dna & 0xFF) / 255.0) * 0.1) % 1.0
        
        # Saturation and Value by role for clear distinction
        saturation = np.choose(role, [0, 0.7, 0.5, 0.9, 0.9])
        value = np.choose(role, [0, 0.8, 1.0, 0.7, 1.0])
        
        hsv = np.stack([hue, saturation, value], axis=-1)
        rgb = np.array([colorsys.hsv_to_rgb(*h) for h in hsv.reshape(-1, 3)]).reshape(self.h, self.w, 3)
        rgb = (rgb * 255).astype(np.uint8)

        # 2. Apply Fog of War
        is_live = board > 0
        dist = distance_transform_edt(~is_live)
        fog_brightness = np.clip(1.0 - dist / 24, 0.05, 1.0)
        rgb = (rgb * fog_brightness[..., np.newaxis]).astype(np.uint8)
        rgb[~is_live] = [10, 10, 15] # Dark background

        # 3. Create Pygame surface and scale it up
        surf = pg.Surface((self.w, self.h))
        surfarray.blit_array(surf, rgb.swapaxes(0, 1))
        scaled_surf = pg.transform.scale(surf, self.screen.get_size())

        # 4. Draw tactical borders and role sprites on the scaled surface
        aura_defender = self.sim._neigh(role == 3) > 0
        aura_attacker = self.sim._neigh(role == 4) > 0
        
        defender_border = self._get_aura_surface(3, (100, 150, 255))
        attacker_border = self._get_aura_surface(4, (255, 100, 100))

        s = self.scale
        for y, x in np.argwhere(board > 0):
            if aura_defender[y, x]:
                scaled_surf.blit(defender_border, (x * s, y * s))
            if aura_attacker[y, x]:
                scaled_surf.blit(attacker_border, (x * s, y * s))
                
            if role[y, x] in self._sprites:
                scaled_surf.blit(self._sprites[role[y, x]], (x * s, y * s))
        
        return scaled_surf

    def run(self, max_ticks=None):
        """Main display and event loop."""
        tick = 0
        while True:
            for e in pg.event.get():
                if e.type == pg.QUIT or (e.type == pg.KEYDOWN and e.key in (pg.K_ESCAPE, pg.K_q)):
                    return
                elif e.type == pg.VIDEORESIZE:
                    self.scale = max(e.w / self.w, e.h / self.h)
                    self._sprites = self._create_role_sprites()
                    self._aura_cache = {}
                elif e.type == pg.KEYDOWN:
                    if e.key == pg.K_p: self.paused = not self.paused
                    if e.key == pg.K_m: self.dna_view = not self.dna_view
                    if e.key == pg.K_r: 
                        self.sim = self.sim.__class__(h=self.h, w=self.w, strains=self.sim.strains)
                        init_dna(self.sim)

            if not self.paused:
                self.sim.step()
                tick += 1
                if max_ticks and tick >= max_ticks:
                    return

            frame = self._frame_surface()
            self.screen.blit(frame, (0, 0))
            pg.display.flip()
            caption = f"SYMBIOTE STRAINS | Gen: {tick}"
            if self.paused: caption += " (PAUSED)"
            pg.display.set_caption(caption)
            self.clock.tick(60)

from ..core import Simulation
from ..dna import init_dna
