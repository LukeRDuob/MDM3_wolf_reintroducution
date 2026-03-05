# lynx_deer_scotland_abm.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt

import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))



# Agents

class Deer(mesa.Agent):
    """Prey agent: moves to forage while avoiding risk, browses saplings, reproduces."""
    def __init__(self, model: "LynxDeerModel"):
        super().__init__(model)
        self.energy = model.deer_initial_energy

    def step(self):
        self.move()
        self.browse()
        self.reproduce()
        self.energy -= self.model.deer_step_cost
        if self.energy <= 0:
            self.remove()

    def move(self):
        x, y = self.pos
        
        neighbors = self.model.grid.get_neighborhood(
            (x, y), moore=True, include_center=True, radius=1
        )

        best_pos = (x, y)
        best_score = -1e9

        for nx, ny in neighbors:
            habitat = self.model.habitat[nx, ny]  # 0=open, 1=woodland
            risk = self.model.risk[nx, ny]
            sap = self.model.saplings[nx, ny]

            
            forage = (1.0 if habitat == 0 else 0.6) + 0.2 * sap

            # Deer prefer open, but will use woodland if risk is low
            habitat_pref = 0.15 if habitat == 0 else 0.0

            score = forage + habitat_pref - self.model.deer_risk_weight * risk

            # Add small noise to avoid tie-locking
            score += self.random.uniform(-0.01, 0.01)

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        if best_pos != (x, y):
            self.model.grid.move_agent(self, best_pos)

    def browse(self):
        x, y = self.pos
        # Browsing reduces saplings in that cell
        self.model.saplings[x, y] = max(
            0.0,
            self.model.saplings[x, y] - self.model.browse_rate
        )
        # Energy gain from feeding 
        habitat = self.model.habitat[x, y]
        self.energy += self.model.deer_gain_open if habitat == 0 else self.model.deer_gain_wood

    def reproduce(self):
        x, y = self.pos
        # Local crowding reduces reproduction
        deer_here = sum(1 for a in self.model.grid.get_cell_list_contents((x, y)) if isinstance(a, Deer))
        crowd_factor = clip01(1.0 - deer_here / self.model.deer_local_crowd_cap)

        deer_total = len(self.model.agents_by_type.get(Deer, []))
        global_factor = clip01(1.0 - deer_total / self.model.deer_K)

        habitat = self.model.habitat[x, y]
        base = self.model.deer_birth_rate_open if habitat == 0 else self.model.deer_birth_rate_wood

        p = base * crowd_factor * global_factor
        if self.random.random() < p:
            fawn = Deer(self.model)
            self.model.grid.place_agent(fawn, (x, y))


class Lynx(mesa.Agent):
    """Predator agent: moves toward deer, hunts, reproduces."""
    def __init__(self, model: "LynxDeerModel"):
        super().__init__(model)
        self.energy = model.lynx_initial_energy

    def step(self):
        self.move()
        self.hunt()
        self.reproduce()
        self.energy -= self.model.lynx_step_cost
        if self.energy <= 0:
            self.remove()

    def move(self):
        x, y = self.pos
        neighbors = self.model.grid.get_neighborhood(
            (x, y), moore=True, include_center=True, radius=1
        )

        best_pos = (x, y)
        best_score = -1e9

        for nx, ny in neighbors:
            # Prefer woodland/edge a bit 
            habitat = self.model.habitat[nx, ny]
            woodland_bonus = 0.2 if habitat == 1 else 0.0

            # Move toward deer density
            deer_count = sum(1 for a in self.model.grid.get_cell_list_contents((nx, ny)) if isinstance(a, Deer))

            score = woodland_bonus + 1.0 * deer_count
            score += self.random.uniform(-0.01, 0.01)

            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        if best_pos != (x, y):
            self.model.grid.move_agent(self, best_pos)

    def hunt(self):
        x, y = self.pos
        cell_agents = self.model.grid.get_cell_list_contents((x, y))
        deer = [a for a in cell_agents if isinstance(a, Deer)]
        if not deer:
            return

        # Try to kill one deer
        if self.random.random() < self.model.lynx_kill_prob:
            victim = self.random.choice(deer)
            victim.remove()
            self.energy += self.model.lynx_gain_from_kill

    def reproduce(self):
        if self.energy < self.model.lynx_repro_energy_threshold:
            return
        if self.random.random() < self.model.lynx_birth_prob:
            kitten = Lynx(self.model)
            self.model.grid.place_agent(kitten, self.pos)
            self.energy *= 0.6  # reproduction cost



# Model

class LynxDeerModel(mesa.Model):
    """
    Lynx–Deer–Woodland regeneration toy ABM.

    - OPEN vs WOODLAND habitat
    - sapling growth + browsing
    - succession: OPEN -> WOODLAND if saplings stay high long enough
    - risk heatmap driven by lynx positions (landscape of fear)
    """
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        init_deer: int = 400,
        init_lynx: int = 10,
        woodland_frac: float = 0.25,
        seed: Optional[int] = 1,
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.grid = MultiGrid(width, height, torus=True)

        # Habitat: 0=open, 1=woodland
        
        self.habitat = (np.random.default_rng(seed).random((width, height)) < woodland_frac).astype(np.int8)

        # Saplings and persistence for succession
        self.saplings = np.zeros((width, height), dtype=float)
        self.sapling_persist = np.zeros((width, height), dtype=np.int16)

        # Risk field (pressure zones)
        self.risk = np.zeros((width, height), dtype=float)

        
        # Parameters 
        
        # Deer
        self.deer_initial_energy = 6.0
        self.deer_step_cost = 0.6
        self.deer_gain_open = 0.3
        self.deer_gain_wood = 0.01
        self.deer_risk_weight = 2.2
        self.deer_K = 600
        self.browse_rate = 0.15
        self.deer_local_crowd_cap = 5  # crowding threshold
        self.deer_birth_rate_open = 0.06
        self.deer_birth_rate_wood = 0.03

        # Lynx
        self.lynx_initial_energy = 12.0
        self.lynx_step_cost = 1.0
        self.lynx_kill_prob = 0.55
        self.lynx_gain_from_kill = 8.0
        self.lynx_repro_energy_threshold = 18.0
        self.lynx_birth_prob = 0.05

        # Vegetation / succession
        self.sapling_growth = 0.06
        self.sapling_max = 3.0
        self.succession_threshold = 1.4
        self.succession_steps = 25

        # Risk field dynamics
        self.risk_decay = 0.18
        self.risk_add_center = 1.0
        self.risk_add_neighbor = 0.35

        # Create agents 
        self._spawn_agents(init_deer, init_lynx)

        self.datacollector = DataCollector(
            model_reporters={
                "Deer": lambda m: len(m.agents_by_type.get(Deer, [])),
                "Lynx": lambda m: len(m.agents_by_type.get(Lynx, [])),
                "MeanSaplings": lambda m: float(np.mean(m.saplings)),
                "WoodlandFrac": lambda m: float(np.mean(m.habitat)),
                "MeanRisk": lambda m: float(np.mean(m.risk)),
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def _random_empty_or_any_cell(self) -> Tuple[int, int]:
        # MultiGrid can hold multiple agents per cell, so just pick any cell.
        return (self.random.randrange(self.width), self.random.randrange(self.height))

    def _spawn_agents(self, n_deer: int, n_lynx: int):
        for _ in range(n_deer):
            d = Deer(self)
            self.grid.place_agent(d, self._random_empty_or_any_cell())
        for _ in range(n_lynx):
            l = Lynx(self)
            self.grid.place_agent(l, self._random_empty_or_any_cell())

    def step(self):
        # 1) Update risk map from lynx positions
        self._update_risk()
        # 2) Agents act 
        self.agents.shuffle_do("step")

        # 3) Update vegetation and succession
        self._update_vegetation()

        # 4) Collect data
        self.datacollector.collect(self)

        # Stop if extinct
        if len(self.agents_by_type.get(Deer, [])) == 0 or len(self.agents_by_type.get(Lynx, [])) == 0:
            self.running = False

    def _update_risk(self):
        self.risk *= (1.0 - self.risk_decay)

        lynx_agents = self.agents_by_type.get(Lynx, [])
        for lynx in lynx_agents:
            x, y = lynx.pos
            self.risk[x, y] += self.risk_add_center

            neigh = self.grid.get_neighborhood((x, y), moore=True, include_center=False, radius=1)
            for nx, ny in neigh:
                self.risk[nx, ny] += self.risk_add_neighbor

        # Keep bounded for display
        np.clip(self.risk, 0.0, 5.0, out=self.risk)

    def _update_vegetation(self):
        # Growth everywhere, then cap
        self.saplings += self.sapling_growth
        np.clip(self.saplings, 0.0, self.sapling_max, out=self.saplings)

        # Browsing is already applied by deer; now handle succession persistence
        high = self.saplings >= self.succession_threshold
        self.sapling_persist[high] += 1
        self.sapling_persist[~high] = 0

        # Succession: open becomes woodland if saplings persist long enough
        can_convert = (self.habitat == 0) & (self.sapling_persist >= self.succession_steps)
        self.habitat[can_convert] = 1
        self.sapling_persist[can_convert] = 0



#  visual runner 

def run_demo(steps: int = 300, seed: int = 1):
    model = LynxDeerModel(seed=seed)

    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Static base layer for habitat
    # 0=open, 1=woodland
    habitat_img = axes[0].imshow(model.habitat.T, origin="lower", interpolation="nearest")
    axes[0].set_title("Habitat (Open vs Woodland)")

    
    axes[1].set_title("Agent positions (movement)")
    base = axes[1].imshow(model.habitat.T, origin="lower", interpolation="nearest", alpha=0.35)

    deer_scatter = axes[1].scatter([], [], s=8, label="Deer", alpha=0.7)
    lynx_scatter = axes[1].scatter([], [], s=30, marker="x", label="Lynx")
    axes[1].legend(loc="upper right")

    for t in range(steps):
        if not model.running:
            print(f"Stopped early at step {t} (extinction).")
            break

        model.step()

        # Update habitat panel 
        habitat_img.set_data(model.habitat.T)

        # Agent positions
        deer_xy = np.array([a.pos for a in model.agents_by_type.get(Deer, [])], dtype=float)
        lynx_xy = np.array([a.pos for a in model.agents_by_type.get(Lynx, [])], dtype=float)

        if len(deer_xy) > 0:
            deer_scatter.set_offsets(deer_xy)
        else:
            deer_scatter.set_offsets(np.empty((0, 2)))

        if len(lynx_xy) > 0:
            lynx_scatter.set_offsets(lynx_xy)
        else:
            lynx_scatter.set_offsets(np.empty((0, 2)))

        deer_n = len(model.agents_by_type.get(Deer, []))
        lynx_n = len(model.agents_by_type.get(Lynx, []))
        fig.suptitle(f"Step {t} | Deer={deer_n} | Lynx={lynx_n} | WoodlandFrac={np.mean(model.habitat):.2f}")

        plt.pause(0.5)

    plt.ioff()
    plt.show()

    df = model.datacollector.get_model_vars_dataframe()
    print(df.tail())
    return df



if __name__ == "__main__":
    run_demo(steps=400, seed=2)
