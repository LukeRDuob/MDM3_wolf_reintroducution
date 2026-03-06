# lynx_deer_scotland_abm.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from project.ABM.DeerClass import Deer
from project.ABM.LynxClass import Lynx
from project.ABM.WolfClass import Wolf


def clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class PredDeerModel(mesa.Model):
    """
    Predator-Deer-Woodland regeneration toy ABM.

    - OPEN vs WOODLAND habitat
    - sapling growth + browsing
    - succession: OPEN -> WOODLAND if saplings stay high long enough
    - risk heatmap driven by predator positions (landscape of fear)
    """
    def __init__(
        self,
        width: int = 50,
        height: int = 50,
        init_deer: int = 400,
        init_lynx: int = 10,
        init_wolf: int = 0,
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

        # Wolf
        self.wolf_initial_energy = 9.0
        self.wolf_step_cost = 0.8
        self.wolf_kill_prob = 0.60
        self.wolf_gain_from_kill = 7.5
        self.wolf_repro_energy_threshold = 18.0
        self.wolf_birth_prob = 0.05

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
        self._spawn_agents(init_deer, init_lynx, init_wolf)

        self.datacollector = DataCollector(
            model_reporters={
                "Deer": lambda m: len(m.agents_by_type.get(Deer, [])),
                "Lynx": lambda m: len(m.agents_by_type.get(Lynx, [])),
                "Wolf": lambda m: len(m.agents_by_type.get(Wolf, [])),
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

    def _spawn_agents(self, n_deer: int, n_lynx: int, n_wolf: int):
        for _ in range(n_deer):
            d = Deer(self)
            self.grid.place_agent(d, self._random_empty_or_any_cell())
        for _ in range(n_lynx):
            l = Lynx(self)
            self.grid.place_agent(l, self._random_empty_or_any_cell())
        for _ in range(n_wolf):
            w = Wolf(self)
            self.grid.place_agent(w, self._random_empty_or_any_cell())


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
        no_deer = len(self.agents_by_type.get(Deer, [])) == 0
        no_lynx = len(self.agents_by_type.get(Lynx, [])) == 0
        no_wolf = len(self.agents_by_type.get(Wolf, [])) == 0
        if no_deer or (no_lynx and no_wolf):
            self.running = False

    def _update_risk(self):
        self.risk *= (1.0 - self.risk_decay)
        wolf_agents = self.agents_by_type.get(Wolf, [])
        lynx_agents = self.agents_by_type.get(Lynx, [])
        for lynx in lynx_agents:
            x, y = lynx.pos
            self.risk[x, y] += self.risk_add_center

            neigh = self.grid.get_neighborhood((x, y), moore=True, include_center=False, radius=1)
            for nx, ny in neigh:
                self.risk[nx, ny] += self.risk_add_neighbor
        
        for wolf in wolf_agents:
            x, y = wolf.pos
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
    model = PredDeerModel(seed=seed)

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
    wolf_scatter = axes[1].scatter([], [], s=30, marker="*", label="Wolf")
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
        wolf_xy = np.array([a.pos for a in model.agents_by_type.get(Wolf, [])], dtype=float)

        if len(deer_xy) > 0:
            deer_scatter.set_offsets(deer_xy)
        else:
            deer_scatter.set_offsets(np.empty((0, 2)))

        if len(lynx_xy) > 0:
            lynx_scatter.set_offsets(lynx_xy)
        else:
            lynx_scatter.set_offsets(np.empty((0, 2)))

        if len(wolf_xy) > 0:
            wolf_scatter.set_offsets(wolf_xy)
        else:
            wolf_scatter.set_offsets(np.empty((0, 2)))

        deer_n = len(model.agents_by_type.get(Deer, []))
        lynx_n = len(model.agents_by_type.get(Lynx, []))
        wolf_n = len(model.agents_by_type.get(Wolf, []))

        fig.suptitle(f"Step {t} | Deer={deer_n} | Lynx={lynx_n} | Wolf={wolf_n} | WoodlandFrac={np.mean(model.habitat):.2f}")

        plt.pause(0.5)

    plt.ioff()
    plt.show()

    df = model.datacollector.get_model_vars_dataframe()
    print(df.tail())
    return df



if __name__ == "__main__":
    run_demo(steps=400, seed=2)
