"""
Class structure for Wolf objects that will be used in the ABM.

It is probably more efficient to have one standard predator class that both 
Lynx and Wolf can inherit from (but this shouldn't effect the results). 

"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import random
import mesa
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector


class Wolf:
    def __init__(self, unique_id, model, pos):
        self.id = unique_id
        self.model = model
        self.pos = pos

        self.energy = random.randint(model.params["wolf_E_min"],
                                     model.params["wolf_E_max"])
        self.age = 0
        self.sex = random.choice(["M", "F"]) 

    def step(self):
        p = self.model.params

        # with each step age increase, energy decreases
        self.age += 1
        self.energy -= p["wolf_energy_decay"]

        # if energy gets too low, or age gets too high = die
        if self.energy <= 0 or self.age >= p["wolf_max_age"]:
            self.model.remove_agent(self)
            return

        # move
        self.pos = self._move_random()

        # hunt if energy/hunger below threshold
        if self.energy < p["wolf_hunt_threshold"]:
            deer = self.model.find_any_deer_within(self.pos, radius=p["wolf_hunt_radius"])
            if deer is not None and random.random() < p["wolf_p_kill"]:
                self.model.remove_deer(deer)
                self.energy = min(p["wolf_Emax"], self.energy + p["wolf_eat_gain"])

        # 4) reproduce (need enough energy to be able to reproduce)
        if self.energy >= p["wolf_repro_threshold"]:
            if random.random() < p["wolf_p_reproduce"]:
                baby_pos = self._pick_empty_neighbor()
                if baby_pos is not None:
                    self.model.add_wolf(pos=baby_pos)
                    self.energy -= p["wolf_birth_cost"]  

    def _move_random(self):
        neighbors = self.model.landscape.neighbors(self.pos)
        
        neighbors = [c for c in neighbors if self.model.landscape.is_passable(c)]
        return random.choice(neighbors) if neighbors else self.pos

    def _pick_empty_neighbor(self):
        neighbors = self.model.landscape.neighbors(self.pos)
        candidates = [c for c in neighbors
                      if self.model.landscape.is_passable(c)
                      and self.model.is_cell_empty_of_wolf(c)]
        return random.choice(candidates) if candidates else None
    

params = {
  "wolf_E_min": 10,
  "wolf_E_max": 20,
  "wolf_Emax": 30,

  "wolf_energy_decay": 0.8,       # energy drops each step
  "wolf_hunt_threshold": 9,   # if energy drops beneath = hunger
  "wolf_hunt_radius": 1,
  "wolf_p_kill": 0.4,
  "wolf_eat_gain": 8,

  "wolf_repro_threshold": 22,
  "wolf_p_reproduce": 0.02,     # low probability??
  "wolf_birth_cost": 8,

  "wolf_max_age": 3650,         # ~10 years if 1 step/day 
}
