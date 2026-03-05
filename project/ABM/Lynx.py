# lynx.py

import random

class Lynx:
    def __init__(self, unique_id, model, pos):
        self.id = unique_id
        self.model = model
        self.pos = pos

        self.energy = random.randint(model.params["lynx_E_min"],
                                     model.params["lynx_E_max"])
        self.age = 0
        self.sex = random.choice(["M", "F"]) 

    def step(self):
        p = self.model.params

        # with each step age increase, energy decreases
        self.age += 1
        self.energy -= p["lynx_energy_decay"]

        # if energy gets too low, or age gets too high = die
        if self.energy <= 0 or self.age >= p["lynx_max_age"]:
            self.model.remove_agent(self)
            return

        # move
        self.pos = self._move_random()

        # hunt if energy/hunger below threshold
        if self.energy < p["lynx_hunt_threshold"]:
            deer = self.model.find_any_deer_within(self.pos, radius=p["lynx_hunt_radius"])
            if deer is not None and random.random() < p["lynx_p_kill"]:
                self.model.remove_deer(deer)
                self.energy = min(p["lynx_Emax"], self.energy + p["lynx_eat_gain"])

        # 4) reproduce (need enough energy to be able to reproduce)
        if self.energy >= p["lynx_repro_threshold"]:
            if random.random() < p["lynx_p_reproduce"]:
                baby_pos = self._pick_empty_neighbor()
                if baby_pos is not None:
                    self.model.add_lynx(pos=baby_pos)
                    self.energy -= p["lynx_birth_cost"]  

    def _move_random(self):
        neighbors = self.model.landscape.neighbors(self.pos)
        
        neighbors = [c for c in neighbors if self.model.landscape.is_passable(c)]
        return random.choice(neighbors) if neighbors else self.pos

    def _pick_empty_neighbor(self):
        neighbors = self.model.landscape.neighbors(self.pos)
        candidates = [c for c in neighbors
                      if self.model.landscape.is_passable(c)
                      and self.model.is_cell_empty_of_lynx(c)]
        return random.choice(candidates) if candidates else None
    

params = {
  "lynx_E_min": 10,
  "lynx_E_max": 20,
  "lynx_Emax": 30,

  "lynx_energy_decay": 1,       # energy drops each step
  "lynx_hunt_threshold": 12,   # if energy drops beneath = hunger
  "lynx_hunt_radius": 1,
  "lynx_p_kill": 0.3,
  "lynx_eat_gain": 10,

  "lynx_repro_threshold": 22,
  "lynx_p_reproduce": 0.02,     # low probability??
  "lynx_birth_cost": 8,

  "lynx_max_age": 3650,         # ~10 years if 1 step/day 
}
