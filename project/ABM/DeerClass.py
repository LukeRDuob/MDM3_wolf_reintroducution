import numpy as np

from mesa import Agent


class Deer(Agent):

    def __init__(
            self, 
            model, 
            heading,
            speed = 10,
            sensing_radius = 10,
            reporduction_rate = 0.1,
            death_rate = 0.01,
            predator = "Deer"
        ):
    
        super().__init__(model)

        # General agent attributes
        self.heading = heading
        self.speed = speed
        self.sensing_radius = sensing_radius
        self.reproduction_rate = reporduction_rate
        self.death_rate = death_rate

        # added lifespan counter
        self.age = 0
        self.sex = self.model.rng.choice(["M", "F"]) 


    def step(self):

        # with each step age increase, energy decreases
        self.age += 1

        # move
        self.pos = self._move_random()

        # reproduce
        self.maybe_reproduce()

        # die
        self.maybe_die()

    def _move_random(self):
        neighbors = self.model.landscape.neighbors(self.pos)
        
        neighbors = [c for c in neighbors if self.model.landscape.is_passable(c)]
        return self.model.rng.choice(neighbors) if neighbors else self.pos

    def _pick_empty_neighbor(self):
        neighbors = self.model.landscape.neighbors(self.pos)
        candidates = [c for c in neighbors
                      if self.model.landscape.is_passable(c)
                      and self.model.is_cell_empty_of_lynx(c)]
        return self.model.rng.choice(candidates) if candidates else None
    
    def maybe_reproduce(self):

        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_pos = self._pick_empty_neighbor()
            if baby_pos is not None:
                self.model.add_deer(pos=baby_pos)

    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate:
            self.model.remove_agent(self)

    
