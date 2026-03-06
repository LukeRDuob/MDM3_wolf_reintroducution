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


# Simpler agent inherited from Deer
class Wolf(mesa.Agent):

    def __init__(
            self, 
            model, 
            heading,
            # pack_id,  # unique identifier describing the unique pack the wolf is part of (will be used for movement and feeding)
            speed = 10,
            sensing_radius = 10,
            reporduction_rate = 0.1,
            death_rate = 0.01,

            
        ):
    
        super().__init__(model)

        # General agent attributes
        self.heading = heading
        self.speed = speed
        self.sensing_radius = sensing_radius
        self.reproduction_rate = reporduction_rate
        self.death_rate = death_rate
        self.age = 0
        self.sex = self.model.rng.choice(["M", "F"]) 
        self.children = []




    def step(self):

        # with each step age increase, energy decreases
        self.age += 1

        # move
        self.pos = self._move_random()
        # self.pos = self.move()  # More complex movement that hasn't been tested
        # hunt
        self.hunt()

        # reproduce
        self.maybe_reproduce()

        # die
        self.maybe_die()

    def move(self):

        # Import locally to avoid circular imports
        from DeerClass import Deer
        from LynxClass import Lynx
        model = self.model
        p = model.params
        # Get all neighbours within sensing radius
        all_neigbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)]

        # Assign weights
        self.pack_headings = []
        self.flee_headings = []
        self.hunt_headings = []

        self.hunt_weight = 0.4
        self.pack_weight = 0.4
        self.flee_weight = 0.2
        for n in all_neigbours:
            # If prey in sensing radius then move towards
            if isinstance(n, Deer):
                # get heading for following Deer
                self.h_heading = self.model.space.get_heading(self.pos, n.pos)
                self.h_heading /= np.linalg.norm(self.h_heading)
                self.hunt_headings.append(self.h_heading)
            # If wolf in the same pack in sensing radius then move in the same heading 
            elif isinstance(n, Wolf):
                if self.pack_id == n.pack_id:
                    self.p_heading = n.heading
                    self.p_heading /= np.linalg.norm(self.p_heading)
                    self.pack_headings.append(self.p_heading)
                else:
                    # Flee
                    self.f_heading = -self.model.space.get_heading(self.pos, n.pos)
                    self.f_heading /= np.linalg.norm(self.f_heading)
                    self.flee_headings.append(self.f_heading)

            # If wolf from different pack or a lynx in sensing radius then move in an opposing direction 
            elif isinstance(n, Lynx) or isinstance(n, Wolf):
                self.f_heading = -self.model.space.get_heading(self.pos, n.pos)
                self.f_heading /= np.linalg.norm(self.f_heading)
                self.flee_headings.append(self.f_heading)

        # Combine heading influences for a final movement direction
        self.flee_heading = np.mean(self.flee_headings, axis=0) if len(self.flee_headings)>0 else np.zeros_like(self.heading)
        self.pack_heading = np.mean(self.pack_headings, axis=0) if len(self.pack_headings)>0 else np.zeros_like(self.heading)
        self.hunt_heading = np.mean(self.hunt_headings, axis=0) if len(self.hunt_headings)>0 else np.zeros_like(self.heading)

        # If all headings are zero, move randomly
        if (not self.flee_headings and not self.pack_headings and not self.hunt_headings):
            rand_heading = np.random.uniform(-1, 1, size=self.heading.shape)
            norm = np.linalg.norm(rand_heading)
            if norm > 0:
                rand_heading /= norm
            self.heading = rand_heading
        else:
            # Use weighted sum to combine
            self.new_heading = (self.flee_weight * self.flee_heading) + (self.hunt_weight * self.hunt_heading) + (self.pack_weight * self.pack_heading)
            norm = np.linalg.norm(self.new_heading)
            if norm > 0:
                self.new_heading /= norm
            self.heading = self.new_heading
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
    
    def hunt(self):
        '''
            This method will be modified when pack dynamics are added to the model (could 
            increase kill probability when there are a larger number of adult wolves in the pack)
        '''
        # Import Deer class locally to avoid circular import
        from DeerClass import Deer
        # Wolves hunt deer in their current position or within killing radius
        model = self.model
        
        # Get all agents in the wolf's killing neigbourhood (circular neighbourhood with attack radius)
        deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, model.wolf_attack_radius, True) if isinstance(n, Deer)]
        # Try to kill the deer if found
        if len(deer_neighbours) > 0:
            # If deer neighbours nearby then attack the closest
            other = self.ret_closest_neighbour(deer_neighbours)
            kill_chance = random.uniform(0,1)
            if kill_chance < model.wolf_kill_prob:
                # Feed (should also feed the whole pack of wolves)
                self.feed()
                # Remove deer
                model.remove_deer(other)

                
    def feed(self):
        '''
            This method will be modified when pack dynamics are added to the model
        '''
        # Get model and key params
        model = self.model
        p = model.params
        # feed all members of the pack 
        # self.pack_members = model.get_pack_members(self.pack_id)
        # for w in self.pack_members:
        #     w.energy = min(p["wolf_Emax"], self.energy + p["wolf_eat_gain"])

        self.energy = min(p["wolf_Emax"], self.energy + p["wolf_eat_gain"])

    
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


    def ret_closest_neighbour(self, neighbours):
        """
        Returns the closest neighbour from a given set of neighbours
        """
        neighbours_distances = np.array([[n, self.model.space.get_distance(self.pos, n.pos)] for n in neighbours])
        return neighbours_distances[neighbours_distances[:,1].argsort()][0][0]


# Copy of Lynx class

# class Wolf:
#     def __init__(self, unique_id, model, pos):
#         self.id = unique_id
#         self.model = model
#         self.pos = pos

#         self.energy = random.randint(model.params["wolf_E_min"],
#                                      model.params["wolf_E_max"])
#         self.age = 0
#         self.sex = random.choice(["M", "F"]) 

#     def step(self):
#         p = self.model.params

#         # with each step age increase, energy decreases
#         self.age += 1
#         self.energy -= p["wolf_energy_decay"]

#         # if energy gets too low, or age gets too high = die
#         if self.energy <= 0 or self.age >= p["wolf_max_age"]:
#             self.model.remove_agent(self)
#             return

#         # move
#         self.pos = self._move_random()

#         # hunt if energy/hunger below threshold
#         if self.energy < p["wolf_hunt_threshold"]:
#             deer = self.model.find_any_deer_within(self.pos, radius=p["wolf_hunt_radius"])
#             if deer is not None and random.random() < p["wolf_p_kill"]:
#                 self.model.remove_deer(deer)
#                 self.energy = min(p["wolf_Emax"], self.energy + p["wolf_eat_gain"])

#         # 4) reproduce (need enough energy to be able to reproduce)
#         if self.energy >= p["wolf_repro_threshold"]:
#             if random.random() < p["wolf_p_reproduce"]:
#                 baby_pos = self._pick_empty_neighbor()
#                 if baby_pos is not None:
#                     self.model.add_wolf(pos=baby_pos)
#                     self.energy -= p["wolf_birth_cost"]  

#     def _move_random(self):
#         neighbors = self.model.landscape.neighbors(self.pos)
        
#         neighbors = [c for c in neighbors if self.model.landscape.is_passable(c)]
#         return random.choice(neighbors) if neighbors else self.pos

#     def _pick_empty_neighbor(self):
#         neighbors = self.model.landscape.neighbors(self.pos)
#         candidates = [c for c in neighbors
#                       if self.model.landscape.is_passable(c)
#                       and self.model.is_cell_empty_of_wolf(c)]
#         return random.choice(candidates) if candidates else None
    

# params = {
#   "wolf_E_min": 10,
#   "wolf_E_max": 20,
#   "wolf_Emax": 30,

#   "wolf_energy_decay": 0.8,       # energy drops each step
#   "wolf_hunt_threshold": 9,   # if energy drops beneath = hunger
#   "wolf_hunt_radius": 1,
#   "wolf_p_kill": 0.4,
#   "wolf_eat_gain": 8,

#   "wolf_repro_threshold": 22,
#   "wolf_p_reproduce": 0.02,     # low probability??
#   "wolf_birth_cost": 8,

#   "wolf_max_age": 3650,         # ~10 years if 1 step/day 
# }
