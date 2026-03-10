"""
Class structure for Wolf objects that will be used in the ABM.

It is probably more efficient to have one standard predator class that both 
Lynx and Wolf can inherit from (but this shouldn't effect the results). 

"""
import numpy as np
import mesa


# Simpler agent inherited from Deer
class Wolf(mesa.Agent):

    def __init__(
            self, 
            model, 
            heading,
            # pack_id,  # unique identifier describing the unique pack the wolf is part of (will be used for movement and feeding)
            speed = 1,
            sensing_radius = 10,
            kill_prob = 0,
            reproduction_rate = 0.03,
            death_rate = 0.01,
            species = "Wolf",
            energy_increase = 1,
            starting_energy_bounds = [0.8,1],  # Assuming energy is in the range [0,1] 
            attack_radius = 5,  # radius within which wolves can attack deer 

        ):
    
        super().__init__(model)

        # General agent attributes
        self.heading = heading
        self.sensing_radius = sensing_radius
        self.reproduction_rate = reproduction_rate
        self.death_rate = death_rate
        self.energy_increase = energy_increase
        self.energy = self.model.rng.uniform(starting_energy_bounds[0], starting_energy_bounds[1])
        self.species = species
        self.kill_prob = kill_prob
        self.speed = speed
        self.wolf_attack_radius = attack_radius


    def step(self):

        # Move
        self.move_random()
        
        # Hunt
        self.hunt()

        # Reproduce
        self.maybe_reproduce()

        # Energy decreases
        self.lose_energy()

        # Die
        self.maybe_die()
        

    def move_random(self):

        """
        Move according to a random walk.
        """
        # Set a random heading
        self.heading += np.random.random(2) * 2 - 1
        self.heading /= np.linalg.norm(self.heading)

        # Calculate new position
        self.pos += self.heading * self.speed

        # Move the agent in space
        self.model.space.move_agent(self, self.pos)

    def hunt(self):
        '''
            This method will be modified when pack dynamics are added to the model (could 
            increase kill probability when there are a larger number of adult wolves in the pack)
        '''
        # Wolves hunt deer in their current position or within killing radius

        
        # Get all agents in the wolf's killing neigbourhood (circular neighbourhood with attack radius)
        deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.wolf_attack_radius, True) if n.species=="Deer"]
        
        # Try to kill the deer if found
        if len(deer_neighbours) > 0:
            # If deer neighbours nearby then attack the closest
            other = self.ret_closest_neighbour(deer_neighbours)
            kill_chance = self.model.rng.uniform(0,1)
            if kill_chance < self.kill_prob:
                # Feed (will feed the whole pack of wolves in later developments)
                self.feed()
                # Remove deer
                other.remove()

                # model.remove_deer(other)

                
    def feed(self):
        '''
            This method will be modified when pack dynamics are added to the model
        '''
        # Increase energy
        self.energy += self.energy_increase            


    def lose_energy(self):
        """ 
            Constant energy loss per step (could be changed to exponential decay)
            (as a function of age later?)
            Could move to the Agent classes   

        """
        self.energy -= self.model.energy_decrease
               

    def maybe_reproduce(self):


        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_heading = self.model.random_heading()
            baby = Wolf(self.model, heading=baby_heading)
            self.model.space.place_agent(baby, self.pos)


    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate:
            self.remove()

        # Also remove agent if energy at minimum energy
        elif self.energy == self.model.energy_min:
            self.remove()


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
