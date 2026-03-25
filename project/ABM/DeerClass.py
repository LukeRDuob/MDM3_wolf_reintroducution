import numpy as np
from mesa import Agent


class Deer(Agent):

    def __init__(
            self, 
            model,
            heading,
            speed = 8000,  # 8km/h (to be changed)
            sensing_radius = 3000,  # (to be changed)
            reproduction_rate = 0.02,  # (to be changed)
            death_rate = 0.01,  # (to be changed)
            species = "Deer",
            # Movement weightings
            flee_weight = 4,
            follow_food_weight = 1,
            eating_radius=20, #random change!!
            # Energy
            starting_energy_bounds = [0.8, 1],
            energy_increase = 0.2
        ):
    
        super().__init__(model) 

        # General agent attributes
        self.heading = heading
        self.speed = speed
        self.sensing_radius = sensing_radius
        self.reproduction_rate = reproduction_rate
        self.death_rate = death_rate

        # Movement weightings
        self.flee_weight = flee_weight
        self.follow_food_weight = follow_food_weight
        
        # added lifespan counter
        self.age = 0
        self.sex = self.model.rng.choice(["M", "F"]) 
        self.species = species

        # Energy
        self.energy = self.model.rng.uniform(starting_energy_bounds[0], starting_energy_bounds[1])
        


        self.eating_radius = eating_radius


    def step(self):

        # with each step age increase, energy decreases
        self.age += 1

        # Move
        if self.model.use_random_movement:
            self.move_random()
        else:
            self.move()  # More complex movement 
        
        # graze in that grid cell
        self.graze()

        # reproduce
        self.maybe_reproduce()

        # lose energy
        self.lose_energy()

        # die
        self.maybe_die()


    def _add_angular_noise(self, heading, max_angle=np.pi / 6):
        """
        Rotates a 2D heading vector by a random angle 
        within [-max_angle, max_angle] (default +-30 degrees).
        """
        angle = self.model.rng.uniform(-max_angle, max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        return rotation_matrix @ heading
    
    def _normalise(self, heading):
            norm = np.linalg.norm(heading)
            if norm > 0:
                return heading / norm
            else:
                return self._add_angular_noise(self.heading.copy())
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


    def move(self):
        # Get all neighbours within sensing radius
        deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Deer']
        wolf_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Wolf']
        veg_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Vegetation']
        sapling_neighbours = [v for v in veg_neighbours if v.stage == 'sapling']
        
        # If wolf in radius then flee 
        if len(wolf_neighbours) > 0:
            # run away from closest wolf
            closest_wolf = self.ret_closest_neighbour(wolf_neighbours)
            flee_heading = -self.model.space.get_heading(self.pos, closest_wolf.pos)
            flee_heading = self._normalise(flee_heading)
            # flee_headings = []
            # for w in wolf_neighbours:
            #     # get heading for following Deer
            #     f_heading =  -self.model.space.get_heading(self.pos, w.pos)
            #     f_heading = self._normalise(f_heading)
            #     flee_headings.append(f_heading)
            # flee_heading = np.mean(flee_headings, axis=0)
        

        # If food in sensing radius then move towards closest sapling
        if len(sapling_neighbours) > 0:
            
            closest_sap = self.ret_closest_neighbour(sapling_neighbours)
            sapling_heading = self.model.space.get_heading(self.pos, closest_sap.pos)
            food_heading = self._normalise(sapling_heading)

        # Combine heading influences for a final movement direction
        # If all headings are zero, move along original heading with some noise
        if len(wolf_neighbours) == 0 and len(sapling_neighbours) == 0:
            new_heading = self._add_angular_noise(self.heading) 
        
        elif len(sapling_neighbours) == 0:
            # Use weighted sum to combine
            new_heading = self.flee_weight * flee_heading

        elif len(wolf_neighbours) == 0:
            # Use weighted sum to combine
            new_heading = self.follow_food_weight * food_heading

        else:
            # Use weighted sum to combine
            new_heading = (self.flee_weight * flee_heading) + (self.follow_food_weight * food_heading)

        norm = np.linalg.norm(new_heading)
        self.heading = self._normalise(new_heading)

        # Move the agent
        new_pos = self.pos + (self.heading * self.speed)
        self.model.space.move_agent(self, new_pos)

    def graze(self):
        vegetation_neighbours = [
            n for n in self.model.space.get_neighbors(self.pos, self.eating_radius, True)
            if n.species == "Vegetation" and n.stage == "sapling"
        ]

        if len(vegetation_neighbours) > 0:
            plant = min(
                vegetation_neighbours, 
                key=lambda v: self.model.space.get_distance(self.pos, v.pos)
            )
            plant.remove()
            
            # Increase energy for deer 


    
    def maybe_reproduce(self):

        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_heading = self.model.random_heading()
            baby = Deer(self.model, heading=baby_heading)
            self.model.space.place_agent(baby, self.pos)

    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate or self.energy==0:
            self.remove()

    
    def ret_closest_neighbour(self, neighbours):
        """
        Returns the closest neighbour from a given set of neighbours
        """
        neighbours_distances = np.array([[n, self.model.space.get_distance(self.pos, n.pos)] for n in neighbours])
        return neighbours_distances[neighbours_distances[:,1].argsort()][0][0]
    
    def lose_energy(self):
        """ 
            Constant energy loss per step (could be changed to exponential decay)
            (as a function of age later?)
            Could move to the Agent classes   

        """
        self.energy -= self.model.energy_decrease