import numpy as np
from mesa import Agent


class Deer(Agent):

    def __init__(
            self, 
            model,
            heading,
            age = 0,
            speed = 4,  # 4km/h when grazing and roaming generally
            sensing_radius = 1.0, 
            yearly_reproduction_rate = 0.4,  # changed from around 1 child per year to account for child mortality 
            min_breeding_age = 2, # (to be changed)
            yearly_death_rate = 0.1,  # (to be changed)
            species = "Deer",
            # Movement weightings
            eating_radius= 0.01, #(to be changed)
            # max age
            max_age = 15 # approx 15 years,
        ):
    
        super().__init__(model) 

        # General agent attributes
        self.heading = heading
        self.speed = speed * self.model.step_size
        self.sensing_radius = sensing_radius
        self.reproduction_rate = (yearly_reproduction_rate / self.model.yearly_sunlight_hours)
        self.death_rate = (yearly_death_rate / self.model.yearly_sunlight_hours)
        self.max_age = (max_age * self.model.yearly_sunlight_hours) / self.model.step_size  
        self.min_breeding_age = min_breeding_age
        

        # Movement weightings
        #self.flee_weight = flee_weight
        #self.follow_food_weight = follow_food_weight
        
        # added lifespan counter
        self.age = age
        self.sex = self.model.rng.choice(["M", "F"]) 
        self.species = species

        # Energy
        # self.energy = self.model.rng.uniform(starting_energy_bounds[0], starting_energy_bounds[1])
        # self.energy_increase = energy_increase
        # self.energy_decrease = energy_decrease * self.model.step_size


        self.eating_radius = eating_radius
        self.use_veg = self.model.use_veg


    def step(self):

        if not self.model.use_base:
            # with each step age increase
            self.age += 1 / self.model.yearly_sunlight_hours

            # Move
            if self.model.use_random_movement:
                self.move_random(self.speed)
            else:
                self.move()  # More complex movement 
        else:
                # Move randomly in base model
                self.move_random(self.speed)

        if self.use_veg:

            # graze in that grid cell
            self.graze()

        # reproduce
        if not self.model.use_base:
        
            if self.sex == "F" and self.age > self.min_breeding_age:
                self.maybe_reproduce()
        else:
            # Might need to adjust rate
            self.maybe_reproduce()

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
            return self._add_angular_noise(self.heading)


    def move_random(self, speed):
        """
        Move according to a random walk.
        """
        # Set a random heading
        self.heading += self.model.rng.random(2) * 2 - 1
        self.heading /= np.linalg.norm(self.heading)

        # Calculate new position
        new_pos = self.pos + self.heading * speed

        # Move the agent in space
        if self.model.use_boundary_conditions:
            new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions             
        self.model.space.move_agent(self, new_pos)
        self.model.spatial_hash.update(self)  # Update spatial hash after moving


    def move(self):

        # Get all neighbours within sensing radius
        # wolf_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Wolf']
        wolf_neighbours = self.model.spatial_hash.get_neighbors_by_species(
            self.pos, self.sensing_radius, 'Wolf', agent=self
        )
        # If wolf in radius then flee (Ignoring food)
        if len(wolf_neighbours) > 0:
            # Run away from closest wolf
            closest_wolf = self.ret_closest_neighbour(wolf_neighbours)
            flee_heading = self.model.space.get_heading(closest_wolf.pos, self.pos)
            flee_heading = self._normalise(flee_heading)
            flee_heading = self._add_angular_noise(flee_heading)
            self.heading = self._normalise(flee_heading)
            # Move the agent
            new_pos = self.pos + (self.heading * self.speed)
            if self.model.use_boundary_conditions:
                new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions 
            self.model.space.move_agent(self, new_pos)
            self.model.spatial_hash.update(self)  # Update spatial hash after moving

            return

        # is using veg
        if self.use_veg:

            # if no wolves in radius then check for food and move towards it if there is any, otherwise move randomly
            veg_neighbours = [
                n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)
                if n.species == "Vegetation"
            ]
            food_patches = [v for v in veg_neighbours if v.saplings > 0]
            
            # If food in sensing radius then move towards closest sapling
            if len(food_patches) > 0:

                closest_patch = self.ret_closest_neighbour(food_patches)
                patch_heading = self.model.space.get_heading(self.pos, closest_patch.pos)
                patch_heading = self._normalise(patch_heading)
                patch_heading = self._add_angular_noise(patch_heading)
                self.heading = self._normalise(patch_heading)
                # Get distance to patch to avoid overstepping
                patch_dist = self.model.space.get_distance(self.pos, closest_patch.pos)

                translation_vector = self.heading * self.speed
                translation_dist = np.linalg.norm(translation_vector)

                if translation_dist > patch_dist:
                    scale = patch_dist / translation_dist
                else:
                    scale = 1

                new_pos = self.pos + (scale * translation_vector)
                if self.model.use_boundary_conditions:
                    new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions             
                self.model.space.move_agent(self, new_pos)                
                self.model.spatial_hash.update(self)  # Update spatial hash after moving

                return

            # Fallback: random walk when no food detected
            self.heading = self._add_angular_noise(self.heading)
            new_pos = self.pos + (self.heading * self.speed)
            if self.model.use_boundary_conditions:
                new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)
            self.model.space.move_agent(self, new_pos)
            self.model.spatial_hash.update(self)  # Update spatial hash after moving

        else:

            # If no wolves or food detected then move randomly
            self.heading = self._add_angular_noise(self.heading)
            
            # Move the agent
            new_pos = self.pos + (self.heading * self.speed)
            if self.model.use_boundary_conditions:
                new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions             
            self.model.space.move_agent(self, new_pos)
            self.model.spatial_hash.update(self)  # Update spatial hash after moving



    def graze(self):

        vegetation_neighbours = [
            n for n in self.model.space.get_neighbors(self.pos, self.eating_radius, True)
            if n.species == "Vegetation" and n.saplings > 0
        ]

        if len(vegetation_neighbours) > 0:
            patch = min(
                vegetation_neighbours,
                key=lambda v: self.model.space.get_distance(self.pos, v.pos)
            )
            # deer only browses about 40% of hours, not every hour
            if self.model.rng.random() < 0.4:
                amount_eaten = 1
                patch.saplings = max(0, patch.saplings - amount_eaten)

                # Increase energy for deer
                # self.energy += self.energy_increase


    
    def maybe_reproduce(self):

        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_heading = self.model.random_heading()
            baby = Deer(self.model, heading=baby_heading)
            self.model.space.place_agent(baby, self.pos)
            self.model.spatial_hash.add(baby)  # Update spatial hash for the new agent

    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate or self.age >= self.max_age:
            self.model.spatial_hash.remove(self)
            self.remove()
            self.model.deer_deaths += 1

    
    # def ret_closest_neighbour(self, neighbours):
    #     """
    #     Returns the closest neighbour from a given set of neighbours
    #     """
    #     neighbours_distances = np.array([[n, self.model.space.get_distance(self.pos, n.pos)] for n in neighbours])
    #     return neighbours_distances[neighbours_distances[:,1].argsort()][0][0]
    
    def ret_closest_neighbour(self, neighbours):
        """Returns the closest neighbour (uses squared distance to avoid sqrt)."""
        return min(
            neighbours,
            key=lambda n: (self.pos[0] - n.pos[0])**2 + (self.pos[1] - n.pos[1])**2
        )
    # def lose_energy(self):
    #     """ 
    #         Constant energy loss per step (could be changed to exponential decay)
    #         (as a function of age later?)
    #         Could move to the Agent classes   

    #     """
    #     self.energy -= self.energy_decrease
