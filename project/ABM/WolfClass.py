"""
Class structure for Wolf objects that will be used in the ABM.
"""
import numpy as np
import mesa


# Simpler agent inherited from Deer
class Wolf(mesa.Agent):

    def __init__(
            self, 
            model,
            heading,
            age = 0,
            speed = 8,  # 8km/h 
            sensing_radius = 2, # 2 km (from smell and sight)
            min_hunting_age = 1,
            hunt_energy_threshold = 0.7,  # maximum energy level to attempt hunt (to be changed)
            hunt_radius = 0.1,  # when the wolf is able to hunt the deer
            kill_prob = 0.05,  # Probability of hunt success 
            kill_energy_increase = 0.5, 
            yearly_reproduction = 0.8,  # 1 pup(s) per year
            min_breeding_age = 2, 
            yearly_death_rate = 0.1,  # 0.125 from Archie's mathematical model
            species = "Wolf",
            starting_energy_bounds = [0.2,1],  # Assuming energy is in the range [0,1] 
            # Weights for deciding which direction to move  
            pack_follow_weight = 1,
            follow_prey_weight = 2,

            # Zonal movement zones
            zone_of_repulsion = 0.005,  # Move away
            zone_of_orientation = 0.75,  # Align with heading
            zone_of_attraction = 2,  # Move towards
            
            max_age = 12, # approx 12 years in the wild 
            pack_id = None,

            num_days_without_food_before_death = 30,  # If a wolf goes without food for this many days, it dies
            num_days_before_hunting_starts_after_eating = 3,  # After eating, a wolf won't hunt for this many days (to simulate satiation and digestion time)
        ):

        super().__init__(model)

        # General agent attributes
        self.heading = heading
        self.reproduction_rate = (yearly_reproduction / self.model.yearly_sunlight_hours)
        self.death_rate = (yearly_death_rate / self.model.yearly_sunlight_hours) 
        self.max_age = (max_age * self.model.yearly_sunlight_hours) / self.model.step_size  
        self.species = species
        self.sex = self.model.rng.choice(['M','F'])
        self.age = age

        # Energy
        self.energy = self.model.rng.uniform(starting_energy_bounds[0], starting_energy_bounds[1])
        self.energy_decrease = self.model.energy_decrease * self.model.step_size
        self.kill_energy_increase = kill_energy_increase


        #active_fraction = self.model.yearly_sunlight_hours / 8760

        #self.num_steps_without_food_before_death = (
        #    num_days_without_food_before_death * 24 * active_fraction
        #    / self.model.step_size)
        
        #self.num_steps_without_food = 0

        #self.num_steps_before_hunting_starts_after_eating = (
        #   num_days_before_hunting_starts_after_eating * 24 * active_fraction
        #    / self.model.step_size)
        
        #self.num_steps_since_last_meal = 0

        # Hunting
        self.sensing_radius = sensing_radius
        self.kill_prob = kill_prob #* self.model.step_size  # Adjust kill probability for step size
        self.hunt_radius = hunt_radius 
        self.roaming_speed = speed * self.model.step_size
        self.hunt_energy_threshold = hunt_energy_threshold
        self.min_hunting_age = min_hunting_age
        self.min_breeding_age = min_breeding_age

        # Pack dynamics
        self.pack_id = pack_id
        
        # Zonal
        self.zor = zone_of_repulsion
        self.zoo = zone_of_orientation
        self.zoa = zone_of_attraction

        # Advanced movement weights
        self.follow_prey_weight = follow_prey_weight
        self.pack_follow_weight = pack_follow_weight



    def step(self):

        if not self.model.use_base:
            # With each step age increase, energy decreases
            self.age += 1 / self.model.yearly_sunlight_hours

            # Move
            self.move()  # More complex movement 
        else:
            self.move_random(self.roaming_speed)  # Move randomly in base model
            
        # Hunt
        if not self.model.use_base:
            if self.energy < self.hunt_energy_threshold and self.age > self.min_hunting_age:
                self.hunt()

        else: 
            self.hunt()  # always hunt in base model

        # Ignore energy and specific reproduction for the base model
        if not self.model.use_base:
            # Reproduce only if female and not a pup
            if self.sex == "F" and self.age > self.min_breeding_age:
                self.maybe_reproduce()

            # Energy decreases
            self.lose_energy()
        else:
            # Might need to adjust rate
            self.maybe_reproduce()

        # Die
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
        

    def zonal_movement(self, w_neighbours):
        '''
            Method that implements a zonal movement system backed up by lit
        '''
        in_repulsion = []
        in_orientation = []
        in_attraction = []

        # Sort neighbours into zones
        for w in w_neighbours:
            dist = self.model.space.get_distance(self.pos, w.pos)
            if dist < self.zor:
                in_repulsion.append(w)
            elif dist < self.zoo: 
                in_orientation.append(w)
            elif dist < self.zoa:
                in_attraction.append(w)

        # Repulsion (overrides others)
        if len(in_repulsion) > 0:
            repulsion = np.array([0.0, 0.0])
            for w in in_repulsion:
                away = self.model.space.get_heading(w.pos, self.pos)
                away = self._normalise(away)
                repulsion += away
            desired_heading = self._normalise(repulsion)
        else:
            # Orientation and Attraction
            n_influences = 0
            desired_heading = np.array([0.0,0.0])

            # Allignment: match heading of wolves in orientation zone 
            if len(in_orientation):
                mean_heading = np.mean([w.heading for w in in_orientation], axis=0)         
                alignment_heading = self._normalise(mean_heading)        
                desired_heading += alignment_heading
                n_influences += len(in_orientation)

            # Attraction: move towards wolves in attraction zone
            if len(in_attraction) > 0:
                centroid = np.mean([w.pos for w in in_attraction], axis=0)
                attraction = self.model.space.get_heading(self.pos, centroid)
                attraction = self._normalise(attraction)
                desired_heading += attraction
                n_influences += len(in_attraction)

        # Normalise and return
        new_heading = self._normalise(desired_heading)
        return new_heading
    

    def move(self):
        # First check if there is a deer that could be hunted
        # all_neighbours = self.model.space.get_neighbors(
        #     self.pos, self.sensing_radius, True
        # )
        # deer_neighbours = [n for n in all_neighbours if n.species == 'Deer']
        # wolf_neighbours = [n for n in all_neighbours if n.species == 'Wolf']

        deer_neighbours = self.model.spatial_hash.get_neighbors_by_species(
        self.pos, self.sensing_radius, 'Deer', agent=self
        )
        wolf_neighbours = self.model.spatial_hash.get_neighbors_by_species(
            self.pos, self.sensing_radius, 'Wolf', agent=self
        )


        # Filter for hunt radius from the already-found deer
        deer_to_hunt = [
            d for d in deer_neighbours 
            if self.model.space.get_distance(self.pos, d.pos) < self.hunt_radius
        ]

        if len(deer_to_hunt) > 0:
             # If prey in sensing radius then move towards closest
            close_deer = self.ret_closest_neighbour(deer_to_hunt)
            # Get heading for following Deer
            hunt_heading = self.model.space.get_heading(self.pos, close_deer.pos)
            hunt_heading = self._normalise(hunt_heading)
            self.heading = self._add_angular_noise(hunt_heading)
            self.heading = self._normalise(self.heading)

            # Get distance from deer to ensure not to overstep
            deer_dist = self.model.space.get_distance(self.pos, close_deer.pos)  
            
            # Move the agent making sure not to overstep the prey
            translation_vector = self.heading * self.roaming_speed
            translation_dist = np.linalg.norm(translation_vector)
            if translation_dist > deer_dist:
                # Scale down avoid overstep
                scale = deer_dist/translation_dist
            else:
                scale = 1    
            
            new_pos = self.pos + (scale * translation_vector)
            if self.model.use_boundary_conditions:
                new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions             
            self.model.space.move_agent(self, new_pos)
            self.model.spatial_hash.update(self)  # Update spatial hash after moving

        else:
            # If no hunting opportunity then check sensing radius for other wolves and deer

            # Get all neighbours within sensing radius
            # deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Deer']
            # wolf_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Wolf']
            
            
            # Get pack members
            # pack_members = self.model.get_pack_members(self.pack_id)
            # pack_members.remove(self)  # don't count self

            pack_members = [w for w in self.model.get_pack_members(self.pack_id) if w is not self]


            # deer_neighbours = self.model.spatial_hash.get_neighbors_by_species(
            # self.pos, self.sensing_radius, 'Deer', agent=self
            # )
            # wolf_neighbours = self.model.spatial_hash.get_neighbors_by_species(
            # self.pos, self.sensing_radius, 'Wolf', agent=self
            # )

            
            if self.model.use_pack_dynamics and len(pack_members) > 0:
                # Use boids for swam dynamics
                # pack_heading = self.boids_movement(pack_members)

                # Use zonal model
                pack_heading = self.zonal_movement(pack_members)
            
            elif len(wolf_neighbours) > 0: 
                # If another wolf in radius then follow the heading (for the first 6 wolves in the radius)
                pack_heading = np.array([0.0, 0.0])
                n_influences = 0
                for w in wolf_neighbours:
                    if n_influences < 6:
                        p_heading = w.heading.copy()
                        # Add some angular noise for stochasticity
                        # p_heading = self._add_angular_noise(p_heading, max_angle=np.pi / 6)
                        p_heading = self._normalise(p_heading)
                        pack_heading += p_heading
                        n_influences += 1
                pack_heading = self._normalise(pack_heading)

        
            if len(deer_neighbours) > 0:
                # If prey in sensing radius then move towards closest
                close_deer = self.ret_closest_neighbour(deer_neighbours)
                # Get heading for following Deer
                hunt_heading = self.model.space.get_heading(self.pos, close_deer.pos)
                hunt_heading = self._normalise(hunt_heading)


            # Combine heading influences for a final movement direction
            # If all headings are zero, move along original heading with some noise
            if len(wolf_neighbours) + len(deer_neighbours) == 0 and not (self.model.use_pack_dynamics and len(pack_members) > 0):
                new_heading = self._add_angular_noise(self.heading)

            elif len(deer_neighbours) == 0:
                new_heading = self._add_angular_noise(self.pack_follow_weight * pack_heading)
            elif len(wolf_neighbours) == 0 and not (self.model.use_pack_dynamics and len(pack_members) > 0):
                new_heading = self._add_angular_noise(self.follow_prey_weight * hunt_heading)
            else:    
                # Use weighted sum to combine
                new_heading = self._add_angular_noise((self.pack_follow_weight * pack_heading) + (self.follow_prey_weight * hunt_heading))

            new_heading = self._normalise(new_heading)
            self.heading = new_heading

            # Move the agent
            new_pos = self.pos + (self.heading * self.roaming_speed)
            if self.model.use_boundary_conditions:
                new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions             
            self.model.space.move_agent(self, new_pos)
            self.model.spatial_hash.update(self)  # Update spatial hash after moving

    def move_random(self, speed):

        """
        Move according to a random walk.
        """
        # Set a random heading
        self.heading += self.model.rng.random(2) * 2 - 1
        self.heading /= np.linalg.norm(self.heading)

        # Move the agent
        new_pos = self.pos + (self.heading * speed)
        if self.model.use_boundary_conditions:
            new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)  # Handles boundary conditions         
        self.model.space.move_agent(self, new_pos)
        self.model.spatial_hash.update(self)  # Update spatial hash after moving

    def hunt(self):
        '''
            This method will be modified when pack dynamics are added to the model (could 
            increase kill probability when there are a larger number of adult wolves in the pack)
        '''
        # Wolves hunt deer in their current position or within killing radius
        
        # Get all agents in the wolf's killing neigbourhood (circular neighbourhood with attack radius)
        # deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.hunt_radius, True) if n.species=="Deer"]
        deer_neighbours = self.model.spatial_hash.get_neighbors_by_species(
        self.pos, self.hunt_radius, 'Deer', agent=self
        )
        # Try to kill the deer if found
        if len(deer_neighbours) > 0:
            # If deer neighbours nearby then attack the closest
            other = self.ret_closest_neighbour(deer_neighbours)
            kill_chance = self.model.rng.uniform(0,1)
            if kill_chance < self.kill_prob:
                # Feed (will feed the whole pack of wolves in later developments)
                self.feed()
                # Remove deer
                self.model.spatial_hash.remove(other)  # Update spatial hash after removing deer
                other.remove()
                # Adjust hunted deer count
                self.model.hunted_deer += 1
                self.model.deer_deaths += 1


    def feed(self):
        '''
            Energy is increased to full for the wolf and its pack members (if there are any)
        '''
 

        # If using pack dynamics then the other pack members also feed 
        if self.model.use_pack_dynamics:
            pack_members = self.model.get_pack_members(self.pack_id)
            pack_size = max(1,len(pack_members))
            share = self.kill_energy_increase / pack_size
            for member in pack_members:
                member.energy = min(member.energy + share, 1.0)
        else:
            # Refill energy of wolf 
            self.energy = min(self.energy + self.kill_energy_increase, 1.0)  

    def lose_energy(self):
        """ 
            Constant energy loss per step (could be changed to exponential decay)
            (as a function of age later?)
            Could move to the Agent classes   

        """
        self.energy -= self.energy_decrease
               

    def maybe_reproduce(self):


        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_heading = self.model.random_heading()
            baby = Wolf(self.model, heading=baby_heading, pack_id=self.pack_id)    
            self.model.space.place_agent(baby, self.pos)
            self.model.spatial_hash.add(baby)  # Update spatial hash for the new agent

    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate or self.energy<=0 or self.age >= self.max_age:
            self.model.spatial_hash.remove(self)
            self.remove()
            self.model.wolf_deaths += 1

 
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