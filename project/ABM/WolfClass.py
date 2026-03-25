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
            speed = 8,
            roaming_speed = 8,  # 8km/h (to be changed)
            hunt_speed = 50,  # 50km/h (probably to be changed)
            sensing_radius = 2,   # sensing a deer/ wolf
            hunt_radius = 0.1,  # when to switch to high speed hunt
            # kill_radius = 0.01,  # how close a wolf must be to kill a deer 
            kill_prob = 0.25,  
            reproduction_rate = 0.02,  # (to be changed)
            death_rate = 0.01,  # (to be changed)
            species = "Wolf",
            starting_energy_bounds = [0.8,1],  # Assuming energy is in the range [0,1] 
            attack_radius = 5,  # radius within which wolves can attack deer 
            # Weights for deciding which direction to move  
            pack_follow_weight = 2,
            follow_prey_weight = 3,
            # Boid's 'flock' weights
            alignment_weight = 1,
            cohesion_weight = 1,
            separation_weight = 1,
            separation_radius = 0.05,
            pack_id = None
        ):


    
        super().__init__(model)

        # General agent attributes
        self.heading = heading
        self.reproduction_rate = reproduction_rate
        self.death_rate = death_rate
        self.species = species

        # Energy
        self.energy = self.model.rng.uniform(starting_energy_bounds[0], starting_energy_bounds[1])
        
        # Hunting
        self.sensing_radius = sensing_radius
        self.kill_prob = kill_prob
        self.hunt_radius = hunt_radius
        self.roaming_speed = roaming_speed
        self.hunt_speed = hunt_speed
        self.wolf_attack_radius = attack_radius


        # Pack dynamics
        self.pack_id = pack_id
        self.alignment_weight = alignment_weight
        self.cohesion_weight = cohesion_weight
        self.separation_weight = separation_weight
        self.separation_radius = separation_radius

        # Advanced movement weights
        self.follow_prey_weight = follow_prey_weight
        self.pack_follow_weight = pack_follow_weight



    def step(self):

        # Move
        if self.model.use_random_movement:
            self.move_random(self.roaming_speed)
        else:
            self.move()  # More complex movement 
        
        # Hunt
        self.hunt()

        # Reproduce
        if self.sex == "F":
            self.maybe_reproduce()

        # Energy decreases
        self.lose_energy()

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
        
    def move(self):
        # First check if there is a deer that could be hunted
        deer_to_hunt = [n for n in self.model.space.get_neighbors(self.pos, self.hunt_radius, True) if n.species == 'Deer']
        if len(deer_to_hunt) > 0:
             # If prey in sensing radius then move towards closest
            close_deer = self.ret_closest_neighbour(deer_to_hunt)
            # Get heading for following Deer
            hunt_heading = self.model.space.get_heading(self.pos, close_deer.pos)
            hunt_heading = self._normalise(hunt_heading)

            # Move the agent
            new_pos = self.pos + (self.heading * self.hunt_speed)
            self.model.space.move_agent(self, new_pos)

        else:
            # If no hunting opportunity then check sensing radius for other wolves and deer

            # Get all neighbours within sensing radius
            deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Deer']
            wolf_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Wolf']

            
            if self.model.use_pack_dynamics:
                # Use boids method for 'flocking'  (will change to more appropriate algorithm)
                # Alignment: Mean heading of pack
                pack_members = self.model.get_pack_members(self.pack_id)
                mean_heading = np.mean([w.heading for w in pack_members], axis=0)
                mean_heading = self._normalise(mean_heading) 

                # Cohesion: Pack centroid 
                centroid = np.mean([w.pos for w in pack_members], axis=0)
                centroid_heading = self.model.space.get_heading(self.pos, centroid)
                centroid_heading = self._normalise(centroid_heading)

                # Separation: Steer away from pack members that are too close
                separation_heading = np.array([0.0, 0.0])
                for w in pack_members:
                    if w is self:
                        continue  # Skip self
                    
                    dist = self.model.space.get_distance(self.pos, w.pos)
                    
                    if dist < self.separation_radius and dist > 0:
                        # Vector pointing away from neighbour
                        # Scaled inversely by distance: closer = stronger repulsion
                        away = self.model.space.get_heading(w.pos, self.pos)
                        away = self._normalise(away)
                        separation_heading += away / dist  # weight by inverse distance
                    separation_heading = self._normalise(separation_heading)
                
                # Combine
                pack_heading = (
                    self.alignment_weight * mean_heading +
                    self.cohesion_weight * centroid_heading +
                    self.separation_weight * separation_heading
                )
                # Normalise
                pack_heading = self._normalise(pack_heading)
            
            elif len(wolf_neighbours) > 0: 
                # If another wolf in radius then follow the heading (for the first 6 wolves in the radius)
                pack_headings = [] 
                for w in wolf_neighbours:
                    if len(pack_headings) < 6:
                        p_heading = w.heading.copy()
                        # Add some angular noise for stochasticity
                        p_heading = self._add_angular_noise(p_heading, max_angle=np.pi / 6)
                        p_heading = self._normalise(p_heading)
                        pack_headings.append(p_heading)
                pack_heading = np.mean(pack_headings, axis=0)

        
            if len(deer_neighbours) > 0:
                # If prey in sensing radius then move towards closest
                close_deer = self.ret_closest_neighbour(deer_neighbours)
                # Get heading for following Deer
                hunt_heading = self.model.space.get_heading(self.pos, close_deer.pos)
                hunt_heading = self._normalise(hunt_heading)


            # Combine heading influences for a final movement direction
            # If all headings are zero, move along original heading with some noise
            if len(wolf_neighbours) + len(deer_neighbours) == 0:
                new_heading = self._add_angular_noise(self.heading)

            elif len(deer_neighbours) == 0:
                new_heading = (self.pack_follow_weight * pack_heading)
            elif len(wolf_neighbours) == 0 and not self.model.use_pack_dynamics:
                new_heading = (self.follow_prey_weight * hunt_heading)
            else:    
                # Use weighted sum to combine
                new_heading = (self.pack_follow_weight * pack_heading) + (self.follow_prey_weight * hunt_heading)

            new_heading = self._normalise(new_heading)
            self.heading = new_heading

            # Move the agent
            new_pos = self.pos + (self.heading * self.roaming_speed)
            self.model.space.move_agent(self, new_pos)
    

    def move_random(self, speed):

        """
        Move according to a random walk.
        """
        # Set a random heading
        self.heading += np.random.random(2) * 2 - 1
        self.heading /= np.linalg.norm(self.heading)

        # Move the agent
        new_pos =self.pos + (self.heading * speed)
        self.model.space.move_agent(self, new_pos)

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
            Energy is increased to full for the wolf and its pack members (if there are any)
        '''
        # Refill energy of wolf 
        self.energy = 1.0   

        # If using pack dynamics then the other pack members also feed 
        if self.model.use_pack_dynamics:
            pack_members = self.model.get_pack_members(self.pack_id)
            for member in pack_members:
                member.energy = 1.0




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
            baby = Wolf(self.model, heading=baby_heading, pack_id=self.pack_id)
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
