"""
Class structure for Wolf objects that will be used in the ABM.

It is probably more efficient to have one standard predator class that both 
Lynx and Wolf can inherit from (but this shouldn't effect the results). 

"""
import numpy as np
import mesa


# Simpler agent inherited from Deer
class Lynx(mesa.Agent):

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
            species = "Lynx",
            energy_increase = 1,
            energy_decrease = 0.001,
            starting_energy_bounds = [0.8,1],  # Assuming energy is in the range [0,1] 
            attack_radius = 5,  # radius within which wolves can attack deer 
            # Weights for deciding which direction to move  
            flee_weight = 1,  
            follow_prey_weight = 1

        ):
    
        super().__init__(model)

        # General agent attributes
        self.heading = heading
        self.sensing_radius = sensing_radius
        self.reproduction_rate = reproduction_rate * self.model.step_size 
        self.death_rate = death_rate * self.model.step_size
        self.energy_increase = energy_increase 
        self.energy = self.model.rng.uniform(starting_energy_bounds[0], starting_energy_bounds[1])
        self.energy_decrease = energy_decrease * self.model.step_size
        self.species = species
        self.kill_prob = kill_prob
        self.speed = speed * self.model.step_size  
        self.lynx_attack_radius = attack_radius
        self.follow_prey_weight = follow_prey_weight
        self.flee_weight = flee_weight


    def step(self):
        
        # Guard: if agent has been removed, skip step
        if self.pos is None:
            return

        # Move
        # self.move_random()
        self.move()  # More complex movement that hasn't been tested
        
        # Hunt
        self.hunt()

        # Reproduce
        self.maybe_reproduce()

        # Energy decreases
        self.lose_energy()

        # Die
        self.maybe_die()
        


    def move(self):
        # Get all neighbours within sensing radius
        all_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)]
        deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Deer']
        lynx_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Lynx']
        
        self.flee_headings = [] if len(deer_neighbours) > 0 else False
        self.hunt_headings = [] if len(lynx_neighbours) > 0 else False

        # If a lynx in sensing radius then move in an opposing direction 
        for l in lynx_neighbours:
            self.f_heading = -self.model.space.get_heading(self.pos, l.pos)
            self.f_heading /= np.linalg.norm(self.f_heading)
            self.flee_headings.append(self.f_heading)

        # If prey in sensing radius then move towards
        for d in deer_neighbours:
            # get heading for following Deer
            self.h_heading = self.model.space.get_heading(self.pos, d.pos)
            self.h_heading /= np.linalg.norm(self.h_heading)
            self.hunt_headings.append(self.h_heading)


        # Combine heading influences for a final movement direction
        # If all headings are zero, move randomly
        if not all_neighbours:
            self.move_random()

        else:
            self.flee_heading = np.mean(self.flee_headings, axis=0)
            self.hunt_heading = np.mean(self.hunt_headings, axis=0)

            # Use weighted sum to combine
            self.new_heading = (self.flee_weight * self.flee_heading) + (self.follow_prey_weight * self.hunt_heading)
            norm = np.linalg.norm(self.new_heading)
            if norm > 0:
                self.new_heading /= norm
            self.heading = self.new_heading

            # Move the agent
            self.pos += self.heading * self.speed
            self.model.space.move_agent(self, self.pos)

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
            This method will be modified when pack dynamics are added to the model 
        '''
        # predators hunt deer in their current position or within killing radius
        
        # Get all agents in the wolf's killing neigbourhood (circular neighbourhood with attack radius)
        deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.wolf_attack_radius, True) if n.species=="Deer"]
        
        # Try to kill the deer if found
        if len(deer_neighbours) > 0:
            # If deer neighbours nearby then attack the closest
            other = self.ret_closest_neighbour(deer_neighbours)
            if other is not None:
                kill_chance = self.model.rng.uniform(0,1)
                if kill_chance < self.kill_prob:
                    # Feed
                    self.feed()
                    # Remove deer
                    self.model.spatial_hash.remove(other)
                    self.model.space.remove_agent(other)
                    other.remove()


                
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
        self.energy -= self.energy_decrease
               

    def maybe_reproduce(self):


        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_heading = self.model.random_heading()
            baby = Lynx(self.model, heading=baby_heading)
            self.model.space.place_agent(baby, self.pos)
            self.model.spatial_hash.add(baby)


    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate:
            self.model.spatial_hash.remove(self)
            self.model.space.remove_agent(self)
            self.remove()

        # Also remove agent if energy at minimum energy
        elif self.energy == self.model.energy_min:
            self.model.spatial_hash.remove(self)
            self.model.space.remove_agent(self)
            self.remove()


    def ret_closest_neighbour(self, neighbours):
        """
        Returns the closest neighbour from a given set of neighbours
        """
        # Filter out any removed agents (pos = None)
        valid_neighbours = [n for n in neighbours if n.pos is not None]
        if not valid_neighbours:
            return None
        neighbours_distances = np.array([[n, self.model.space.get_distance(self.pos, n.pos)] for n in valid_neighbours])
        return neighbours_distances[neighbours_distances[:,1].argsort()][0][0]


