import numpy as np
from mesa import Agent


class Deer(Agent):

    def __init__(
            self, 
            model,
            heading,
            speed = 8000,  # 8km/h (to be changed)
            sensing_radius = 10,  # (to be changed)
            reproduction_rate = 0.03,  # (to be changed)
            death_rate = 0.01,  # (to be changed)
            species = "Deer",
            # Movement weightings
            flee_weight = 1,
            food_weight = 1
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
        self.food_weight = food_weight
        
        # added lifespan counter
        self.age = 0
        self.sex = self.model.rng.choice(["M", "F"]) 
        self.species = species


    def step(self):

        # with each step age increase, energy decreases
        self.age += 1

        # Move
        if self.model.use_random_movement:
            self.move_random()
        else:
            self.move()  # More complex movement 
        
        # graze in that grid cell
        self.model.graze_vegetation(self.pos)

        # reproduce
        self.maybe_reproduce()

        # die
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


    def move(self):
        # Get all neighbours within sensing radius
        all_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)]
        # deer_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Deer']
        wolf_neighbours = [n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True) if n.species == 'Wolf']
        
        # Initialise lists to store neighbour headings
        flee_headings = [] if len(wolf_neighbours) > 0 else False
        food_headings = False # TO BE ADDED

        # If wolf in radius then flee 
        for w in wolf_neighbours:
            # get heading for following Deer
            f_heading =  -self.model.space.get_heading(self.pos, w.pos)
            f_heading /= np.linalg.norm(f_heading)
            flee_headings.append(f_heading)
        flee_heading = np.mean(flee_headings, axis=0) if flee_headings else np.array([0.0,0.0])
        

        # If food in sensing radius then move towards
        # TO BE ADDED
        
        # Combine heading influences for a final movement direction
        # If all headings are zero, move randomly
        if not all_neighbours:
            self.move_random() 

        else:


            # Use weighted sum to combine
            new_heading = (self.flee_weight * flee_heading) #+ (self.follow_food_weight * food_heading)
            norm = np.linalg.norm(new_heading)
            if norm > 0:
                new_heading /= norm
            self.heading = new_heading

            # Move the agent
            self.pos += self.heading * self.speed
            self.model.space.move_agent(self, self.pos)

    
    def maybe_reproduce(self):

        # For simplicity, we can use a fixed reproduction rate, but this could be expanded to include factors like age, energy, presence of mates, etc.
        if self.model.rng.random() < self.reproduction_rate:

            baby_heading = self.model.random_heading()
            baby = Deer(self.model, heading=baby_heading)
            self.model.space.place_agent(baby, self.pos)

    def maybe_die(self):

        # For simplicity, we can use a fixed death rate, but this could be expanded to include factors like age, predation risk, etc.
        if self.model.rng.random() < self.death_rate:
            self.remove()


    
