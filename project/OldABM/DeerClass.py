import numpy as np
from mesa import Agent


class Deer(Agent):

    def __init__(
            self, 
            model,
            heading,
            speed = 2,
            sensing_radius = 10,
            reporduction_rate = 0.03,
            death_rate = 0.01,
            species = "Deer"
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
        self.species = species


    def step(self):

        # with each step age increase, energy decreases
        self.age += 1

        # move
        self.move_random()

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


    
