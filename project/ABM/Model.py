import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from .DeerClass import Deer
from .LynxClass import Lynx
from .WolfClass import Wolf


class SpeciesModel(Model):
    "Model class for the Zombie World model"

    def __init__(
            self, 
            initial_num_wolves=10,
            initial_num_lynx=5,
            initial_num_deer = 1000,
            height=100,     
            width=100,
            seed=None
        ):
        super().__init__(seed=seed)
    
        # Model-specific parameters
        self.height = height
        self.width = width
        self.initial_num_wolves = initial_num_wolves
        self.initial_num_lynx = initial_num_lynx
        self.initial_num_deer = initial_num_deer

        # Intialise continous space, looping boundaries
        self.space = ContinuousSpace(self.width, self.height, torus=True)

        # Create and place agents
        self.make_agents()

        # Create data collector
        self.datacollector = DataCollector(
            model_reporters = {
            "Wolves": lambda m: len(m.agents_by_type[Wolf]),
            "Lynx": lambda m: len(m.agents_by_type[Lynx]),
            "Deer": lambda m: len(m.agents_by_type[Deer]),
            }
        )

        self.running = True
        self.datacollector.collect(self)


    def random_position(self):
            """Generate a random position within the space."""
            x = self.rng.random() * self.space.x_max
            y = self.rng.random() * self.space.y_max
            return np.array((x, y))
        
    def random_heading(self):
        # Random initial heading
        heading = self.rng.random(2) * 2 - 1  # Random vector between -1 and 1
        heading /= np.linalg.norm(heading)
        return heading
    
    def make_agents(self):

        """Create and place all agents randomly in the space."""

        self.deer = []
        self.lynx = []
        self.wolves = []

        # Deer (change based of deer density)
        for _ in range(self.initial_num_deer):
            pos = self.random_position()
            heading = self.random_heading()
            deer = self.create_agent(Deer, heading=heading)
            self.space.place_agent(deer, pos)

        # change based on lynx/ wolf release strategy
        # change for species specific parameters (e.g. energy, speed, etc.)

        # Lynx
        for _ in range(self.initial_num_lynx):
            pos = self.random_position()
            heading = self.random_heading()
            lynx = self.create_agent(Lynx, heading=heading)
            self.space.place_agent(lynx, pos)

        # Wolves
        for _ in range(self.initial_num_wolves):
            pos = self.random_position()
            heading = self.random_heading()
            wolf = self.create_agent(Wolf, heading=heading)
            self.space.place_agent(wolf, pos)

    def step(self):
        """
        Run one step of the model.
        """
        # All agents step based on model schudule
        self.agents.shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)
        


