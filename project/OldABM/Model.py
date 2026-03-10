import numpy as np

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from DeerClass import Deer
from LynxClass import Lynx
from WolfClass import Wolf



class SpeciesModel(Model):
    "Model class for the model"

    def __init__(
            self, 
            init_predators=1,
            init_deer = 10,
            height=100,     
            width=100,
            seed=None,
            predator = 'Wolf',  # Helper attribute to avoid imports when accessing agent type
            energy_decrease = 0.05,  # Energy decrease parameter 
            energy_min = 0  # Point at which the animal will die of exhaustion
        ):
        super().__init__(seed=seed)
    

        # Model-specific parameters
        self.height = height
        self.width = width
        self.initial_num_pred = init_predators
        self.initial_num_deer = init_deer
        self.predator = predator
        self.energy_decrease = energy_decrease
        self.energy_min = energy_min
        # Intialise continous space, looping boundaries
        self.space = ContinuousSpace(self.width, self.height, torus=True)

        # Create and place agents
        self.make_agents()

        # Create data collector
        if self.predator == "Lynx":
            pred_obj = Lynx
        elif self.predator == "Wolf":
            pred_obj = Wolf
            
        self.datacollector = DataCollector(
            model_reporters = {
            self.predator: lambda m: len(m.agents_by_type[pred_obj]),
            "Deer": lambda m: len(m.agents_by_type[Deer]),
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def random_position(self):
        
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

        deer_headings = [self.random_heading() for _ in range(self.initial_num_deer)] 
        deer_agents = Deer.create_agents(self, self.initial_num_deer, heading= deer_headings)

        for agent in deer_agents:
            self.space.place_agent(agent, self.random_position())


        # change based on lynx/ wolf release strategy
        # change for species specific parameters (e.g. energy, speed, etc.)

        pred_headings = [self.random_heading() for _ in range(self.initial_num_pred)]

        # placing predators
        if self.predator == "Lynx":

            lynx_agents = Lynx.create_agents(self, self.initial_num_pred, heading= pred_headings)
            for agent in lynx_agents:
                self.space.place_agent(agent, self.random_position())

        elif self.predator == "Wolf":

            wolf_agents = Wolf.create_agents(self, self.initial_num_pred, heading= pred_headings)
            for agent in wolf_agents:
                self.space.place_agent(agent, self.random_position())

    def step(self):
        """
        Run one step of the model.
        """
        print(self.steps)

        # All agents step based on model schudule
        self.agents.shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)

        # Stop after max steps
        if self.steps >= 100:
            self.running = False
        
        # Stop if deer or wolves are extinct
        if len(self.agents_by_type[Deer]) == 0:
            self.running = False
        if len(self.agents_by_type[Wolf]) == 0:
            self.running = False
        




