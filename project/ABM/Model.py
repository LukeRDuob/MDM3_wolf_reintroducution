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
            init_predators=15,
            init_deer = 10,
            height=30000,     
            width=30000,
            seed=None,
            predator = 'Wolf',  # Helper attribute to avoid imports when accessing agent type
            energy_decrease = 0.05,  # Energy decrease parameter 
            energy_min = 0,  # Point at which the animal will die of exhaustion
            # Introducing vegetation
            veg_cell_size = 10
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

        # Vegetation
        self.veg_cell_size = veg_cell_size
        self.veg_width = self.width // self.veg_cell_size
        self.veg_height = self.height // self.veg_cell_size

        # Decides which cells are at which stage of growth, later, more realistic to make specific regions 
        # more likely to be specific stages, like clumps of trees etc rather than randomly dotted everywhere
        self.veg_stage = self.rng.choice(
            [0, 1, 2, 3],
            size=(self.veg_width, self.veg_height),
            p=[0.2, 0.3, 0.3, 0.2] # probability of which stages selected
        )  
        # 0 = empty, 1 = new_growth, 2 = sapling, 3 = tree.

        # Randomises the growth timer for each stage, so not all the saplings become trees at
        # the same time etc. 
        
        self.veg_timer = np.zeros((self.veg_width, self.veg_height), dtype=int)

        for x in range(self.veg_width):
            for y in range(self.veg_height):
                stage = self.veg_stage[x, y]

                if stage == 0:   # empty
                    self.veg_timer[x, y] = self.rng.integers(0, 6) # position each grid cell within in a stage at a random growth point

                elif stage == 1: # new growth
                    self.veg_timer[x, y] = self.rng.integers(0, 10)

                elif stage == 2: # sapling
                    self.veg_timer[x, y] = self.rng.integers(0, 20)

                elif stage == 3: # tree
                    self.veg_timer[x, y] = self.rng.integers(0, 30)
        
        self.veg_bites = np.zeros((self.veg_width, self.veg_height), dtype=int)

        self.browse_thresholds = {
            1: 1,  # new growth -> empty
            2: 3,  # sapling -> new growth
            3: 6   # tree -> sapling
        }



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
            "Empty": lambda m: np.sum(m.veg_stage == 0),
            "NewGrowth": lambda m: np.sum(m.veg_stage == 1),
            "Sapling": lambda m: np.sum(m.veg_stage == 2),
            "Tree": lambda m: np.sum(m.veg_stage == 3),
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

         
    # Defines the growth of the vegetation 
    def step_vegetation(self):
        for x in range(self.veg_width):
            for y in range(self.veg_height):
                self.veg_timer[x, y] += 1

                if self.veg_stage[x, y] == 0:  # empty
                    if self.veg_timer[x, y] >= 8: # how many steps must be reached before possible new growth
                        if self.rng.random() < 0.2: # adds a random element as to if smth will grow
                            self.veg_stage[x, y] = 1
                            self.veg_timer[x, y] = 0
                            self.veg_bites[x, y] = 0

                elif self.veg_stage[x, y] == 1:  # new growth
                    if self.veg_timer[x, y] >= 15:
                        self.veg_stage[x, y] = 2
                        self.veg_timer[x, y] = 0
                        self.veg_bites[x, y] = 0

                elif self.veg_stage[x, y] == 2:  # sapling
                    if self.veg_timer[x, y] >= 25:
                        self.veg_stage[x, y] = 3
                        self.veg_timer[x, y] = 0
                        self.veg_bites[x, y] = 0

    def get_veg_cell(self, pos):
        """ Returns the vegetation 'patch' the animal is in """
        x = int(pos[0] // self.veg_cell_size)
        y = int(pos[1] // self.veg_cell_size)
        x = min(max(x, 0), self.veg_width - 1)
        y = min(max(y, 0), self.veg_height - 1)

        return x, y
    
    def graze_vegetation(self, pos):
        """Tracks grazing on vegetation, so regresses to smaller stages"""
        x, y = self.get_veg_cell(pos)
        stage = self.veg_stage[x, y]

        if stage == 0:
            return

        self.veg_bites[x, y] += 1

        if self.veg_bites[x, y] >= self.browse_thresholds[stage]:
            self.veg_stage[x, y] = stage - 1
            self.veg_timer[x, y] = 0
            self.veg_bites[x, y] = 0

    def step(self):
        """
        Run one step of the model.
        """

        # All agents step based on model schudule
        self.agents.shuffle_do("step")

        # vegetation
        self.step_vegetation()

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
        




