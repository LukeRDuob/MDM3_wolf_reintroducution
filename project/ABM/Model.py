import numpy as np
import pandas as pd

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.space import ContinuousSpace

from DeerClass import Deer
from LynxClass import Lynx
from WolfClass import Wolf
from VegetationClass import Vegetation


class SpeciesModel(Model):
    "Model class"

    def __init__(
            self,  
            init_predators=15,
            init_deer = 100,
            height=30000,     
            width=30000,
            seed=None,
            init_num_of_packs = 3,
            predator = 'Wolf',  # Helper attribute to avoid imports when accessing agent type
            energy_decrease = 0.05,  # Energy decrease parameter 
            energy_min = 0,  # Point at which the animal will die of exhaustion

            init_veg=10,  # Introducing vegetation
            sapling_growth_time=100, #change
            veg_regrowth_prob=0.1, #change
            # Options to control complexity of the model
            use_pack_dynamics = True,  
            use_random_movement = False,
            use_veg = True
        ):
        super().__init__(seed=seed)
    

        # Model-specific parameters
        self.height = height
        self.width = width
        self.initial_num_pred = init_predators
        self.initial_num_deer = init_deer
        self.predator = predator 
        self.use_pack_dynamics = use_pack_dynamics
        self.use_random_movement = use_random_movement
        self.use_veg = use_veg
        self.init_num_of_packs = init_num_of_packs

        # Energy
        self.energy_decrease = energy_decrease
        self.energy_min = energy_min

        # Create data collector
        if self.predator == "Lynx":
            pred_obj = Lynx
        elif self.predator == "Wolf":
            pred_obj = Wolf

        
        if use_veg:
            # Vegetation
            self.init_veg = init_veg
            self.sapling_growth_time = sapling_growth_time
            self.veg_regrowth_prob = veg_regrowth_prob

            model_reporters = {
            self.predator: lambda m: len(m.agents_by_type.get(pred_obj, [])),
            "Deer": lambda m: len(m.agents_by_type.get(Deer, [])),
            "Sapling": lambda m: sum(1 for v in m.agents_by_type.get(Vegetation, []) if v.stage == "sapling"),
            "Tree": lambda m: sum(1 for v in m.agents_by_type.get(Vegetation, []) if v.stage == "tree")
            }
        else: 
            model_reporters = {
            self.predator: lambda m: len(m.agents_by_type[pred_obj]),
            "Deer": lambda m: len(m.agents_by_type[Deer]),
            }


        # Intialise continous space, looping boundaries
        self.space = ContinuousSpace(self.width, self.height, torus=True) 

        # Get elevation grid
        # self.add_elevation_map()

        # Create and place agents
        self.make_agents()
            

        self.datacollector = DataCollector(
            model_reporters = model_reporters
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
            # Generate packs
            pack_ids = self.rng.integers(1, self.init_num_of_packs + 1, self.initial_num_pred)
            wolf_agents = Wolf.create_agents(self, self.initial_num_pred, heading= pred_headings, pack_id= pack_ids)
            for agent in wolf_agents:
                self.space.place_agent(agent, self.random_position())

        if self.use_veg:
            # Create the vegetation as individual points
            for _ in range(self.init_veg):
                stage = self.rng.choice(["sapling", "tree"], p=[0.6, 0.4])
                veg = Vegetation(self, stage=stage, growth_time=self.sapling_growth_time)

                if stage == "sapling":
                    veg.age = self.rng.integers(0, self.sapling_growth_time // 2)
                else:
                    veg.age = self.rng.integers(self.sapling_growth_time, self.sapling_growth_time + 20) # change depending on time frame
            
                self.space.place_agent(veg, self.random_position())


    
    def regrow_vegetation(self):
        #if vegetation is on, allows new saplings to appear randomly
        if self.rng.random() < self.veg_regrowth_prob:
            veg = Vegetation(self, stage="sapling", growth_time=self.sapling_growth_time)
            self.space.place_agent(veg, self.random_position())
    
    

    def get_pack_members(self, pack_id):
        ''' 
            Gets all the pack members for a given pack_id 
        '''
        pack = []
        for w in self.agents_by_type[Wolf]:
            if w.pack_id == pack_id:
                pack.append(w)
        return pack        


    def add_elevation_map(self, location='glen_affric'):
        df = pd.read_csv(rf'project\data\clean_data\{location}_elevation.csv')
        grid = df.pivot(
            index='northing(y)',     # rows (y)
            columns='easting(x)',    # cols (x)
            values='elevation(z)'    # values (z)
        )

        # Store coordinate axes
        self.elev_xs = grid.columns.values
        self.elev_ys = grid.index.values

        # resolution (assumes regular grid)
        self.elev_dx = self.elev_xs[1] - self.elev_xs[0]
        self.elev_dy = self.elev_ys[1] - self.elev_ys[0]

        # origin (lower-left corner)
        self.elev_xmin = self.elev_xs.min()
        self.elev_ymin = self.elev_ys.min()

        # Convert to numpy array
        self.elevation_grid = grid.values
    
    
    def get_elevation(self, pos):
        """Return elevation at a continuous position"""

        # Get coords
        x_real, y_real = pos

        # Scale model -> real coords
        # x_real = self.elev_xmin + (pos[0] / self.width) * (self.elev_xs.max() - self.elev_xmin)
        # y_real = self.elev_ymin + (pos[1] / self.height) * (self.elev_ys.max() - self.elev_ymin)

        # Convert to grid indices
        i = int((y_real - self.elev_ymin) // self.elev_dy)
        j = int((x_real - self.elev_xmin) // self.elev_dx)

        # Clamp to bounds
        i = max(0, min(i, self.elevation_grid.shape[0] - 1))
        j = max(0, min(j, self.elevation_grid.shape[1] - 1))

        return self.elevation_grid[i, j]
    
    
    def step(self):
        """
        Run one step of the model.
        """

        # All agents step based on model schudule
        self.agents.shuffle_do("step")

        if self.use_veg:
            self.regrow_vegetation()

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
        







