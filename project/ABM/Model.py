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
            max_steps = 1500,
            init_predators=15,
            init_deer =90,  # approx 10 deer per km^2 (9000 deer)
            height=30,     
            width=30,
            step_size = 1, # 1 hour per step
            yearly_sunlight_hours = 8760,
            seed=None,
            predator = 'Wolf',  # Helper attribute to avoid imports when accessing agent type
            energy_decrease = 0.002,  # Energy decrease parameter 
            pack_limit = 12,  # packs will split if too large 
            #Vegetation Parameters
            
            veg_patch_spacing=4,
            sapling_density=20,
            tree_density=8,
            sapling_regrowth_prob=1/10, # every 10 steps (10 hours) new sapling grows
            sapling_maturation_prob=1/20000, # sapling becomes a tree every 20000 hours (2.3 years)

            # Options to control complexity of the model
            use_pack_dynamics = True,  
            use_random_movement = False,
            use_veg = True
        ):
        super().__init__(seed=seed)
    

        # Model-specific parameters
        self.max_steps = max_steps
        self.height = height
        self.width = width
        self.step_size = step_size
        self.initial_num_pred = init_predators
        self.initial_num_deer = init_deer
        self.predator = predator 
        self.use_pack_dynamics = use_pack_dynamics
        self.use_random_movement = use_random_movement
        self.use_veg = use_veg
        
        self.num_of_packs = max(self.initial_num_pred // 6, 2) # 6 wolves per pack (minimum 2 packs)
        self.pack_limit = pack_limit
        # Number of hours each year
        self.yearly_sunlight_hours = yearly_sunlight_hours


        # Energy
        self.energy_decrease = energy_decrease

        # Counts to show
        self.hunted_deer = 0
        self.deer_deaths = 0
        self.wolf_deaths = 0
        

        # Create data collector
        if self.predator == "Lynx":
            pred_obj = Lynx
        elif self.predator == "Wolf":
            pred_obj = Wolf

        
        if use_veg:
            # Vegetation
            
            self.veg_patch_spacing = veg_patch_spacing
            self.sapling_density = sapling_density
            self.tree_density = tree_density
            self.sapling_regrowth_prob = sapling_regrowth_prob
            self.sapling_maturation_prob = sapling_maturation_prob

            model_reporters = {
            'Time': lambda m: m.steps * m.step_size,
            self.predator: lambda m: len(m.agents_by_type.get(pred_obj, [])),
            "Deer": lambda m: len(m.agents_by_type.get(Deer, [])),
            "Total Saplings": lambda m: sum(v.saplings for v in m.agents_by_type.get(Vegetation, [])),
            "Total Trees": lambda m: sum(v.trees for v in m.agents_by_type.get(Vegetation, [])),
            "Deer Hunted": lambda m: m.hunted_deer,
            "Total Deer Deaths": lambda m: m.deer_deaths,
            "Total Wolf Deaths": lambda m: m.wolf_deaths,
            "Total Wolf Energy": lambda m: sum([w.energy for w in m.agents_by_type.get(Wolf,[])]),
            "Mean Wolf Energy": lambda m: (sum([w.energy for w in m.agents_by_type.get(Wolf,[])]))/(len(m.agents_by_type.get(pred_obj, [])))  ,

            }
        else: 
            model_reporters = {
            'Time': lambda m: m.steps * m.step_size,
            self.predator: lambda m: len(m.agents_by_type[pred_obj]),
            "Deer": lambda m: len(m.agents_by_type[Deer]),
            "Deer Hunted": lambda m: m.hunted_deer,
            "Total Deer Deaths": lambda m: m.deer_deaths,
            "Total Wolf Deaths": lambda m: m.wolf_deaths,
            "Total Wolf Energy": lambda m: sum([w.energy for w in m.agents_by_type.get(Wolf,[])]),
            "Mean Wolf Energy": lambda m: (sum([w.energy for w in m.agents_by_type.get(Wolf,[])]))/(len(m.agents_by_type.get(pred_obj, [])))  ,

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

            lynx_agents = Lynx.create_agents(self, self.initial_num_pred, heading= pred_headings, energy_decrease = self.energy_decrease)
            for agent in lynx_agents:
                self.space.place_agent(agent, self.random_position())

        elif self.predator == "Wolf":
            # Generate packs
            self.pack_ids = self.rng.integers(1, self.num_of_packs, self.initial_num_pred)
            wolf_agents = Wolf.create_agents(self, self.initial_num_pred, heading= pred_headings, pack_id= self.pack_ids)
            for agent in wolf_agents:
                self.space.place_agent(agent, self.random_position())

        if self.use_veg:
            # Create the vegetation as clusters
            positions = self.vegetation_patch_positions(
                target_spacing=self.veg_patch_spacing,
                jitter_fraction=0.4)

            for pos in positions:
                veg = Vegetation.random_patch(
                    self,
                    patch_spacing=self.veg_patch_spacing,
                    sapling_density=self.sapling_density,
                    tree_density=self.tree_density,
                    sapling_regrowth_prob=self.sapling_regrowth_prob,
                    sapling_maturation_prob=self.sapling_maturation_prob
                )
                self.space.place_agent(veg, pos)
               

    def get_pack_members(self, pack_id):
        ''' 
            Gets all the pack members for a given pack_id 
        '''
        pack = []
        for w in self.agents_by_type[Wolf]:
            if w.pack_id == pack_id:
                pack.append(w)
        return pack        

    def maybe_split_pack(self, pack_id):
        """
            Splits the pack in two if it is too large 
        """
        # Get members
        members = self.get_pack_members(pack_id)

        # Count
        count = len(members)
        if count > self.pack_limit:
            # Randomly choose half of the members 
            half_size = count // 2
            removed_members = self.rng.choice(members, size=half_size)

            # Find next uniqie pack_id 
            new_id = self.num_of_packs + 1

            # Modify pack ids for removed members
            for m in removed_members:
                m.pack_id = new_id

            self.num_of_packs += 1


    def vegetation_patch_positions(self, target_spacing=None, jitter_fraction=0.2):
        positions = []

        if target_spacing is None:
            target_spacing = self.veg_patch_spacing

        ncols = max(1, round(self.width / target_spacing))
        nrows = max(1, round(self.height / target_spacing))

        cell_width = self.width / ncols
        cell_height = self.height / nrows

        jitter_x = jitter_fraction * cell_width
        jitter_y = jitter_fraction * cell_height

        for row in range(nrows):
            for col in range(ncols):
                x = (col + 0.5) * cell_width
                y = (row + 0.5) * cell_height

                x += self.rng.uniform(-jitter_x, jitter_x)
                y += self.rng.uniform(-jitter_y, jitter_y)

                x = max(0, min(x, self.width))
                y = max(0, min(y, self.height))

                positions.append(np.array((x, y)))

        self.rng.shuffle(positions)
        return positions



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

        # Check if packs need splitting
        for id in range(1, self.num_of_packs):
            self.maybe_split_pack(id)

        # Collect data
        self.datacollector.collect(self)

        # Stop after max steps
        if self.steps >= self.max_steps:
            self.running = False
        
        # Stop if deer or wolves are extinct
        if len(self.agents_by_type[Deer]) == 0:
            self.running = False
        if len(self.agents_by_type[Wolf]) == 0:
            self.running = False