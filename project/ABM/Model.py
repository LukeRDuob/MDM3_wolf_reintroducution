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
            max_steps = 400000,
            data_collection_period = 1, # collect data every 1 step 
            init_predators = 20,  # approx 4 wolves per km^2 (400 wolves)
            init_deer = 1000,  # approx 10 deer per km^2 (1000 deer)
            height=10,     
            width=10,
            step_size = 0.25, # 15 min per step
            # yearly_sunlight_hours = 8760,
            yearly_sunlight_hours = 5000,  # represents the fact that the agents are not active for all hours of the year (e.g. not active at night, less active in winter, etc.)

            seed=None,
            predator = 'Wolf',  # Helper attribute to avoid imports when accessing agent type
            energy_decrease = 0.002,  # Energy decrease parameter 
            pack_limit = 12,  # packs will split if too large 

            # Vegetation Parameters
            veg_patch_spacing=4,
            sapling_density=20,
            tree_density=8,
            sapling_regrowth_prob=1/10, # every 10 steps (10 hours) new sapling grows
            sapling_maturation_prob=1/20000, # sapling becomes a tree every 20000 hours (2.3 years)

            # Options to control complexity of the model
            use_base = False, 
            use_pack_dynamics = True,  
            use_random_movement = False,
            use_veg = True,
            given_positions = False, # whether to use random positions or pre-chosen positions (for testing purposes)
            use_boundary_conditions = True, # whether to use boundary conditions (reflecting off walls) or toroidal space
        ):
        super().__init__(seed=seed)
    

        # Model-specific parameters
        self.data_collection_period = data_collection_period
        self.max_steps = max_steps
        self.height = height
        self.width = width
        self.step_size = step_size
        self.initial_num_pred = init_predators
        self.initial_num_deer = init_deer
        self.predator = predator 
        self.use_pack_dynamics = not use_base and use_pack_dynamics  # Only use pack dynamics if not using base model
        self.use_random_movement = use_random_movement
        self.use_veg = use_veg
        self.use_base = use_base
        self.use_boundary_conditions = use_boundary_conditions
        
        self.num_of_packs = max(self.initial_num_pred // 6, 2) # 6 wolves per pack (minimum 2 packs)
        self.pack_limit = pack_limit
        # Number of hours each year
        self.yearly_sunlight_hours = yearly_sunlight_hours / self.step_size
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
            pass
            self.veg_patch_spacing = veg_patch_spacing
            self.sapling_density = sapling_density
            self.tree_density = tree_density
            self.sapling_regrowth_prob = sapling_regrowth_prob
            self.sapling_maturation_prob = sapling_maturation_prob

            model_reporters = {
            'Time': lambda m: m.steps * m.step_size,
            self.predator: lambda m: len(m.agents_by_type.get(pred_obj, [])),
            "Deer": lambda m: len(m.agents_by_type.get(Deer, [])),
            "Wolf Population Normalised": lambda m: len(m.agents_by_type.get(Wolf, [])) / (self.initial_num_pred),
            "Deer Population Normalised": lambda m: len(m.agents_by_type.get(Deer, [])) / (self.initial_num_deer),
            "Total Saplings": lambda m: sum(v.saplings for v in m.agents_by_type.get(Vegetation, [])),
            "Total Trees": lambda m: sum(v.trees for v in m.agents_by_type.get(Vegetation, [])),
            "Deer Hunted": lambda m: m.hunted_deer,
            "Total Deer Deaths": lambda m: m.deer_deaths,
            "Total Wolf Deaths": lambda m: m.wolf_deaths,
            "Total Wolf Energy": lambda m: sum([w.energy for w in m.agents_by_type.get(Wolf,[])]),
            "Mean Wolf Energy": lambda m: (sum([w.energy for w in m.agents_by_type.get(Wolf,[])]))/(len(m.agents_by_type.get(pred_obj, [])))  ,
            "Number of Packs": lambda m: m.num_of_packs,
            "Mean Pack Size": lambda m: m.get_mean_pack_size(),
            }
        else: 
            model_reporters = {
            'Time': lambda m: m.steps * m.step_size,
            self.predator: lambda m: len(m.agents_by_type[pred_obj]),
            "Deer": lambda m: len(m.agents_by_type[Deer]),
            # "Wolf Population Normalised": lambda m: len(m.agents_by_type.get(Wolf, [])) / (self.initial_num_pred),
            # "Deer Population Normalised": lambda m: len(m.agents_by_type.get(Deer, [])) / (self.initial_num_deer),
            "Deer Hunted": lambda m: m.hunted_deer,
            "Total Deer Deaths": lambda m: m.deer_deaths,
            "Total Wolf Deaths": lambda m: m.wolf_deaths,
            # "Total Wolf Energy": lambda m: sum([w.energy for w in m.agents_by_type.get(Wolf,[])]),
            # "Mean Wolf Energy": lambda m: (sum([w.energy for w in m.agents_by_type.get(Wolf,[])]))/(len(m.agents_by_type.get(pred_obj, [])))  ,

            }


        # Intialise continous space, looping boundaries
        self.space = ContinuousSpace(self.width, self.height, torus=not self.use_boundary_conditions) 

        # Get elevation grid
        # self.add_elevation_map()

        # Create and place agents
        if given_positions:
            self.place_agents(given_positions)
        else:
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
        # Random starting ages between 0 and 10 years (in hours)
        starting_ages = self.rng.uniform(0, 10, self.initial_num_deer) 
        deer_agents = Deer.create_agents(self, self.initial_num_deer, heading= deer_headings, age=starting_ages)

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
            self.pack_ids = self.rng.integers(1, self.num_of_packs + 1, self.initial_num_pred)
            # Random starting ages between 0 and 10 years (in hours)
            starting_ages = self.rng.uniform(0, 10, self.initial_num_pred) 
            wolf_agents = Wolf.create_agents(self, self.initial_num_pred, heading= pred_headings, age=starting_ages, pack_id= self.pack_ids)
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
               
    def place_agents(self,positions):

        """Create and place all agents randomly in the space."""

        deer_headings = [self.random_heading() for _ in range(self.initial_num_deer)] 
        # Random starting ages between 0 and 10 years 
        starting_ages = self.rng.uniform(0, 10, self.initial_num_deer) 
        deer_agents = Deer.create_agents(self, self.initial_num_deer, heading= deer_headings, age=starting_ages)
        
        for agent in deer_agents:
            position = positions['Deer'].pop()
            self.space.place_agent(agent, position)


        # change based on lynx/ wolf release strategy
        # change for species specific parameters (e.g. energy, speed, etc.)

        pred_headings = [self.random_heading() for _ in range(self.initial_num_pred)]



        if self.predator == "Wolf":
            # Generate packs
            self.pack_ids = self.rng.integers(1, self.num_of_packs + 1, self.initial_num_pred)
            # Random starting ages between 0 and 10 years (in hours)
            starting_ages = self.rng.uniform(0, 10*365*24, self.initial_num_pred) 
            wolf_agents = Wolf.create_agents(self, self.initial_num_pred, heading= pred_headings, age=starting_ages, pack_id= self.pack_ids)
            for agent in wolf_agents:
                position = positions['Wolf'].pop()
                self.space.place_agent(agent, position)

               

    def clip_and_reflect(self, pos, heading):
        """
        Clips position to space bounds and reflects heading off walls.
        Returns (new_pos, new_heading).
        """
        x, y = pos
        hx, hy = heading

        # Reflect off left/right walls
        if x <= 0:
            x = abs(x)
            hx = abs(hx)
        elif x >= self.space.x_max:
            x = 2 * self.space.x_max - x
            hx = -abs(hx)

        # Reflect off top/bottom walls
        if y <= 0:
            y = abs(y)
            hy = abs(hy)
        elif y >= self.space.y_max:
            y = 2 * self.space.y_max - y
            hy = -abs(hy)

        # Safety clamp to stay strictly within bounds
        x = np.clip(x, 0.001, self.space.x_max - 0.001)
        y = np.clip(y, 0.001, self.space.y_max - 0.001)

        return np.array([x, y]), np.array([hx, hy])   


    # def get_pack_members(self, pack_id):
    #     ''' 
    #         Gets all the pack members for a given pack_id 
    #     '''
    #     pack = []
    #     for w in self.agents_by_type[Wolf]:
    #         if w.pack_id == pack_id:
    #             pack.append(w)
    #     return pack        
    
    def get_pack_members(self, pack_id):
        '''Uses cached registry for O(1) lookup'''
        return list(self.pack_registry.get(pack_id, []))

    def get_mean_pack_size(self):
        """
            Helper method for model reporters
        """
        sizes = []
        for id in range(1, self.num_of_packs + 1):
            size = len(self.get_pack_members(id))
            sizes.append(size)
        return sum(sizes)/self.num_of_packs
        


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
            removed_members = self.rng.choice(members, size=half_size, replace=False)

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



    
    def step(self):
        """
        Run one step of the model.
        """
        # Rebuild pack registry once per step
        self.pack_registry = {}
        for w in self.agents_by_type[Wolf]:
            self.pack_registry.setdefault(w.pack_id, []).append(w)
    
        # All agents step based on model schudule
        self.agents.shuffle_do("step")

        # check if any packs need to be split
        if not self.use_base and self.use_pack_dynamics:
            for id in range(1, self.num_of_packs + 1):
                self.maybe_split_pack(id)

        # Collect data every certain number of steps 
        if self.steps % self.data_collection_period == 0:
            self.datacollector.collect(self)    

        # Stop after max steps
        if self.steps >= self.max_steps:
            self.running = False
        # Stop if deer or wolves are extinct
        if len(self.agents_by_type[Deer]) == 0:
            self.running = False
        if len(self.agents_by_type[Wolf]) == 0:
            self.running = False