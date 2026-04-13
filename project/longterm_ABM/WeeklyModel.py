# WeeklyModel.py

import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from DeerHerdAgent import DeerHerd
from WolfPackAgent import WolfPack
#from VegetationClass import Vegetation


class WeeklySpeciesModel(Model):
    def __init__(
        self,
        width=30,
        height=30,
        max_steps=520,
        init_deer_groups=10,
        herd_size_bounds=(6, 35),
        init_wolf_packs=3,
        pack_size_bounds = (5, 11),
        veg_patch_spacing=2.5,
        seed=None
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.space = ContinuousSpace(self.width, self.height, torus=False)

        self.weekly_deer_kills = 0
        self.total_deer_killed = 0
        self.total_saplings_eaten = 0
        self.pack_size_bounds = pack_size_bounds
        self.herd_size_bounds = herd_size_bounds
        
        self.make_deer_groups(init_deer_groups, self.herd_size_bounds)
        self.make_wolf_packs(init_wolf_packs, self.pack_size_bounds)
        #self.make_vegetation(veg_patch_spacing)

        self.datacollector = DataCollector(
            model_reporters={
                "Total Deer": lambda m: sum(d.group_size for d in m.agents_by_type.get(DeerHerd, [])),
                "Total Wolves": lambda m: sum(w.pack_size for w in m.agents_by_type.get(WolfPack, [])),
                #"Total Saplings": lambda m: sum(v.saplings for v in m.agents_by_type.get(Vegetation, [])),
                #"Total Trees": lambda m: sum(v.trees for v in m.agents_by_type.get(Vegetation, [])),
                "Weekly Deer Killed": lambda m: m.weekly_deer_kills,
                "Total Deer Killed": lambda m: m.total_deer_killed,
                "Total Saplings Eaten": lambda m: m.total_saplings_eaten,

                "Pack 1 Size": lambda m: m.get_pack_size(1),
                "Pack 2 Size": lambda m: m.get_pack_size(2),
                "Pack 3 Size": lambda m: m.get_pack_size(3),
                "Pack 4 Size": lambda m: m.get_pack_size(4),
                
                "Pack 1 Energy": lambda m: m.get_pack_energy(1),
                "Pack 2 Energy": lambda m: m.get_pack_energy(2),
                "Pack 3 Energy": lambda m: m.get_pack_energy(3),
                "Pack 4 Energy": lambda m: m.get_pack_energy(4),
            }
        )

        self.running = True
        self.datacollector.collect(self)

    def random_position(self):
        x = self.rng.random() * self.width
        y = self.rng.random() * self.height
        return np.array([x, y])

    def random_heading(self):
        heading = self.rng.random(2) * 2 - 1
        heading /= np.linalg.norm(heading)
        return heading

    def clip_and_reflect(self, pos, heading):
        x, y = pos
        hx, hy = heading

        if x <= 0:
            x = abs(x)
            hx = abs(hx)
        elif x >= self.space.x_max:
            x = 2 * self.space.x_max - x
            hx = -abs(hx)

        if y <= 0:
            y = abs(y)
            hy = abs(hy)
        elif y >= self.space.y_max:
            y = 2 * self.space.y_max - y
            hy = -abs(hy)

        x = np.clip(x, 0.001, self.space.x_max - 0.001)
        y = np.clip(y, 0.001, self.space.y_max - 0.001)

        return np.array([x, y]), np.array([hx, hy])

    def make_deer_groups(self, n_groups, herd_size_bounds):
        headings = [self.random_heading() for _ in range(n_groups)]

        herd_sizes = [
            self.rng.integers(herd_size_bounds[0], herd_size_bounds[1] + 1)
            for _ in range(n_groups)
        ]

        deer_groups = DeerHerd.create_agents(
            self,
            n_groups,
            heading=headings,
            group_size=herd_sizes
        )

        for group in deer_groups:
            self.space.place_agent(group, self.random_position())

    def make_wolf_packs(self, n_packs, pack_size_bounds):
        headings = [self.random_heading() for _ in range(n_packs)]

        pack_sizes = [
            self.rng.integers(pack_size_bounds[0], pack_size_bounds[1] + 1)
            for _ in range(n_packs)
        ]

        pack_ids = list(range(1, n_packs + 1))

        packs = WolfPack.create_agents(
            self,
            n_packs,
            heading=headings,
            pack_size=pack_sizes,
            pack_id = pack_ids,
        )

        for p in packs:
            self.space.place_agent(p, self.random_position())
    '''
    def make_vegetation(self, patch_spacing):
        ncols = max(1, round(self.width / patch_spacing))
        nrows = max(1, round(self.height / patch_spacing))

        for row in range(nrows):
            for col in range(ncols):
                x = (col + 0.5) * self.width / ncols
                y = (row + 0.5) * self.height / nrows

                veg = Vegetation.random_patch(
                    self,
                    patch_spacing=patch_spacing,
                    sapling_density=5,
                    tree_density=2,
                    sapling_regrowth_prob=0.2,
                    sapling_maturation_prob=0.02
                )
                self.space.place_agent(veg, np.array([x, y]))
    '''

    def get_pack_size(self, pack_id):
        for pack in self.agents_by_type.get(WolfPack, []):
            if pack.pack_id == pack_id:
                return pack.pack_size
        return 0
    
    def get_next_pack_id(self):
        existing_ids = [
            w.pack_id for w in self.agents_by_type.get(WolfPack, [])
            if w.pack_id is not None
        ]
        return max(existing_ids, default=0) + 1
    
    def get_pack_energy(self, pack_id):
        for pack in self.agents_by_type.get(WolfPack, []):
            if pack.pack_id == pack_id:
                return pack.energy
        return 0

    def step(self):
        self.weekly_deer_kills = 0
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        if self.steps >= self.max_steps:
            self.running = False