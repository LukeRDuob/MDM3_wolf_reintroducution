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
        max_steps=1040,
        init_total_deer=11835,
        herd_size_bounds=(6, 35),
        init_wolf_packs=5,
        pack_size_bounds = (5, 11),

        seed=None
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.space = ContinuousSpace(self.width, self.height, torus=False)

        self.weekly_deer_kills = 0
        self.total_deer_killed = 0

        self.max_wolf_growth_per_year = 0.2
        self.year_start_wolves = None
        self.pack_size_bounds = pack_size_bounds
        self.herd_size_bounds = herd_size_bounds
        
        self.make_deer_groups(init_total_deer, self.herd_size_bounds)
        self.make_wolf_packs(init_wolf_packs, self.pack_size_bounds)

        self.year_start_wolves = sum(w.pack_size for w in self.agents_by_type.get(WolfPack, []))
        #self.make_vegetation(veg_patch_spacing)

        self.datacollector = DataCollector(
            model_reporters={
                "Total Deer": lambda m: sum(d.group_size for d in m.agents_by_type.get(DeerHerd, [])),
                "Total Wolves": lambda m: sum(w.pack_size for w in m.agents_by_type.get(WolfPack, [])),
                #"Total Saplings": lambda m: sum(v.saplings for v in m.agents_by_type.get(Vegetation, [])),
                #"Total Trees": lambda m: sum(v.trees for v in m.agents_by_type.get(Vegetation, [])),
                "Weekly Deer Killed": lambda m: m.weekly_deer_kills,
                "Total Deer Killed": lambda m: m.total_deer_killed,
                #"Total Saplings Eaten": lambda m: m.total_saplings_eaten,

                "Pack 1 Size": lambda m: m.get_pack_size(1),
                "Pack 2 Size": lambda m: m.get_pack_size(2),
                "Pack 3 Size": lambda m: m.get_pack_size(3),
                "Pack 4 Size": lambda m: m.get_pack_size(4),
                "Pack 5 Size": lambda m: m.get_pack_size(5),
                "Pack 6 Size": lambda m: m.get_pack_size(6),
                "Pack 7 Size": lambda m: m.get_pack_size(7),

                "Pack 1 Energy": lambda m: m.get_pack_energy(1),
                "Pack 2 Energy": lambda m: m.get_pack_energy(2),
                "Pack 3 Energy": lambda m: m.get_pack_energy(3),
                "Pack 4 Energy": lambda m: m.get_pack_energy(4),
                "Pack 5 Energy": lambda m: m.get_pack_energy(5),
                "Pack 6 Energy": lambda m: m.get_pack_energy(6),
                "Pack 7 Energy": lambda m: m.get_pack_energy(7),
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
    
    def generate_herd_sizes(self, total_deer, herd_size_bounds):
        min_size, max_size = herd_size_bounds

        if total_deer < min_size:
            raise ValueError(
                f"Total deer ({total_deer}) is too small for minimum herd size {min_size}."
            )

        herd_sizes = []
        deer_remaining = total_deer

        while deer_remaining > 0:
            # If remaining deer already form one valid herd, finish
            if min_size <= deer_remaining <= max_size:
                herd_sizes.append(deer_remaining)
                break

            # Choose a herd size that leaves a valid remainder
            possible_sizes = []
            for size in range(min_size, max_size + 1):
                remainder = deer_remaining - size

                if remainder == 0 or remainder >= min_size:
                    possible_sizes.append(size)

            if not possible_sizes:
                raise ValueError(
                    f"Could not split {total_deer} deer into herds within bounds {herd_size_bounds}."
                )

            chosen_size = int(self.rng.choice(possible_sizes))
            herd_sizes.append(chosen_size)
            deer_remaining -= chosen_size

        self.rng.shuffle(herd_sizes)
        return herd_sizes
    
    def enforce_wolf_growth_cap(self):
        wolves = self.agents_by_type.get(WolfPack, [])
        total_wolves = sum(w.pack_size for w in wolves)

        if self.year_start_wolves is None:
            self.year_start_wolves = total_wolves
            return

        max_allowed = int(np.floor(self.year_start_wolves * (1 + self.max_wolf_growth_per_year)))

        if total_wolves <= max_allowed:
            return

        excess = total_wolves - max_allowed

        # Remove excess wolves gradually from the largest packs first
        packs_sorted = sorted(wolves, key=lambda w: w.pack_size, reverse=True)

        for pack in packs_sorted:
            if excess <= 0:
                break

            removable = min(pack.pack_size - 1, excess)  # leave at least 1 wolf if possible
            if removable > 0:
                pack.pack_size -= removable
                excess -= removable

        # Remove empty packs if any somehow hit zero
        for pack in list(self.agents_by_type.get(WolfPack, [])):
            if pack.pack_size <= 0:
                pack.remove()

    def make_deer_groups(self, total_deer, herd_size_bounds):
        herd_sizes = self.generate_herd_sizes(total_deer, herd_size_bounds)
        n_groups = len(herd_sizes)

        headings = [self.random_heading() for _ in range(n_groups)]

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
        self.enforce_wolf_growth_cap()

        self.datacollector.collect(self)

        if self.steps % 104 == 0:
            self.year_start_wolves = sum(
                w.pack_size for w in self.agents_by_type.get(WolfPack, [])
            )

        if self.steps >= self.max_steps:
            self.running = False