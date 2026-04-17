# WeeklyModel.py

import numpy as np
from mesa import Model
from mesa.space import ContinuousSpace
from mesa.datacollection import DataCollector

from DeerHerdAgent_ import DeerHerd
from WolfPackAgent_ import WolfPack


class WeeklySpeciesModel(Model):
    def __init__(
        self,
        width=30,
        height=30,
        max_steps=150000,
        init_total_deer=11835,
        herd_size_bounds=(6, 45),
        init_total_wolves=40,
        pack_size_bounds=(5, 11),

        # vegetation settings
        veg_cell_size=1,
        initial_veg_range=(60, 100),
        max_veg=100.0,
        veg_regrowth_rate=1.0,
        deer_grazing_rate=0.12,

        seed=None
    ):
        super().__init__(seed=seed)

        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.space = ContinuousSpace(self.width, self.height, torus=False)

        self.weekly_deer_kills = 0
        self.total_deer_killed = 0
        self.pack_size_bounds = pack_size_bounds
        self.herd_size_bounds = herd_size_bounds

        # Vegetation grid
        self.veg_cell_size = veg_cell_size
        self.veg_width = int(np.ceil(self.width / self.veg_cell_size))
        self.veg_height = int(np.ceil(self.height / self.veg_cell_size))

        self.max_veg = float(max_veg)
        self.veg_regrowth_rate = float(veg_regrowth_rate)
        self.deer_grazing_rate = float(deer_grazing_rate)

        low, high = initial_veg_range
        self.veg_value = self.rng.uniform(
            low, high, size=(self.veg_width, self.veg_height)
        ).astype(float)

        # stores total grazing pressure applied this step
        self.veg_grazing_pressure = np.zeros((self.veg_width, self.veg_height), dtype=float)

        # make agents
        self.make_deer_groups(init_total_deer, self.herd_size_bounds)
        self.make_wolf_packs(init_total_wolves, self.pack_size_bounds)

        self.datacollector = DataCollector(
            model_reporters={
                "Total Deer": lambda m: sum(d.group_size for d in m.agents_by_type.get(DeerHerd, [])),
                "Total Wolves": lambda m: sum(w.pack_size for w in m.agents_by_type.get(WolfPack, [])),
                "Total Deer Killed": lambda m: m.total_deer_killed,
                "Weekly Deer Killed": lambda m: m.weekly_deer_kills,
                "Deer per Wolf": lambda m: (
                    sum(d.group_size for d in m.agents_by_type.get(DeerHerd, [])) /
                    max(sum(w.pack_size for w in m.agents_by_type.get(WolfPack, [])), 1)
                ),
                "Mean Vegetation": lambda m: float(np.mean(m.veg_value)),
                "Min Vegetation": lambda m: float(np.min(m.veg_value)),
                "Max Vegetation": lambda m: float(np.max(m.veg_value)),
                "Low Vegetation Cells": lambda m: int(np.sum(m.veg_value < 25)),
                "High Vegetation Cells": lambda m: int(np.sum(m.veg_value > 75)),
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

    def generate_group_sizes(self, total, size_bounds, label="groups"):
        min_size, max_size = size_bounds

        if total < min_size:
            raise ValueError(
                f"Total {label} size ({total}) is too small for minimum size {min_size}."
            )

        sizes = []
        remaining = total

        while remaining > 0:
            if min_size <= remaining <= max_size:
                sizes.append(remaining)
                break

            possible_sizes = []
            for size in range(min_size, max_size + 1):
                remainder = remaining - size
                if remainder == 0 or remainder >= min_size:
                    possible_sizes.append(size)

            if not possible_sizes:
                raise ValueError(
                    f"Could not split total {total} into sizes within bounds {size_bounds}."
                )

            chosen_size = int(self.rng.choice(possible_sizes))
            sizes.append(chosen_size)
            remaining -= chosen_size

        self.rng.shuffle(sizes)
        return sizes

    def make_deer_groups(self, total_deer, herd_size_bounds):
        herd_sizes = self.generate_group_sizes(total_deer, herd_size_bounds, label="deer")
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

    def make_wolf_packs(self, total_wolves, pack_size_bounds):
        pack_sizes = self.generate_group_sizes(total_wolves, pack_size_bounds, label="wolves")
        n_packs = len(pack_sizes)

        headings = [self.random_heading() for _ in range(n_packs)]
        pack_ids = list(range(1, n_packs + 1))

        packs = WolfPack.create_agents(
            self,
            n_packs,
            heading=headings,
            pack_size=pack_sizes,
            pack_id=pack_ids,
        )

        for p in packs:
            self.space.place_agent(p, self.random_position())


    def get_veg_cell(self, pos):
        x = int(pos[0] // self.veg_cell_size)
        y = int(pos[1] // self.veg_cell_size)

        x = min(max(x, 0), self.veg_width - 1)
        y = min(max(y, 0), self.veg_height - 1)

        return x, y

    def get_veg_value(self, pos):
        x, y = self.get_veg_cell(pos)
        return float(self.veg_value[x, y])

    def vegetation_score(self, pos):
        """
        Used by deer movement.
        Returns a 0-1 score based on vegetation abundance.
        """
        return self.get_veg_value(pos) / self.max_veg

    def reset_grazing_pressure(self):
        self.veg_grazing_pressure.fill(0.0)

    def graze_vegetation(self, pos, herd_size):
        """
        Deer herd adds grazing pressure to the vegetation cell it occupies.
        Actual vegetation reduction happens later in step_vegetation().
        """
        x, y = self.get_veg_cell(pos)
        self.veg_grazing_pressure[x, y] += herd_size * self.deer_grazing_rate

    def step_vegetation(self):
        """
        Vegetation regrows, then loses biomass to grazing pressure.
        """
        # mostly constant regrowth
        self.veg_value += self.veg_regrowth_rate

        # subtract this step's grazing
        self.veg_value -= self.veg_grazing_pressure

        # keep values in bounds
        self.veg_value = np.clip(self.veg_value, 0.0, self.max_veg)

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

    def step(self):
        self.weekly_deer_kills = 0

        # start a fresh grazing map each week
        self.reset_grazing_pressure()

        # agents move/hunt/graze
        self.agents.shuffle_do("step")

        # update vegetation after all herds have contributed grazing pressure
        self.step_vegetation()

        # collect data
        self.datacollector.collect(self)

        total_deer = sum(d.group_size for d in self.agents_by_type.get(DeerHerd, []))
        total_wolves = sum(w.pack_size for w in self.agents_by_type.get(WolfPack, []))

        if total_deer <= 0 or total_wolves <= 0:
            self.running = False
            print("Extinction")
            return

        if self.steps >= self.max_steps:
            self.running = False