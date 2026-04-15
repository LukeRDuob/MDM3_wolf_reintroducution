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
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

        total_deer = sum(d.group_size for d in self.agents_by_type.get(DeerHerd, []))
        total_wolves = sum(w.pack_size for w in self.agents_by_type.get(WolfPack, []))

        if total_deer <= 0 or total_wolves <= 0:
            self.running = False
            print("Extiction")
            return

        if self.steps >= self.max_steps:
            self.running = False