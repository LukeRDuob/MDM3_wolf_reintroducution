#DeerHerdAgent.py

import numpy as np
from mesa import Agent


class DeerHerd(Agent):
    def __init__(
        self,
        model,
        heading,
        group_size=20,
        roaming_speed= 3.5, # displacement of about 3.5km a week
        sensing_radius= 1.5,
        split_threshold = 12,
        split_probability = 0.08,
        weekly_reproduction_rate = 0.0018, # 2 births per herd per year
        weekly_death_rate=0.0015,
        species="DeerHerd",
    ):
        super().__init__(model)

        self.heading = heading
        self.group_size = group_size
        self.roaming_speed = roaming_speed
        self.split_threshold = split_threshold
        self.split_probability = split_probability
        self.sensing_radius = sensing_radius
        self.weekly_reproduction_rate = weekly_reproduction_rate
        self.weekly_death_rate = weekly_death_rate
        self.species = species

    def step(self):
        self.move()
        self.update_group_size()
        self.maybe_split()

        if self.group_size <= 0:
            self.remove()
            return

    def _normalise(self, heading):
        norm = np.linalg.norm(heading)
        if norm > 0:
            return heading / norm
        return self._add_angular_noise(self.heading.copy())

    def _add_angular_noise(self, heading, max_angle=np.pi / 6):
        angle = self.model.rng.uniform(-max_angle, max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        return rotation_matrix @ heading

    def ret_closest_neighbour(self, neighbours):
        distances = np.array([
            [n, self.model.space.get_distance(self.pos, n.pos)]
            for n in neighbours
        ], dtype=object)
        return distances[distances[:, 1].argsort()][0][0]

    def move(self):
        wolf_neighbours = [
            n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)
            if n.species == "WolfPack"
        ]

        if len(wolf_neighbours) > 0:
            closest_wolf = self.ret_closest_neighbour(wolf_neighbours)
            flee_heading = self.model.space.get_heading(closest_wolf.pos, self.pos)
            flee_heading = self._normalise(flee_heading)
            flee_heading = self._add_angular_noise(flee_heading)
            self.heading = self._normalise(flee_heading)

        else:
            self.heading = self._normalise(self._add_angular_noise(self.heading))

        new_pos = self.pos + (self.heading * self.roaming_speed)
        new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)
        self.model.space.move_agent(self, new_pos)



    def update_group_size(self):
        births = self.model.rng.binomial(self.group_size, self.weekly_reproduction_rate)
        natural_deaths = self.model.rng.binomial(self.group_size, self.weekly_death_rate)

        self.group_size = max(0, self.group_size + births - natural_deaths)


    def maybe_split(self):
        if self.group_size <= self.split_threshold:
            return

        if self.model.rng.random() >= self.split_probability:
            return

        half_1 = self.group_size // 2
        half_2 = self.group_size - half_1

        self.group_size = half_1

        new_heading = self.model.random_heading()

        new_herd = DeerHerd(
            self.model,
            heading=new_heading,
            group_size=half_2,
            roaming_speed=self.roaming_speed,
            sensing_radius=self.sensing_radius,
            weekly_reproduction_rate=self.weekly_reproduction_rate,
            weekly_death_rate=self.weekly_death_rate,
            species=self.species,
        )

        offset = self.model.random_heading() * 0.5
        new_pos = self.pos + offset
        new_pos, new_herd.heading = self.model.clip_and_reflect(new_pos, new_herd.heading)

        self.model.space.place_agent(new_herd, new_pos)