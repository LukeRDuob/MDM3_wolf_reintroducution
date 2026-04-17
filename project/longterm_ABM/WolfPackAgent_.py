#WolfPackAgent.py

import numpy as np
from mesa import Agent
from DeerHerdAgent_ import DeerHerd

class WolfPack(Agent):
    def __init__(
        self,
        model,
        heading,
        pack_size=5,
        pack_id=None,
        roaming_speed=7.0,
        sensing_radius=4.0, 
        hunt_radius=1.5,
        weekly_reproduction_rate=0.0018,
        weekly_death_rate=0.0012,
        species="WolfPack",
    ):
        super().__init__(model)

        self.heading = heading
        self.pack_size = pack_size
        self.pack_id = pack_id
        self.roaming_speed = roaming_speed
        self.sensing_radius = sensing_radius
        self.hunt_radius = hunt_radius
        self.weekly_reproduction_rate = weekly_reproduction_rate
        self.weekly_death_rate = weekly_death_rate
        self.species = species

    def step(self):
        self.move()
        self.hunt()
        self.update_pack_size()
        self.maybe_split()

        if self.pack_size <= 0:
            self.remove()
            return

    def _normalise(self, heading):
        norm = np.linalg.norm(heading)
        if norm > 0:
            return heading / norm
        return self.heading.copy()

    def _add_angular_noise(self, heading, max_angle=np.pi / 8):
        angle = self.model.rng.uniform(-max_angle, max_angle)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([
            [cos_a, -sin_a],
            [sin_a,  cos_a]
        ])
        return rotation_matrix @ heading

    def ret_closest_neighbour(self, neighbours):
        distances = np.array(
            [[n, self.model.space.get_distance(self.pos, n.pos)] for n in neighbours],
            dtype=object,
        )
        return distances[distances[:, 1].argsort()][0][0]

    def move(self):
        deer_neighbours = [
            n for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)
            if n.species == "DeerHerd" and n.group_size > 0
        ]

        if len(deer_neighbours) > 0:
            target = max(
                deer_neighbours,
                key=lambda d: d.group_size / (self.model.space.get_distance(self.pos, d.pos) + 0.1),
            )
            hunt_heading = self.model.space.get_heading(self.pos, target.pos)
            hunt_heading = self._normalise(hunt_heading)
            hunt_heading = self._add_angular_noise(hunt_heading)
            self.heading = self._normalise(hunt_heading)
        else:
            self.heading = self._normalise(self._add_angular_noise(self.heading))

        new_pos = self.pos + (self.heading * self.roaming_speed)
        new_pos, self.heading = self.model.clip_and_reflect(new_pos, self.heading)
        self.model.space.move_agent(self, new_pos)

    def hunt(self):
        deer_neighbours = [
            n for n in self.model.space.get_neighbors(self.pos, self.hunt_radius, True)
            if n.species == "DeerHerd" and n.group_size > 0
        ]

        if len(deer_neighbours) == 0:
            return

        target = self.ret_closest_neighbour(deer_neighbours)

        # Very simple hunt rule:
        # larger herds are a bit easier to hunt from
        if target.group_size >= 18:
            actual_kills = self.model.rng.choice([0, 1, 2], p=[0.10, 0.60, 0.30])
        elif target.group_size >= 8:
            actual_kills = self.model.rng.choice([0, 1, 2], p=[0.20, 0.65, 0.15])
        else:
            actual_kills = self.model.rng.choice([0, 1], p=[0.35, 0.65])

        actual_kills = min(actual_kills, target.group_size)

        target.group_size -= actual_kills
        self.model.weekly_deer_kills += actual_kills
        self.model.total_deer_killed += actual_kills

    def update_pack_size(self):
        total_deer = sum(d.group_size for d in self.model.agents_by_type.get(DeerHerd, []))
        total_wolves = sum(w.pack_size for w in self.model.agents_by_type.get(WolfPack, []))

        deer_per_wolf = total_deer / max(total_wolves, 1)

        # Wolves only grow when deer are very abundant
        if deer_per_wolf > 200:
            births = self.model.rng.binomial(self.pack_size, self.weekly_reproduction_rate)
            extra_deaths = 0

        # Narrow stable zone near target
        elif deer_per_wolf > 180:
            births = self.model.rng.binomial(self.pack_size, 0.0005)
            extra_deaths = 0

        # Decline starts earlier than before
        elif deer_per_wolf > 160:
            births = 0
            extra_deaths = self.model.rng.binomial(self.pack_size, 0.006)

        elif deer_per_wolf > 115:
            births = 0
            extra_deaths = self.model.rng.binomial(self.pack_size, 0.012)

        else:
            births = 0
            extra_deaths = self.model.rng.binomial(self.pack_size, 0.015)

        natural_deaths = self.model.rng.binomial(self.pack_size, self.weekly_death_rate)
        self.pack_size = max(0, self.pack_size + births - natural_deaths - extra_deaths)

        self.model.weekly_wolf_births += births
        self.model.weekly_wolf_natural_deaths += natural_deaths
    

    def maybe_split(self):
        if self.pack_size < 12:
            return

        half_1 = self.pack_size // 2
        half_2 = self.pack_size - half_1

        self.pack_size = half_1

        new_heading = self.model.random_heading()
        new_pack_id = self.model.get_next_pack_id()

        new_pack = WolfPack(
            self.model,
            heading=new_heading,
            pack_size=half_2,
            pack_id=new_pack_id,
            roaming_speed=self.roaming_speed,
            sensing_radius=self.sensing_radius,
            hunt_radius=self.hunt_radius,
            weekly_reproduction_rate=self.weekly_reproduction_rate,
            weekly_death_rate=self.weekly_death_rate,
            species=self.species,
        )

        offset = self.model.random_heading() * 0.5
        new_pos = self.pos + offset
        new_pos, new_pack.heading = self.model.clip_and_reflect(new_pos, new_pack.heading)

        self.model.space.place_agent(new_pack, new_pos)