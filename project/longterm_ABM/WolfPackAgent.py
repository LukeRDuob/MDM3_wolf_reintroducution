# WolfPackAgent.py

import numpy as np
from mesa import Agent


class WolfPack(Agent):
    def __init__(
        self,
        model,
        heading,
        pack_size = 5, # in a 30x30 area, each pack gets about 180km^2
        roaming_speed= 8, # 8km a week displacement, allow patrol and find deer
        sensing_radius= 4.0, # can track and sense deer in a 3.5km radius
        hunt_radius=2,
        pack_id = None,
        baseline_kills_per_pack_week=1.0,
        max_kills_per_pack_week= 3,
        half_saturation_herd_size=8,
        weekly_reproduction_rate=0.01, # roughly 2-3 cubs a year per pack
        weekly_death_rate=0.001, # 5% chance of death per year per wolf, from natural causes (not starvation)
        starting_energy_bounds = [0.7, 1],
        weekly_energy_loss=0.15,
        energy_gain_per_deer=0.75,
        reproduction_energy_threshold=0.6,
        starvation_threshold=0.15,
        species="WolfPack",
    ):
        super().__init__(model)

        self.heading = heading
        self.pack_size = pack_size
        self.pack_id = pack_id
        self.roaming_speed = roaming_speed
        self.sensing_radius = sensing_radius
        self.hunt_radius = hunt_radius
        self.energy = self.model.rng.uniform(starting_energy_bounds[0], 
                                             starting_energy_bounds[1])
        self.weekly_energy_loss = weekly_energy_loss
        self.energy_gain_per_deer = energy_gain_per_deer
        self.reproduction_energy_threshold = reproduction_energy_threshold
        self.starvation_threshold = starvation_threshold
        self.baseline_kills_per_pack_week = baseline_kills_per_pack_week
        self.max_kills_per_pack_week = max_kills_per_pack_week
        self.half_saturation_herd_size = half_saturation_herd_size
        self.weekly_reproduction_rate = weekly_reproduction_rate
        self.weekly_death_rate = weekly_death_rate
        self.species = species

    def step(self):
        self.move()
        self.hunt()
        self.lose_energy()
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
            n
            for n in self.model.space.get_neighbors(self.pos, self.sensing_radius, True)
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

        # Prey availability effect: small herds harder to hunt successfully
        prey_factor = target.group_size / (
            target.group_size + self.half_saturation_herd_size
        )

        # Expected kills for the pack this week under current prey conditions
        expected_kills = self.baseline_kills_per_pack_week * prey_factor
        expected_kills = min(expected_kills, self.max_kills_per_pack_week)

        # Convert expected kills into probabilities for 0, 1, or 2 kills
        # This keeps average-based behaviour but allows stochastic outcomes.
        if expected_kills < 0.33:
            probs = [0.75, 0.23, 0.02]
        elif expected_kills < 0.66:
            probs = [0.45, 0.45, 0.10]
        elif expected_kills < 1.0:
            probs = [0.20, 0.65, 0.15]
        elif expected_kills < 1.33:
            probs = [0.10, 0.65, 0.25]
        elif expected_kills < 1.66:
            probs = [0.05, 0.50, 0.45]
        else:
            probs = [0.02, 0.35, 0.63]

        actual_kills = self.model.rng.choice([0, 1, 2], p=probs)
        actual_kills = min(actual_kills, target.group_size)

        if actual_kills > 0:
            self.energy = min(
                1.0,
                self.energy + actual_kills * self.energy_gain_per_deer
            )

        target.group_size -= actual_kills
        self.model.weekly_deer_kills += actual_kills
        self.model.total_deer_killed += actual_kills

    def lose_energy(self):
        self.energy = max(0.0, self.energy - self.weekly_energy_loss)

    def update_pack_size(self):
        if self.energy > self.reproduction_energy_threshold:
            births = self.model.rng.binomial(self.pack_size, self.weekly_reproduction_rate)
        else:
            births = 0

        natural_deaths = self.model.rng.binomial(self.pack_size, self.weekly_death_rate)

        starvation_deaths = 0
        if self.energy < self.starvation_threshold:
            starvation_pressure = (self.starvation_threshold - self.energy) / self.starvation_threshold
            starvation_rate = 0.05 + 0.20 * starvation_pressure
            starvation_deaths = self.model.rng.binomial(self.pack_size, starvation_rate)

        self.pack_size = max(0, self.pack_size + births - natural_deaths - starvation_deaths)

    

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
            baseline_kills_per_pack_week=self.baseline_kills_per_pack_week,
            max_kills_per_pack_week= self.max_kills_per_pack_week,
            half_saturation_herd_size=self.half_saturation_herd_size,
            weekly_reproduction_rate=self.weekly_reproduction_rate,
            weekly_death_rate=self.weekly_death_rate,
            starting_energy_bounds=[self.energy, self.energy],
            weekly_energy_loss=self.weekly_energy_loss,
            energy_gain_per_deer=self.energy_gain_per_deer,
            reproduction_energy_threshold=self.reproduction_energy_threshold,
            starvation_threshold=self.starvation_threshold,
            species=self.species,
        )

        offset = self.model.random_heading() * 0.5
        new_pos = self.pos + offset
        new_pos, new_pack.heading = self.model.clip_and_reflect(new_pos, new_pack.heading)

        self.model.space.place_agent(new_pack, new_pos)
   