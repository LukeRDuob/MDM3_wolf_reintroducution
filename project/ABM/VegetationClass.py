#VegetationClass.py

from mesa import Agent


class Vegetation(Agent):
    def __init__(
        self,
        model,
        stage="sapling",
        growth_time=20, # change depending on time frame, adjusted for step size in model init
        species="Vegetation"
    ):
        super().__init__(model)
        self.stage = stage
        self.growth_time = growth_time
        self.age = 0
        self.species = species

    def step(self):
        self.age += 1

        if self.stage == "sapling" and self.age >= self.growth_time:
            self.stage = "tree"