#VegetationClass.py

from mesa import Agent


class Vegetation(Agent):
    def __init__(
        self,
        model,
        saplings,
        trees,
        max_saplings,
        sapling_regrowth_prob,
        sapling_maturation_prob,
        species="Vegetation"
    ):
        super().__init__(model)
        self.saplings = saplings
        self.trees = trees
        self.max_saplings = max_saplings
        self.sapling_regrowth_prob = sapling_regrowth_prob
        self.sapling_maturation_prob = sapling_maturation_prob
        self.species = species

    @classmethod
    def random_patch(
        cls,
        model,
        min_patch_saplings,
        max_patch_saplings,
        min_patch_trees,
        max_patch_trees,
        max_saplings_per_patch,
        sapling_regrowth_prob,
        sapling_maturation_prob
    ):
        saplings = model.rng.integers(min_patch_saplings, max_patch_saplings + 1)
        trees = model.rng.integers(min_patch_trees, max_patch_trees + 1)

        return cls(
            model,
            saplings=saplings,
            trees=trees,
            max_saplings=max_saplings_per_patch,
            sapling_regrowth_prob=sapling_regrowth_prob,
            sapling_maturation_prob=sapling_maturation_prob
        )

    def regrow_saplings(self):
        if self.saplings < self.max_saplings:
            fullness = self.saplings / self.max_saplings
            adjusted_regrowth_prob = self.sapling_regrowth_prob * (1 - fullness)

            if self.model.rng.random() < adjusted_regrowth_prob:
                self.saplings += 1

    def mature_saplings(self):
        if self.saplings > 0:
            num_matured = 0
            for _ in range(self.saplings):
                if self.model.rng.random() < self.sapling_maturation_prob:
                    num_matured += 1

            self.saplings -= num_matured
            self.trees += num_matured

    def step(self):
        self.regrow_saplings()
        self.mature_saplings()