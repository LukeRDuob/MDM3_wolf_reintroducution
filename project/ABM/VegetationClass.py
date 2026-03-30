import numpy as np
from mesa import Agent


class Vegetation(Agent):
    def __init__(
        self,
        model,
        saplings,
        trees,
        max_saplings,
        max_trees,
        patch_spacing,
        sapling_regrowth_prob,
        sapling_maturation_prob,
        patch_type="mixed",
        species="Vegetation"
    ):
        super().__init__(model)
        self.saplings = saplings
        self.trees = trees
        self.max_saplings = max_saplings
        self.max_trees = max_trees
        self.patch_spacing = patch_spacing
        self.sapling_regrowth_prob = sapling_regrowth_prob
        self.sapling_maturation_prob = sapling_maturation_prob
        self.patch_type = patch_type
        self.species = species

    @classmethod
    def random_patch(
        cls,
        model,
        patch_spacing,
        sapling_density,
        tree_density,
        sapling_regrowth_prob,
        sapling_maturation_prob
    ):
        """
        Create a vegetation patch whose capacity is based on the area
        represented by the patch, and whose composition depends on patch type.
        """

        # Area represented by one patch
        cell_area = patch_spacing ** 2

        # Baseline capacities from represented area
        base_max_saplings = max(1, int(sapling_density * cell_area))
        base_max_trees = max(1, int(tree_density * cell_area))

        # Choose a patch type to create a more realistic landscape mosaic
        patch_type = model.rng.choice(
            ["woodland", "browse", "mixed", "sparse"],
            p=[0.25, 0.30, 0.35, 0.10]
        )

        if patch_type == "woodland":
            max_saplings = int(base_max_saplings * model.rng.uniform(0.4, 0.8))
            max_trees = int(base_max_trees * model.rng.uniform(1.3, 2.0))

            saplings = model.rng.integers(
                max(1, int(0.2 * max_saplings)),
                max(2, int(0.5 * max_saplings) + 1)
            )
            trees = model.rng.integers(
                max(1, int(0.6 * max_trees)),
                max(2, int(1.0 * max_trees) + 1)
            )

        elif patch_type == "browse":
            max_saplings = int(base_max_saplings * model.rng.uniform(1.2, 2.0))
            max_trees = int(base_max_trees * model.rng.uniform(0.2, 0.7))

            saplings = model.rng.integers(
                max(1, int(0.5 * max_saplings)),
                max(2, int(1.0 * max_saplings) + 1)
            )
            trees = model.rng.integers(
                max(1, int(0.1 * max_trees)),
                max(2, int(0.4 * max_trees) + 1)
            )

        elif patch_type == "mixed":
            max_saplings = int(base_max_saplings * model.rng.uniform(0.8, 1.3))
            max_trees = int(base_max_trees * model.rng.uniform(0.8, 1.3))

            saplings = model.rng.integers(
                max(1, int(0.4 * max_saplings)),
                max(2, int(0.8 * max_saplings) + 1)
            )
            trees = model.rng.integers(
                max(1, int(0.4 * max_trees)),
                max(2, int(0.8 * max_trees) + 1)
            )

        else:  # sparse
            max_saplings = int(base_max_saplings * model.rng.uniform(0.2, 0.6))
            max_trees = int(base_max_trees * model.rng.uniform(0.2, 0.6))

            saplings = model.rng.integers(
                max(1, int(0.2 * max_saplings)),
                max(2, int(0.6 * max_saplings) + 1)
            )
            trees = model.rng.integers(
                max(1, int(0.2 * max_trees)),
                max(2, int(0.6 * max_trees) + 1)
            )

        return cls(
            model,
            saplings=max(1, saplings),
            trees=max(1, trees),
            max_saplings=max(1, max_saplings),
            max_trees=max(1, max_trees),
            patch_spacing=patch_spacing,
            sapling_regrowth_prob=sapling_regrowth_prob,
            sapling_maturation_prob=sapling_maturation_prob,
            patch_type=patch_type
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

            if self.trees > self.max_trees:
                self.trees = self.max_trees

    def get_occupancy(self):
        """
        Fraction of patch vegetation capacity currently filled.
        Trees count more heavily because they represent more established cover.
        """
        current_amount = self.saplings + 2 * self.trees
        max_amount = self.max_saplings + 2 * self.max_trees

        if max_amount == 0:
            return 0.0

        return min(1.0, current_amount / max_amount)

    def get_max_radius(self):
        """
        Maximum visual radius in map units.
        """
        return 0.45 * self.patch_spacing

    def get_patch_radius(self):
        """
        Radius scales with sqrt(occupancy) so visible area scales with vegetation amount.
        """
        return self.get_max_radius() * np.sqrt(self.get_occupancy())

    def step(self):
        self.regrow_saplings()
        self.mature_saplings()