import numpy as np
from collections import defaultdict


class SpatialHash:
    """
    A spatial hash grid overlay for fast neighbor lookups
    in a ContinuousSpace.
    """

    def __init__(self, width, height, cell_size):
        """
        Args:
            width:     space width
            height:    space height
            cell_size: side length of each grid cell.
                       Should be >= the largest query radius you use frequently.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Number of cells in each dimension
        self.cols = int(np.ceil(width / cell_size))
        self.rows = int(np.ceil(height / cell_size))

        # cell (col, row) -> set of agents
        self.grid = defaultdict(set)

        # agent -> current cell key (for fast removal)
        self.agent_cell = {}

    def _key(self, pos):
        """Convert a continuous position to a grid cell key."""
        col = int(pos[0] // self.cell_size)
        row = int(pos[1] // self.cell_size)
        # Clamp to valid range
        col = max(0, min(col, self.cols - 1))
        row = max(0, min(row, self.rows - 1))
        return (col, row)

    def add(self, agent):
        """Register an agent in the hash."""
        key = self._key(agent.pos)
        self.grid[key].add(agent)
        self.agent_cell[agent] = key

    def remove(self, agent):
        """Remove an agent from the hash."""
        key = self.agent_cell.pop(agent, None)
        if key is not None:
            self.grid[key].discard(agent)

    def update(self, agent):
        """
        Call after moving an agent. Checks if the cell changed,
        and only updates the dict if it did.
        """
        new_key = self._key(agent.pos)
        old_key = self.agent_cell.get(agent)

        if old_key == new_key:
            return  # No cell change, nothing to do

        # Remove from old cell
        if old_key is not None:
            self.grid[old_key].discard(agent)

        # Add to new cell
        self.grid[new_key].add(agent)
        self.agent_cell[agent] = new_key

    def get_neighbors(self, pos, radius, include_center=True, agent=None):
        """
        Return all agents within `radius` of `pos`.

        Args:
            pos:            (x, y) query position
            radius:         search radius
            include_center: if False, exclude `agent` from results
            agent:          the querying agent (excluded when include_center=False)
        """
        # Determine which cells to check
        min_col = max(0, int((pos[0] - radius) // self.cell_size))
        max_col = min(self.cols - 1, int((pos[0] + radius) // self.cell_size))
        min_row = max(0, int((pos[1] - radius) // self.cell_size))
        max_row = min(self.rows - 1, int((pos[1] + radius) // self.cell_size))

        radius_sq = radius * radius
        neighbors = []

        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                for a in self.grid[(col, row)]:
                    if not include_center and a is agent:
                        continue
                    dx = a.pos[0] - pos[0]
                    dy = a.pos[1] - pos[1]
                    if dx * dx + dy * dy <= radius_sq:
                        neighbors.append(a)

        return neighbors

    def get_neighbors_by_species(self, pos, radius, species, agent=None):
        """
        Optimised query that filters by species during the search
        instead of building the full list first.
        """
        min_col = max(0, int((pos[0] - radius) // self.cell_size))
        max_col = min(self.cols - 1, int((pos[0] + radius) // self.cell_size))
        min_row = max(0, int((pos[1] - radius) // self.cell_size))
        max_row = min(self.rows - 1, int((pos[1] + radius) // self.cell_size))

        radius_sq = radius * radius
        neighbors = []

        for col in range(min_col, max_col + 1):
            for row in range(min_row, max_row + 1):
                for a in self.grid[(col, row)]:
                    if a is agent:
                        continue
                    if a.species != species:
                        continue
                    dx = a.pos[0] - pos[0]
                    dy = a.pos[1] - pos[1]
                    if dx * dx + dy * dy <= radius_sq:
                        neighbors.append(a)

        return neighbors

    def rebuild(self, agents):
        """Rebuild the entire hash from scratch (useful at start of step)."""
        self.grid.clear()
        self.agent_cell.clear()
        for agent in agents:
            key = self._key(agent.pos)
            self.grid[key].add(agent)
            self.agent_cell[agent] = key