import torch
import filters.FilterPredicates


class EdgeCellPredicate(filters.FilterPredicates.Predicate):
    """
    This predicate selects cells that are away from the borders of the tile.
    The borders are passed as parameters in the constructor. if a tile is 128x128 then the start=0 and end=127
    """

    def __init__(self, coord_start=2, coord_end=125):
        self.edge_0 = coord_start
        self.edge_1 = coord_end

    def apply(self, x: dict[str, torch.tensor]):
        return [not ((b < self.edge_0) | (b > self.edge_1)).any() for b in x["boxes"]]
