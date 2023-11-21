import torch
import filters.FilterPredicates


class EdgeCellPredicate(filters.FilterPredicates.Predicate):
    """
    This predicate selects cells that are away from the borders of the tile.
    The borders are passed as parameters in the constructor. if a tile is 128x128 then the start=0 and end=127
    """

    def __init__(self, image_shape, border=3):
        self.image_shape = image_shape
        self.border = border

    def foo(self, b):
        output = (b < self.border).any()
        output |= b[2] > self.image_shape[1] - self.border
        output |= b[3] > self.image_shape[0] - self.border

        return output

    def apply(self, x: dict[str, torch.tensor]):
        return [not self.foo(b) for b in x["boxes"]]
