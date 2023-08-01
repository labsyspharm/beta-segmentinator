import torch
import filters.FilterPredicates


class CellSizePredicate(filters.FilterPredicates.Predicate):
    """
    This filter takes upper and lower bound for cell size and selects those cells that fit in that range.
    Optionally you can set either to @code{None} and it would be ignored.
    """

    def __init__(self, max_threshold=None, min_threshold=None):
        self.max_threshold = max_threshold
        self.min_threshold = min_threshold

    def apply(self, x: dict[str, torch.tensor]):
        if self.max_threshold is not None and self.min_threshold is not None:
            return [(b[2] - b[0]) * (b[3] - b[1]) <= self.max_threshold and (b[2] - b[0]) * (b[3] - b[1]) >= self.min_threshold for b in x["boxes"]]

        if self.max_threshold is not None:
            return [(b[2] - b[0]) * (b[3] - b[1]) <= self.max_threshold for b in x["boxes"]]

        if self.min_threshold is not None:
            return [(b[2] - b[0]) * (b[3] - b[1]) >= self.min_threshold for b in x["boxes"]]

        return [True] * len(x["boxes"])
