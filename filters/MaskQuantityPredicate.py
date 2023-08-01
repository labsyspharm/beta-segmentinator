import torch
import filters.FilterPredicates


class MaskQuantityPredicate(filters.FilterPredicates.Predicate):
    """
    This predicate filters on the absolute number of pixels that make the mask of a single cell. Inherently is also filtering by size.
    """
    def __init__(self, min_quantity):
        self.threshold = min_quantity

    def apply(self, x: dict[str, torch.tensor]):
        return [b[1].sum() > self.threshold for b in x["masks"]]


class MaskQuantityPercentagePredicate(filters.FilterPredicates.Predicate):
    """
    This predicate filters on the percentage of the boundingbox that is make up of masked pixels.
    """
    def __init__(self, min_percentage):
        self.threshold = min_percentage

    def apply(self, x: dict[str, torch.tensor]):
        return [b[1].sum() / (b[1].shape[0] * b[1].shape[1]) > self.threshold for b in x["masks"]]
