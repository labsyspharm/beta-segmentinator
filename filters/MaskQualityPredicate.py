import torch
import filters.FilterPredicates


class MaskQualityPredicate(filters.FilterPredicates.Predicate):
    def __init__(self, min_threshold):
        self.threshold = min_threshold

    def apply(self, x: dict[str, torch.tensor]):
        pass