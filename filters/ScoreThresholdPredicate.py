import torch

import filters.FilterPredicates


class ScoreThresholdPredicate(filters.FilterPredicates.Predicate):
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, x: dict[str, torch.tensor]):
        return (x["scores"] > self.threshold).numpy()

