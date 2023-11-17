import torch

import filters.FilterPredicates


class ScoreThresholdPredicate(filters.FilterPredicates.Predicate):
    def __init__(self, threshold):
        self.threshold = threshold

    def apply(self, x: dict[str, torch.tensor]):
        output = list()

        for s in x["scores"]:
            output.append(s[0] > self.threshold)

        return output
        #return [s > self.threshold for s in x["scores"]]

