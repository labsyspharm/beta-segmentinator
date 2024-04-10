import torch
import filters.FilterPredicates


class MaskQuantityPredicate(filters.FilterPredicates.Predicate):
    """
    This predicate filters on the absolute number of pixels that make the mask of a single cell. Inherently is also filtering by size.
    """
    def __init__(self, min_quantity):
        self.threshold = min_quantity

    def apply(self, x: dict[str, torch.tensor]):
        return [(b != 0).sum() > self.threshold for b in x["masks"]]


class MaskQuantityPercentagePredicate(filters.FilterPredicates.Predicate):
    """
    This predicate filters on the percentage of the boundingbox that is make up of masked pixels.
    """
    def __init__(self, min_percentage):
        self.threshold = min_percentage

    def apply(self, x: dict[str, torch.tensor]):
        #return [(b != 0).sum() / ((b.shape[0] + 1) * (b.shape[1] + 1)) > self.threshold for b in x["masks"]]
        output = list()

        boxes = x["boxes"]
        masks = x["masks"]

        for i in range(len(boxes)):
            temp = (masks[i] != 0).sum()
            temp = temp / ( (boxes[i][2] - boxes[i][0] + 1) * (boxes[i][3] - boxes[i][1] + 1) )

            output.append(temp > self.threshold)

        return output
