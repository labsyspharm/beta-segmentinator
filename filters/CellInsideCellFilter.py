import torch
import torchvision

from filters import FilterPredicates


class CellIsInsidePredicate(FilterPredicates.Predicate):
    """
    This predicate removes cells that are fully enclosed in other cells.

    This filter is necessary because the different size cells dont get removed by nms, if this happens we would look at
    \code{criteria} which can be the area of the cells or the score of the prediction. The bigger of either will remain.
    """

    def __init__(self, criteria="area"):
        self.criteria = criteria

    def apply(self, x: dict[str, torch.tensor]):
        boxes = x["boxes"]

        output = [True] * len(boxes)

        areas = torchvision.ops.box_area(boxes)

        #criteria = areas if self.criteria == "area" else x["scores"]
        inter, _ = torchvision.ops.boxes._box_inter_union(boxes, boxes)

        # fill diagonal to avoid comparing cells to themselves and calculate as fraction of area of cell so = 1 when fully inside
        inter = inter.fill_diagonal_(0) / areas

        indexes = (inter == 1).nonzero(as_tuple=True)

        for index in indexes[1]:
            if index.numel() == 0:
                continue  # technically should be break but jic
            output[index] = False

        return output
