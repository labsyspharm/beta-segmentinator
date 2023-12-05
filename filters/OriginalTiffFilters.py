import torchvision


class OriginalIntesityFilter:
    """
    Filters cell predictions give the total intensity of the predicted box.
    """
    def __init__(self, low=None, high=None):
        self.low = low if low is not None else -1
        self.high = high if high is not None else 2**64  # tiff usually in uint32 so this way we ensure to not overlap

    def apply(self, tiff, boxes):
        output = list()

        for box in boxes:
            temp = torchvision.transforms.functional.crop(
                tiff,
                box[1],
                box[0],
                box[3] - box[1],
                box[2] - box[0]
            )

            total = temp.sum()

            output.append(self.low < total < self.high)

        return output
