import math

import numpy
import torch
import tqdm
import zarr
from numcodecs import Blosc
import collections

PredictionTuple = collections.namedtuple("Prediction", ["box", "score", "mask"])


class SegmentinatorDatasetWrapper:
    """
    This class is a wrapper for storing and loading the segmentation predictions, it handles automatically the trimming and padding of the masks for the zarr array.
    """
    def __init__(self, path):
        self.store = zarr.DirectoryStore(path)
        self.data = zarr.group(self.store)

    def __del__(self):
        self.store.close()

    def __getitem__(self, item: int | str) -> PredictionTuple | list:
        """
        Allows to access the whole array of predictions or a single prediction namedtuple.
        It handles removing the extra padding on cell masks.
        :param item: If an int, will return the itemth prediction for each prediction type, if a str the whole item prediction set.
        :returns: A named tuple with the itemth box, score and mask. If a str, then it will return all the predictions regarding item.
        """

        if isinstance(item, str):
            if item not in list(self.data.keys()):
                raise IndexError("Element {} not in Zarr directory, expected one of the following: {}".format(item, list(self.data.keys())))

            #return [getattr(self.__getitem__(x), item[:-1] if item != "boxes" else "box") for x in range(self.__len__())]
            return self.data[item][:]
        else:
            if item < 0 or item >= self.__len__():
                raise IndexError("Invalid index {} for dataset of size {}.".format(item, self.__len__()))

            box = self.data["boxes"][item]
            score = self.data["scores"][item][0]
            #mask = self.data["masks"][item]

            # trim the padding from the mask to match box shape
            #mask = mask[0:box[2] - box[0], 0:box[3] - box[1]]

            return PredictionTuple(box, score, None)#, mask)

    def __len__(self):
        return self.data["boxes"].shape[0]


    @staticmethod
    def save_all(path: str, boxes: torch.tensor, scores: torch.tensor, masks: torch.tensor) -> None:
        """
        This method creates a Zarr directory with the given parameters.
        """
        compressor = Blosc(cname="zstd", shuffle=Blosc.BITSHUFFLE)
        zarr.storage.default_compressor = compressor

        store = zarr.DirectoryStore(path)
        dir_grp = zarr.group(store, overwrite=True)

        size_1 = -1
        size_2 = -1

        for box in boxes:
            t1 = int(box[2] - box[0])
            t2 = int(box[3] - box[1])

            if t1 > size_1:
                size_1 = t1

            if t2 > size_2:
                size_2 = t2

        boxes_dataset = dir_grp.create_dataset("boxes", shape=(len(boxes), 4), dtype="u8")
        masks_dataset = dir_grp.create_dataset("masks", shape=(len(boxes), size_2 + 1, size_1 + 1))
        scores_dataset = dir_grp.create_dataset("scores", shape=(len(boxes), 1))

        for i in tqdm.tqdm(range(len(boxes)), desc="Storing data"):
            boxes_dataset[i, :] = boxes[i]
            masks_dataset[i, 0:masks[i].shape[0], 0:masks[i].shape[1]] = (masks[i] != 0).int()  # watch out for shapes
            scores_dataset[i, 0] = scores[i]

        store.close()
