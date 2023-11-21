import unittest
import numpy
from ZarrStorageHandler import SegmentinatorDatasetWrapper


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.path_to_save = "/tmp/lol"
        self.boxes = [[1, 1, 6, 8], [3, 3, 6, 12]]
        self.scores = [0.5, 0.6]
        self.masks = [numpy.random.rand(5, 7), numpy.random.rand(3, 9)]

        SegmentinatorDatasetWrapper.save_all(self.path_to_save, self.boxes, self.scores, self.masks)
        self.dataLoaded = SegmentinatorDatasetWrapper(self.path_to_save)

    def test_accessing_by_index(self) -> None:
        self.assertTrue((self.boxes[0] == self.dataLoaded[0].box).all(), msg="Box 0 is not equal")
        self.assertTrue((self.boxes[1] == self.dataLoaded[1].box).all(), msg="Box 1 is not equal")

        self.assertTrue((self.scores[0] == self.dataLoaded[0].score).all(), msg="Score 0 is not equal")
        self.assertTrue((self.scores[1] == self.dataLoaded[1].score).all(), msg="Score 1 is not equal")

        self.assertTrue((self.masks[0] == self.dataLoaded[0].mask).all(), msg="Mask 0 is not equal")
        self.assertTrue((self.masks[1] == self.dataLoaded[1].mask).all(), msg="Mask 1 is not equal")

    def test_accessing_by_name(self) -> None:
        self.assertTrue(self.scores == self.dataLoaded["scores"],
                        msg="Accessing scores is not equal")
        self.assertTrue(all([(self.masks[i] == self.dataLoaded["masks"][i]).all() for i in range(len(self.masks))]),
                        msg="Accessing masks is not equal")
        self.assertTrue(all([(self.boxes[i] == self.dataLoaded["boxes"][i]).all() for i in range(len(self.boxes))]),
                        msg="Accessing boxes is not equal")


if __name__ == '__main__':
    unittest.main()
