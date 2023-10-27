import numpy
import scipy.ndimage
import torch
import torchvision.transforms.functional
import sklearn.mixture


class MaskGenerator:
    def __init__(self, component_index=2, mask_strategy="addition"):
        """
        :param component_index: GMM component to use for thresholding 0 (lowest) - 2 (highest).
        :type component_index: int
        :param mask_strategy: How to integrate masks into single file, addition where overlapping masks are added. xor for logical xor for masking where overlapping masks are ignored on intersection.
        """
        self.component_index = component_index
        self.mask_strategy = mask_strategy

    def _calculate_threshold_gmm(self, tile: torch.tensor) -> float:
        gmm = sklearn.mixture.GaussianMixture(n_components=3)
        gmm.fit(
            torch.log10(tile[tile > 0]).reshape((-1, 1))
        )

        m = gmm.means_
        c = gmm.covariances_

        index = numpy.where(m == sorted(m)[self.component_index])[0][0]

        m = m[index]
        c = c[index]

        threshold = 10 ** (m - (numpy.sqrt(c) * 2))

        return threshold.item()

    def _calculate_threshold_from_tiff(self, tiff: torch.tensor, boxes: torch.IntTensor | numpy.ndarray) -> float:
        tiles = [
            torchvision.transforms.functional.crop(
                tiff,
                box[1],
                box[0],
                box[3] - box[1],
                box[2] - box[0]
            ).flatten() for box in boxes
        ]  # do supertile in one step for easier memory

        return self._calculate_threshold_gmm(torch.cat(tiles))

    def _generate_mask(self, cell: torch.tensor, threshold: float, mask: torch.FloatTensor) -> torch.tensor:
        output = cell.clone()
        output[cell >= threshold] = 1
        output[cell < threshold] = 0

        if mask is not None and output.shape == mask.shape:
            output = torch.logical_and(output, torch.IntTensor(mask))

        output = scipy.ndimage.binary_fill_holes(output).astype(int)

        return output

    def generate_list(self, tiff: torch.FloatTensor, boxes: torch.IntTensor | numpy.ndarray):
        """
        Generates a \code{list} with each cell detected by \code{boxes} marked with a binary mask by a GMM with 3 components.
        :param tiff: Original tiff file containing raw pixel data, normalized to 8 bit.
        :param boxes: Output from the segmentation model following x0,y0,x1,y1 format.
        """
        threshold = self._calculate_threshold_from_tiff(tiff, boxes)

        output = list()

        for i, box in enumerate(boxes):
            output.append(
                self._generate_mask(
                    torchvision.transforms.functional.crop(
                        tiff,
                        box[1],
                        box[0],
                        box[3] - box[1],
                        box[2] - box[0]
                    ),
                    threshold
                )
            )

        return output

    def generate_mask_output(self, tiff: torch.FloatTensor, boxes: torch.IntTensor | numpy.ndarray, masks: list = None) -> numpy.ndarray:
        """
        Generates a \code{numpy.ndarray} with the same dimensions as \code{tiff} but with each cell detected by \code{boxes} masked by a GMM with 3 components.
        Each cell is assigned a unique sequential integer value.
        :param tiff: Original tiff file containing raw pixel data, normalized to 8 bit.
        :param boxes: Output from the segmentation model following x0,y0,x1,y1 format.
        :param masks: (Optional) Masks from the prediction, thresholded to binary.
        """
        threshold = self._calculate_threshold_from_tiff(tiff, boxes)

        output = numpy.zeros_like(tiff)

        for i, box in enumerate(boxes):
            m = self._generate_mask(
                    torchvision.transforms.functional.crop(
                        tiff,
                        box[1],
                        box[0],
                        box[3] - box[1],
                        box[2] - box[0]
                    ),
                    threshold,
                    masks[i] if masks is not None else None
                )

            if self.mask_strategy == "xor":
                m = numpy.logical_xor(output[box[1]:box[3], box[0]:box[2]], m).astype(int)
                m *= (i + 1)
                output[box[1]:box[3], box[0]:box[2]] = m
            elif self.mask_strategy == "addition":
                m *= (i + 1)
                output[box[1]:box[3], box[0]:box[2]] += m
        return output
