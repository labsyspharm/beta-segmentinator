import numpy
import scipy.ndimage
import torch
import torchvision.transforms.functional
import sklearn.mixture
import skimage
import tqdm


class MaskGenerator:
    def __init__(self, component_index=2, mask_strategy="addition", gmm_strategy: str = "all"):
        """
        :param component_index: GMM component to use for thresholding 0 (lowest) - 2 (highest).
        :type component_index: int
        :param mask_strategy: How to integrate masks into single file, addition where overlapping masks are added. xor for logical xor for masking where overlapping masks are ignored on intersection.
        :param gmm_strategy: Choose whether to train the gmm with all the tiles (all) or one gmm per cell (independent)
        """
        self.component_index = component_index
        self.mask_strategy = mask_strategy
        self.random = numpy.random
        self.gmm_strategy = gmm_strategy

    def select_random(self, x, y):
        i = 0
        if x.shape[0] != 0:
            i = self.random.randint(0, x.shape[0])
        return x[i], y[i]

    def _calculate_threshold_gmm(self, tile: torch.tensor) -> float:
        gmm = sklearn.mixture.GaussianMixture(n_components=3)
        gmm.fit(
            torch.log10(tile[tile > 0]).reshape((-1, 1))
        )

        m = gmm.means_
        c = gmm.covariances_

        index = numpy.where(m == sorted(m)[self.component_index if self.gmm_strategy == "all" else 1])[0][0]

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

    def _make_square(self, input: numpy.ndarray) -> numpy.ndarray:
        if input.shape[0] == input.shape[1]:
            return input

        output = None
        i = max(input.shape)
        j = i - min(input.shape)

        if input.shape[0] < input.shape[1]:
            output = numpy.vstack( (input, numpy.zeros((j, i))) )
        else:
            output = numpy.hstack( (input, numpy.zeros((i, j))) )

        output[0:input.shape[0], 0:input.shape[1]] = input

        return output

    def _generate_mask(self, cell: torch.tensor, threshold: float, mask: numpy.ndarray | None) -> torch.tensor:
        output = cell.clone()
        output[cell >= threshold] = 1
        output[cell < threshold] = 0

        """
        temp = self._make_square(output)
        index, labels = scipy.sparse.csgraph.connected_components(temp, directed=False)
        if index != 1:
            items, counts = numpy.unique(labels, return_counts=True)
            sel = sorted(zip(items, counts), key= lambda x: x[1])
            sel = sel[-1][0]
            origx, origy = numpy.where(temp == 1)
            origx = origx[labels == sel]
            origy = origy[labels == sel]
            origx, origy = self.select_random(origx, origy)
            output = skimage.segmentation.flood(output, (origx, origy))


        del index
        del temp
        del labels
        """
        if mask is not None and output.shape == mask.shape:
            m = numpy.copy(mask)
            x, y = numpy.meshgrid(numpy.linspace(-3, 3, m.shape[0]), numpy.linspace(-3, 3, m.shape[1]))
            coords = numpy.column_stack((x.ravel(), y.ravel()))  # Combine x and y into (x, y) pairs
            gauss = scipy.stats.multivariate_normal(mean=[0, 0], cov=[[1.5, 0], [0, 1.5]])
            m = m * gauss.pdf(coords).reshape(m.shape)

            x, y = (m == numpy.max(m)).nonzero()
            x, y = self.select_random(x, y)
            output = skimage.segmentation.flood(output, (x, y))

            #output = numpy.logical_and(output, mask)


        output = scipy.ndimage.binary_fill_holes(output).astype(int)

        return output

    def _generate_mask_from_scratch(self, cell: torch.tensor):
        threshold = self._calculate_threshold_gmm(cell)
        return self._generate_mask(cell, threshold, None)


    def generate_list(self, tiff: torch.FloatTensor, boxes: torch.IntTensor | numpy.ndarray):
        """
        Generates a \code{list} with each cell detected by \code{boxes} marked with a binary mask by a GMM with 3 components.
        :param tiff: Original tiff file containing raw pixel data, normalized to 8 bit.
        :param boxes: Output from the segmentation model following x0,y0,x1,y1 format.
        """
        output = list()

        if self.gmm_strategy == "all":
            threshold = self._calculate_threshold_from_tiff(tiff, boxes)

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
        else:
            for i, box in enumerate(boxes):
                output.append(
                    self._generate_mask_from_scratch(
                        torchvision.transforms.functional.crop(
                            tiff,
                            box[1],
                            box[0],
                            box[3] - box[1],
                            box[2] - box[0]
                        )
                    )
                )

        return output

    def _write_tile(self, original, new, index):
        output = None

        if self.mask_strategy == "xor":
            output = numpy.logical_xor(original, new)
        else:
            output = numpy.copy(new)

        output *= index

        return output

    def generate_mask_output(self, tiff: torch.FloatTensor, boxes: torch.IntTensor | numpy.ndarray, masks: list = None) -> numpy.ndarray:
        """
        Generates a \code{numpy.ndarray} with the same dimensions as \code{tiff} but with each cell detected by \code{boxes} masked by a GMM with 3 components.
        Each cell is assigned a unique sequential integer value.
        :param tiff: Original tiff file containing raw pixel data, normalized to 8 bit.
        :param boxes: Output from the segmentation model following x0,y0,x1,y1 format.
        :param masks: (Optional) Masks from the prediction, thresholded to binary.
        """
        output = numpy.zeros_like(tiff)

        if self.gmm_strategy == "all":
            threshold = self._calculate_threshold_from_tiff(tiff, boxes)

            for i, box in tqdm.tqdm(enumerate(boxes), desc="Creating masks with threshold", total=boxes.shape[0]):
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
                elif self.mask_strategy == "ignore":
                    m = m - output[box[1]:box[3], box[0]:box[2]]
                    m[m <= 0] = 0
                    m *= (i + 1)
                    output[box[1]:box[3], box[0]:box[2]] = m
        else:
            for i, box in tqdm.tqdm(enumerate(boxes), desc="Creating masks and thresholds", total=boxes.shape[0]):
                m = self._generate_mask_from_scratch(
                    torchvision.transforms.functional.crop(
                        tiff,
                        box[1],
                        box[0],
                        box[3] - box[1],
                        box[2] - box[0]
                    )
                )

                if self.mask_strategy == "xor":
                    m = numpy.logical_xor(output[box[1]:box[3], box[0]:box[2]], m).astype(int)
                    m *= (i + 1)
                    output[box[1]:box[3], box[0]:box[2]] = m
                elif self.mask_strategy == "addition":
                    m *= (i + 1)
                    output[box[1]:box[3], box[0]:box[2]] += m
                elif self.mask_strategy == "new xor":
                    temp = numpy.logical_xor(output[box[1]:box[3], box[0]:box[2]], m).astype(int)
                    m1 = output[box[1]:box[3], box[0]:box[2]] * temp
                    m1 += (numpy.logical_xor(m1, m) * (i + 1))
                    output[box[1]:box[3], box[0]:box[2]] = m1
                elif self.mask_strategy == "ignore":  # ignores intersection of boxes to prevent bleedthough of masks
                    m = m - output[box[1]:box[3], box[0]:box[2]]
                    m[m <= 0] = 0
                    m *= (i + 1)
                    output[box[1]:box[3], box[0]:box[2]] = m

        return output
