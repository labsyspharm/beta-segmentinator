import argparse
import os.path
import pickle
import sys

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patches
import torchvision.transforms.functional
import napari
import tifffile
import torch.cuda
import torchvision.ops
import numpy
import tqdm

import GetStatistics
import MaskGenerator
import ZarrStorageHandler
import filters.FilterPredicates
import filters.CellSizePredicate
import filters.MaskQuantityPredicate
import filters.EdgeCellPredicate
import filters.ScoreThresholdPredicate
import filters.MaskQualityPredicate


def parse_args():
    output = argparse.ArgumentParser("Cell segmentor using MaskRCNN for DNA channel.")

    output.add_argument("input", type=str, help="Path to input tiff file.")
    output.add_argument("output", type=str, help="Path to store output.")
    output.add_argument("--thres-nms", type=float, help="NMS threshold.", default=0.1)
    output.add_argument("--thres-prediction", type=float, help="Prediction score threshold, if None it will use outliers from current input mean - 2 * std.", default=None)
    output.add_argument("--thres-mask", type=float, help="Mask threshold, if None it will use outliers from current input mean - 2 * std.", default=None)
    output.add_argument("--thres-size", type=float, help="Size threshold for cells in pixels, if None it will use outliers from current input mean - 2 * std.", default=None)
    output.add_argument("--tile-size", type=int, help="Size of tiles to feed the segmentor model.", default=128)
    output.add_argument("--model-path", type=str, help="Path to MaskRCNN segmentor.", default="model.pt")
    output.add_argument("--device", type=str, help="Device to run segmentation. eg \"cpu\" or \"cuda:N\" where N is gpu id.", default="cuda:0")
    output.add_argument("--rolling-window", type=int, help="How many pixels to move for each rolling window step.", default=10)
    output.add_argument("--no-viewer", action="store_true", help="Do not launch the Napari viewer to visualize output.")
    output.add_argument("--no-output", action="store_true", help="Do not save the final tiff.")
    output.add_argument("--dapi-channel", type=int, help="Which channel in the input file is DAPI.", default=0)
    output.add_argument("--dilation-pixels", type=int, help="How many pixels to dilated for cytoplasm inclusion. 0 or lower skips this step.", default=3)
    output.add_argument("--no-intermediate", action="store_true", help="Do not store intermediate steps.")
    output.add_argument("--dilation-pixels-microns", type=int, help="How many microns to dilate for cytoplasm inclusion. 0 or lower skips this step.", default=None)

    return output.parse_args(sys.argv[1:])


def normalize_8_bit(image):
    if image.dtype == numpy.int8:
        return image / (2**8)  # tecnically not necessary but for completion-wise
    elif image.dtype == numpy.float16 or image.dtype == numpy.uint16:
        return image / (2**16)
    elif image.dtype == numpy.float32 or image.dtype == numpy.uint32:
        return image / (2**32)
    elif image.dtype == numpy.float64 or image.dtype == numpy.uint64:
        return image / (2**64)
    else:
        raise Exception("Invalid dtype {}".format(image.dtype))


def filter_tile2(args, res, tile_area, i, j):
    pass


def correct_coords_trim_masks(args, res, i, j):
    n_masks = list()
    for k in range(len(res["boxes"])):
        # trim mask to bounded box
        n_masks.append(res["masks"][k][
                       0,
                       int(res["boxes"][k][1]):int(res["boxes"][k][3]),
                       int(res["boxes"][k][0]):int(res["boxes"][k][2])
                       ])
        # update boxes coordinates from tile coord to image coord
        res["boxes"][k][0] += j
        res["boxes"][k][1] += i
        res["boxes"][k][2] += j
        res["boxes"][k][3] += i

    res["masks"] = n_masks
    res["boxes"] = res["boxes"].int()


def filter_tile(args, res, tile_area, i, j):
    for key in res:
        res[key] = res[key].detach().cpu()

    res["masks"] = res["masks"].numpy()
    res["boxes"] = res["boxes"].numpy()

    index = (res["scores"] >= args.thres_prediction).numpy()

    res["scores"] = res["scores"][index]
    res["masks"] = res["masks"][index]
    res["boxes"] = res["boxes"][index]

    if len(res["boxes"]) == 0:
        return res

    if res["boxes"].ndim == 1:
        res["boxes"] = res["boxes"].reshape((1, *res["boxes"].shape))

    if res["masks"].ndim == 1:
        res["masks"] = res["masks"].reshape((1, *res["masks"].shape))

    # filter big prediction boxes (consider making it a parameter)
    index = [(b[2] - b[0]) * (b[3] - b[1]) / tile_area <= args.thres_size for b in res["boxes"]]

    res["scores"] = res["scores"][index]
    res["masks"] = res["masks"][index]
    res["boxes"] = res["boxes"][index]

    if len(res["boxes"]) == 0:
        return res

    # filter boxes that end on tile edge
    index = [not ((b < 3) | (b > args.tile_size - 3)).any() for b in res["boxes"]]

    res["scores"] = res["scores"][index]
    res["masks"] = res["masks"][index]
    res["boxes"] = res["boxes"][index]

    if len(res["boxes"]) == 0:
        return res

    # keep only decently sized cells
    index = [(b[2] - b[0]) * (b[3] - b[1]) > 30 for b in res["boxes"]]

    res["scores"] = res["scores"][index]
    res["masks"] = res["masks"][index]
    res["boxes"] = res["boxes"][index]

    if len(res["boxes"]) == 0:
        return res

    # keep only those who have masks
    index = [mask.sum() > 30 for mask in res["masks"]]
    res["scores"] = res["scores"][index]
    res["masks"] = res["masks"][index]
    res["boxes"] = res["boxes"][index]

    if len(res["boxes"]) == 0:
        return res

    n_masks = list()

    for k in range(len(res["masks"])):
        #res["masks"][k] = (res["masks"][k] >= args.thres_mask).astype(int)


        # trim mask to bounded box
        n_masks.append(res["masks"][k][
                                0,
                                int(res["boxes"][k][1]):int(res["boxes"][k][3]),
                                int(res["boxes"][k][0]):int(res["boxes"][k][2])
                                ])

        """
        #n_masks.append( ( (j, i), res["masks"][k]) )
        """
        # update boxes coordinates from tile coord to image coord
        res["boxes"][k][0] += j
        res["boxes"][k][1] += i
        res["boxes"][k][2] += j
        res["boxes"][k][3] += i

    #del res["masks"]
    res["masks"] = n_masks

    return res


def save_intermediate_step(res, path):
    with open(path, "wb") as f:
        output = {"scores": res["scores"], "masks": res["masks"], "boxes": res["boxes"]}
        pickle.dump(output, f)


def load_all_steps(path):
    output = {"scores": list(), "masks": list(), "boxes": list()}

    for p in os.listdir(path):
        fpath = os.path.join(path, p)
        if os.path.isfile(fpath):
            with open(fpath, "rb") as f:
                temp = pickle.load(f)
                for box in temp["boxes"]:
                    if box.ndim == 1:
                        output["boxes"].append(torch.tensor(box))
                    else:
                        for b in box:
                            output["boxes"].append(torch.tensor(b))

                for score in temp["scores"]:
                    if score.ndim == 0:
                        output["scores"].append(score.view(1))
                    else:
                        output["scores"].append(score)

                for mask in temp["masks"]:
                    output["masks"].append(mask)
                """
                for key in temp:
                    for element in temp[key]:
                        if element.ndim == 2:
                            output[key].extend(element)
                        elif element.ndim != 0:
                            output[key].append(element)
                        else:
                            output[key].append(element.view(1))
                """
    output["boxes"] = torch.vstack(output["boxes"])
    output["scores"] = torch.cat(output["scores"], 0)

    return output


def extract_tile_run_model_zarr(args, tiff, model, coordX, coordY, counter):
    tile = torchvision.transforms.functional.crop(tiff, coordX, coordY, args.tile_size, args.tile_size)
    tile = torch.FloatTensor(tile).reshape((1, args.tile_size, args.tile_size)).cuda()

    res = model([tile])[0]

    correct_coords_trim_masks(args, res, coordX, coordY)

    return res


def extract_tile_run_model_save(args, tiff, model, coordX, coordY, counter):
    tile = torchvision.transforms.functional.crop(tiff, coordX, coordY, args.tile_size, args.tile_size)
    tile = torch.FloatTensor(tile).reshape((1, args.tile_size, args.tile_size)).cuda()

    res = model([tile])[0]
        
    res = filter_tile(args, res, args.tile_size ** 2, coordX, coordY)

    if len(res["boxes"]) != 0:
        save_intermediate_step(res, os.path.join(args.output, "step1", str(counter) + ".pkl"))


def merge_predictions(args, base, new):
    for key in new.keys():
        if isinstance(new[key], list):
            base[key].extend([x.detach().cpu() for x in new[key]])
        else:
            base[key].extend(new[key].detach().cpu())


def tile_extraction_part(args, tiff, model):
    TILE_AREA = args.tile_size ** 2
    counter = 0

    data = dict()
    data["boxes"] = list()
    data["masks"] = list()
    data["scores"] = list()
    data["labels"] = list()

    for i in tqdm.tqdm(range(0, tiff.shape[0], args.rolling_window), desc="out"):
        if i + args.tile_size > tiff.shape[0]:
            break

        for j in tqdm.tqdm(range(0, tiff.shape[1], args.rolling_window), desc="in"):
            if j + args.tile_size > tiff.shape[1]:
                break

            #extract_tile_run_model_save(args, tiff, model, i, j, counter)
            res = extract_tile_run_model_zarr(args, tiff, model, i, j, counter)
            merge_predictions(args, data, res)
            counter += 1

    if tiff.shape[0] % args.rolling_window != 0:
        # last row
        for j in tqdm.tqdm(range(0, tiff.shape[1], args.rolling_window), desc="j"):
            if j + args.tile_size >= tiff.shape[1]:
                break

            #extract_tile_run_model_save(args, tiff, model, tiff.shape[0] - args.tile_size, j, counter)
            res = extract_tile_run_model_zarr(args, tiff, model, tiff.shape[0] - args.tile_size, j, counter)
            merge_predictions(args, data, res)
            counter += 1

    if tiff.shape[1] % args.rolling_window != 0:
        # last columns
        for i in tqdm.tqdm(range(0, tiff.shape[0], args.rolling_window), desc="i"):
            if i + args.tile_size >= tiff.shape[0]:
                break

            #extract_tile_run_model_save(args, tiff, model, i, tiff.shape[1] - args.tile_size, counter )
            res = extract_tile_run_model_zarr(args, tiff, model, i, tiff.shape[1] - args.tile_size, counter )
            merge_predictions(args, data, res)
            counter += 1

    if tiff.shape[0] % args.rolling_window != 0 and tiff.shape[1] % args.rolling_window != 0:
        #extract_tile_run_model_save(args, tiff, model, tiff.shape[0] - args.tile_size, tiff.shape[1] - args.tile_size, counter)
        res = extract_tile_run_model_zarr(args, tiff, model, tiff.shape[0] - args.tile_size, tiff.shape[1] - args.tile_size, counter)
        merge_predictions(args, data, res)
        counter += 1

    data["boxes"] = numpy.vstack(data["boxes"])

    return data

def plot_full(tiff, boxes, scores):
    plt.imshow(tiff)
    ax = plt.gca()
    for i  in range(len(boxes)):
        ax.add_patch(
            matplotlib.patches.Rectangle((boxes[i][0], boxes[i][1]), boxes[i][2] - boxes[i][0], boxes[i][3]-boxes[i][1], fill=False, alpha=1, color="red")
        )
        plt.text(boxes[i][0], boxes[i][1], str(scores[i]), fontsize=8, color="white")
    plt.show()


def load_tiff(path, channel=None):
    output = None

    if channel is None:
        output = tifffile.imread(path)
    else:
        output = tifffile.imread(path, key=channel)

    if output.ndim >= 3 and output.shape[0] > 1:
        output = output[0]

    return output


def pipeline(args):
    device = "cpu"
    original_shape = None

    if "cuda" in args.device:
        if int(args.device.split(":")[-1]) < torch.cuda.device_count():
            device = args.device
        else:
            print("Invalid gpu id. Detected {} but id {} was selected.".format(torch.cuda.device_count(), int(args.device.split(":")[-1])))
            sys.exit(1)

    tiff = load_tiff(args.input, args.dapi_channel)

    tiff = normalize_8_bit(tiff) * 255.0
    tiff = torch.FloatTensor(tiff.astype(numpy.float16))

    original_shape = tiff.shape
    data = None

    if len(os.listdir(os.path.join(args.output, "step1"))) == 0:
        print("No previous run found")

        model = torch.load(args.model_path, map_location=torch.device(device))
        model.eval()

        # iterate tiff by tile size
        # strategy would be to load 3x3 square surrounding current tile to ensure we have all overlapping predictions for current tile
        # first step is to filter out predictions outside current tile (remeber to adapt x,y coordinates for adjacent tiles)
        # Then apply size threshold
        # apply prediction threshold followed by nms threshold
        # store after binary mask threshold
        # next tile...

        data = tile_extraction_part(args, tiff, model)

        os.makedirs(os.path.join(args.output, "step1"), exist_ok=True)
        if not args.no_intermediate:
            ZarrStorageHandler.SegmentinatorDatasetWrapper.save_all(
                os.path.join(args.output, "step1"),
                data["boxes"],
                data["scores"],
                data["masks"]
            )

        # free some memory
        del model


    #output = load_all_steps(os.path.join(args.output, "step1"))
    #del output["masks"]
    output = None

    if args.no_intermediate:
        output = data

        output["boxes"] = output["boxes"].astype(int)
        output["scores"] = numpy.array([x.reshape((1)) for x in output["scores"]])
        # lets not use theese masks for now
        """
        size_1, size_2 = 0, 0

        for mask in output["masks"]:
            if mask.shape[0] > size_1:
                size_1 = mask.shape[0]
            if mask.shape[1] > size_2:
                size_2 = mask.shape[1]

        for i in range(len(output["masks"])):
            m = torch.zeros((size_1, size_2), dtype=int)
            m[0:output["masks"][i].shape[0], 0:output["masks"][i].shape[1]] = (output["masks"][i] != 0).int()
            output["masks"][i] = m
        
        output["masks"] = numpy.array(output["masks"])
        """
    else:
        output = ZarrStorageHandler.SegmentinatorDatasetWrapper(os.path.join(args.output, "step1"))

    stats = GetStatistics.do_plots(output["boxes"], output["masks"], output["scores"], img=tiff)
    plt.savefig(os.path.join(args.output, "stats.png"), dpi=200)

    print("Loaded")

    filterPredicates = filters.FilterPredicates.FilterPredicateHandler()

    if args.thres_size is None:
        filterPredicates.add_filter(
            filters.CellSizePredicate.CellSizePredicate(max_threshold=stats["box_size_std"],
                                                        min_threshold=10)
        )
    else:
        filterPredicates.add_filter(
            filters.CellSizePredicate.CellSizePredicate(max_threshold=args.thres_size,
                                                        min_threshold=10)
        )

    filterPredicates.add_filter(
        filters.EdgeCellPredicate.EdgeCellPredicate(image_shape=tiff.shape)
    )

    if args.thres_mask is None:
        filterPredicates.add_filter(
            filters.MaskQuantityPredicate.MaskQuantityPercentagePredicate(min_percentage=max(0, stats["box_perc_std_down"]))
        )
    else:
        filterPredicates.add_filter(
            filters.MaskQuantityPredicate.MaskQuantityPercentagePredicate(args.thres_mask)
        )

    if args.thres_prediction is None:
        filterPredicates.add_filter(
            filters.ScoreThresholdPredicate.ScoreThresholdPredicate(max(0, stats["scores_std_down"]))
        )
    else:
        filterPredicates.add_filter(
            filters.ScoreThresholdPredicate.ScoreThresholdPredicate(args.thres_prediction)
        )

    del tiff

    index = filterPredicates.apply(output)

    b = output["boxes"][index]
    s = output["scores"][index]
    #m = [output["masks"][i] for i in range(len(output["masks"])) if index[i]]

    b = torch.tensor(b.astype(numpy.float32))
    s = torch.tensor(s.reshape((-1)).astype(numpy.float32))

    indexes = torchvision.ops.nms(b, s, args.thres_nms).numpy()

    print("From original {} cells, after filtering we are left with {}, after NMS we are left with {}.".format(
        len(output), index.sum(), len(indexes)
    ))

    b = b.type(torch.int32)

    b = b[indexes]
    s = s[indexes]
    #m = m[indexes]

    #m = [output["masks"][i] for i in indexes if index[i]]

    tiff = load_tiff(args.input, args.dapi_channel)
    tiff = torch.FloatTensor(tiff.astype(numpy.int32))

    mg = MaskGenerator.MaskGenerator(component_index=2,
                                     mask_strategy="ignore",
                                     gmm_strategy="individual",
                                     dilation=args.dilation_pixels)

    final, final_dilated = mg.generate_mask_output(tiff, b, None)

    del tiff

    if not args.no_viewer:
        """
        original_mask = numpy.zeros_like(final)

        for i, box in enumerate(b):
            if box[3] - box[1] != m[i].shape[0] or box[2] - box[0] != m[i].shape[1]:
                continue
            original_mask[box[1]:box[3], box[0]:box[2]] += (m[i] != 0).int()
        """
        viewer = napari.Viewer()
        original = tifffile.imread(args.input)

        if original.ndim >= 3 and original.shape[0] > 1:
            original = original[0]

        viewer.add_image(original)
        shapes = viewer.add_shapes(
            [
                [
                    [box[1].item(), box[0].item()],
                    [box[3].item(), box[2].item()]
                ] for box in b
            ],
            edge_width=1,
            edge_color="coral",
            text={"string": "{scores:.4f}", "anchor": "center", "color": "red", "size": 6},
            features={"scores": [x.item() for x in s]},
            blending="translucent",
            name="Bounding Boxes",
            opacity=1.0,
            shape_type="rectangle",
            face_color="transparent"
        )
        mask = viewer.add_labels(
            final.astype(int),
            name="Masks",
            opacity=0.4
        )
        """
        viewer.add_image(
            original_mask,
            name="Original Masks",
            opacity=0.4
        )
        """
        if final_dilated is not None:
            viewer.add_labels(
                final_dilated.astype(int),
                name="Masks dilated",
                opacity=0.4
            )

        viewer.show(block=True)

    tifffile.imwrite(os.path.join(args.output, "output.tiff"), final)

    if final_dilated is not None:
        tifffile.imwrite(os.path.join(args.output, "output_dilated.tiff"), final_dilated)

    print("Done :)")


if __name__ == "__main__":
    args = parse_args()
    print(args)

    if torch.cuda.device_count() == 0 and args.device == "cuda":
        print("Pytorch cannot find GPU devices (and gpu device is selected).")
        sys.exit(1)

    if not os.path.isfile(args.input):
        print("Input file not found {}.".format(args.input))
        sys.exit(1)

    if not os.path.isfile(args.model_path):
        print("Model file not found {}.".format(args.model_path))
        sys.exit(1)

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, "step1"), exist_ok=True)
        
    if not os.path.exists(os.path.join(args.output, "step1")):
        os.makedirs(os.path.join(args.output, "step1"), exist_ok=True)

    with torch.no_grad():
        pipeline(args)
