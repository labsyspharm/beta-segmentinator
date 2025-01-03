import sys
import matplotlib.pyplot as plt
import numpy
import tifffile
import torchvision

import main
import argparse


def parse_args():
    output = argparse.ArgumentParser()

    output.add_argument("path", type=str, help="Path to intermediate segmenting step")

    output = output.parse_args(sys.argv[1:])

    return output


def calculate_box_area(box):
    return (box[3] - box[1] + 1) * (box[2] - box[0] + 1)


def get_histogram_of_mask_percentage(boxes, masks):
    data = list()

    for i in range(len(boxes)):
        data.append(
            masks[i].sum() / calculate_box_area(boxes[i])
        )

    return data


def get_histogram_of_box_size(boxes):
    data = list()

    for i in range(len(boxes)):
        data.append(
            calculate_box_area(boxes[i])
        )

    return data

def get_histogram_of_mask_scores(masks):
    h = list()

    for i in range(len(masks)):
        h.extend(masks[i].flatten())

    return h


def do_plots(boxes, masks, scores, img=None):
    f, ax = plt.subplots(nrows=2, ncols=3, figsize=(32, 18))
    ax = ax.flatten()
    output = dict()

    x = numpy.array(get_histogram_of_mask_percentage(boxes, masks))
    mean = x.mean()
    std = x.mean() - 2*x.std()
    std2 = x.mean() + 2*x.std()
    ax[0].hist(x, bins=20)
    ax[0].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")
    ax[0].axvline(ymin=0, ymax=1, x=std, label="mean - 2 * std: {0:.3f}".format(std), color="red")
    ax[0].axvline(ymin=0, ymax=1, x=std2, label="mean + 2 * std: {0:.3f}".format(std2), color="red")
    ax[0].set_title("% box as masks.")
    ax[0].legend()
    output["box_perc_mean"] = mean
    output["box_perc_std_down"] = std
    output["box_perc_std_up"] = std2
    print("Box % as mask - Mean: {} - Low std: {} - Up std: {}".format(mean, std, std2))

    x = numpy.array(get_histogram_of_box_size(boxes))
    mean = x.mean()
    std = x.mean() + 2*x.std()
    std2 = max(0, x.mean() - 2*x.std())

    ax[1].hist(x, bins=100)
    ax[1].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")
    ax[1].axvline(ymin=0, ymax=1, x=std, label="mean + 2 * std: {0:.3f}".format(std), color="red")
    ax[1].set_title("Box size in pixels")
    ax[1].legend()
    output["box_size_mean"] = mean
    output["box_size_std_up"] = std
    output["box_size_std_down"] = std2
    print("Box size mean: {} - 2*std: {}".format(mean, std))

    x = numpy.array(scores)
    mean = x.mean()
    std = x.mean() + 2*x.std()

    ax[2].hist(x, bins=30)
    ax[2].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")

    if std > 1:
        std = std - x.std()
        ax[2].axvline(ymin=0, ymax=1, x=std, label="mean + std: {0:.3f}".format(std), color="red")
    else:
        ax[2].axvline(ymin=0, ymax=1, x=std, label="mean + 2 * std: {0:.3f}".format(std), color="red")

    ax[2].set_title("Scores")
    ax[2].legend()
    output["scores_mean"] = mean
    output["scores_std_up"] = std
    print("Prediction score mean: {} std: {}".format(mean, x.std()))

    if img is not None:
        data = list()
        for box in boxes:
            m = torchvision.transforms.functional.crop(
                    img,
                    box[1],
                    box[0],
                    box[3] - box[1],
                    box[2] - box[0]
                )

            data.append(m.sum())

        ax[3].hist(data, bins=30)
        x = numpy.array(data)
        mean = x.mean()
        std = x.mean() + 2 * x.std()
        std2 = x.mean() - 2*x.std()
        ax[3].set_title("Sum of pixel intensity per cell")
        ax[3].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")
        ax[3].axvline(ymin=0, ymax=1, x=std, label="mean + 2 * std: {0:.3f}".format(std), color="red")
        ax[3].axvline(ymin=0, ymax=1, x=std2, label="mean - 2 * std: {0:.3f}".format(std2), color="red")
        ax[3].legend()
        output["pixel_intensity_mean"] = mean
        output["pixel_intensity_std_down"] = std
        output["pixel_intensity_std_up"] = std2
        print("Sum of pixel intensity per prediction - Mean: {} - Up std: {} - Low std: {}".format(mean, std, std2))

        x = numpy.array(get_histogram_of_mask_scores(masks))
        ax[4].hist(x[x > .8], bins=200)
        ax[4].set_title("Masks Scores")
        #ax[4].axvline(ymin=0, ymax=1, x=x.std(), label="mean", color="green")

    return output


def pipeline():
    args = parse_args()

    data = main.load_all_steps(args.path)

    do_plots(data["boxes"], data["masks"], data["scores"])

    plt.savefig(args.path)
    plt.show()


if __name__ == "__main__":
    pipeline()
