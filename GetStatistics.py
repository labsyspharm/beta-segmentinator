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


def do_plots(boxes, masks, scores, img=None):
    f, ax = plt.subplots(nrows=2, ncols=3)
    ax = ax.flatten()

    x = numpy.array(get_histogram_of_mask_percentage(boxes, masks))
    mean = x.mean()
    std = x.mean() - 2*x.std()
    ax[0].hist(x, bins=20)
    ax[0].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")
    ax[0].axvline(ymin=0, ymax=1, x=std, label="mean - 2 * std: {0:.3f}".format(std), color="red")
    ax[0].set_title("% box as masks.")
    ax[0].legend()
    print("Mask % mean: {} - 2*std: {}".format(mean, std))

    x = numpy.array(get_histogram_of_box_size(boxes))
    mean = x.mean()
    std = x.mean() + 2*x.std()

    ax[1].hist(x, bins=100)
    ax[1].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")
    ax[1].axvline(ymin=0, ymax=1, x=std, label="mean + 2 * std: {0:.3f}".format(std), color="red")
    ax[1].set_title("Box size in pixels")
    ax[1].legend()
    print("Box size mean: {} - 2*std: {}".format(mean, std))

    x = numpy.array(scores)
    mean = x.mean()
    std = x.mean() - 2*x.std()

    ax[2].hist(x, bins=30)
    ax[2].axvline(ymin=0, ymax=1, x=mean, label="mean: {0:.3f}".format(mean), color="green")
    ax[2].axvline(ymin=0, ymax=1, x=std, label="mean - 2 * std: {0:.3f}".format(std), color="red")
    ax[2].set_title("Scores")
    ax[2].legend()
    print("Prediction score mean: {} - 2*std: {}".format(mean, std))

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
        print("Sum of pixel intensity per prediction - Mean: {} - Up std: {} - Low std: {}".format(mean, std, std2))



def pipeline():
    args = parse_args()

    data = main.load_all_steps(args.path)

    do_plots(data["boxes"], data["masks"], data["scores"])

    plt.savefig(args.path)
    plt.show()


if __name__ == "__main__":
    pipeline()
