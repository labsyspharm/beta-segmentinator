# UnMicst Segmentation step (Segmentinator... inator)
This repo contains the segmentation step for debugging purposes using Mask-RCNN models.

<p align="center">
<img src="https://github.com/labsyspharm/beta-segmentinator/blob/master/image.png?raw=true" />
</p>

## How to run
1. Install required packages ```pip install requirements.txt``` (advisable to use a virtual environment)
2. ```python main.py input output``` Where ```input``` is the tiff file to segment and ```output``` is the directory to store the output tff.

The reference model is located at hits/lsp-analysis/UnMICSTdev/FOR ALEX HUMAN ANNOTATIONS USE THESE/models_that_work/model_trained_without_ignore_manyWindowsAndSynthetic.pt

### If you find bugs or comments please open an issue in this repo so we can keep track of them, remember to add as much detail as posible.

## IMPORTANT
- **This model has been trained with 0.325 microns per pixel, you should be wary of different resolutions.**
- Use gpu, otherwise it take a while.
- **The ```---mode-path``` flag is necesary with the full path to the saved model file to load that instead of the default.**
- For documentation on the available flags, just run ``` python main.py``` without arguments or ```python main.py -h```.
- Paramenters can be set with the ```--thres-*``` flags like ```--thres-nms```.
- Default running device is GPU 1, if you get error make sure you have a GPU and that it is available to run.
