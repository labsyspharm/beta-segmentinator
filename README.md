# UnMicst Segmentation step (Segmentinator... inator)
This repo contains the segmentation step for debugging purposes using Mask-RCNN models.

## How to run
1. Install required packages ```pip install -u requirements.txt``` (advisable to use a virtual environment)
2. ```python main.py input output``` Where ```input``` is the tiff file to segment and ```output``` is the directory to store the output tff.

### If you find bugs or comments please open an issue in this repo so we can keep track of them, remember to add as much detail as posible.

## IMPORTANT
- Use gpu, otherwise it take a while.
- For documentation on the available flags, just run ``` python main.py``` without arguments or ```python main.py -h```.
- If you want to use a specific model, you can pass the ```---mode-path``` flag with the full path to the saved model file to load that instead of the default.
- Paramenters can be set with the ```--thres-*``` flags like ```--thres-nms```.
- Default running device is GPU 1, if you get error make sure you have a GPU and that it is available to run.
