# Sparse U-Net

This is a 2D U-Net Keras implementation that works with sparsely annotated ground-truth data. 

# Installation

Clone this repository and run `pip install -r requirements.txt` to install the required packages.

# Usage

## Collecting training data

For binary classification, the training data that is collected must contain two classes: (1) foreground and (2) background. The training, testing and validation data directories should have the following structure:

```
path/to/directory
|
└───images
|   |--img_1.png
|   |--img_2.png
|   |--...
|   |--img_n.png
|
└───masks
    └───background
    |   |--img_1.png
    |   |--img_2.png
    |   |--...
    |   |--img_n.png
    |
    └───foreground
        |--img_1.png
        |--img_2.png
        |--...
        |--img_n.png
```

## Training

To train the model run the following command:
```
python train.py --train_dir path/to/train_dir --val_dir path/to/val_dir --out_dir path/to/out_dir
```
## Testing / Prediction

To test the model run the following command:
```
python test.py --test_dir path/to/test_dir --out_dir path/to/out_dir --ckpt path/to/checkpoint
```
