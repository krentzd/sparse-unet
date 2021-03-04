# Sparse U-Net

This is a 2D U-Net implementation that works with sparsely annotated ground-truth data. 

# Installation

Clone this repository and install required packages from requirements.txt 

# Usage

## Collecting training data

For binary classification, the training data that is collected must contain two classes: (1) foreground and (2) background. The training and testing data directories should have the following structure:

```
path/to/train_dir
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

path/to/test_dir
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

## Testing

## Prediction 
