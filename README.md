# Brain Tumor Segmentator
The idea of this project came from a combination of the desire to learn computer vision and the magic of CT scans, which was the first time I was dealing with so close.

## Goals
The goals of this project was to learn how to implement real model from the paper, how to process images, visualize them and train big model.

## Model
This project is based directly on "Attention U-Net: Learning Where to Look for the Pancreas" paper, where Attention U-Net model is described in details.

## Dataset
To train the model I've used dataset from Kaggle: https://www.kaggle.com/datasets/pkdarabi/brain-tumor-image-dataset-semantic-segmentation.

## Training
The model was trained on GPU P100 on Kaggle using batch size 32.
All the details of the training are in [train](train.ipynb) file.

## Results
The model deals with image segmentation increadibly well, being very close to and sometimes even better than original mask. More photos can be seen in [show notebook](show.ipynb)

![alt text](https://github.com/hudyweas/brain-tumor-segmenator/blob/master/out/output_3.png?raw=true)
