# Image Classifier
This project was part of the Udacity course AI programming with python.
It is a command line application which contains two parts train and predict

## Train
Train a new network on a data set with train.py
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

## Predict
Uses a trained network to predict the class for an input image.
Options:
1. Non-Optional: path to image
2. Non-Optional: model checkpoint
3. Return top K most likely classes: --top_k
4. Use a mapping of categories to real names: --category_names cat_to_name.json
5. Use GPU for inference: --gpu   
