#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Johannes SB
# DATE CREATED: 05/20/21
# REVISED DATE: 
# PURPOSE: 
# Train a new network on a data set with train.py
# Basic usage: python train.py data_directory
# Prints out training loss, validation loss, and validation accuracy as the network trains
# Options:
# Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
# Choose architecture: python train.py data_dir --arch "vgg13"
# Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
# Use GPU for training: python train.py data_dir --gpu
#
#   Example call:
#    python train.py pet_images/ --save_dir checkpoint/ --arch "vgg19" 
##

# Imports python modules
import torch
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Imports functions created for this program
from get_input_args import get_input_args_train
from build_train_model import build_train_model

def load_data(data_dir):
    """
    Retrieves a directory which contains the training date, transforms the data,
    loads the datasets with ImageFolder and creats dataloaders torch.utils.data.DataLoader
    
    Parameters:
     directory to training data
    Returns:
     trainloader, testloader, validloader data
    """
    # build path to training data
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)

    # Using the image datasets and the transforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
    
    return train_data, test_data, valid_data, trainloader, testloader, validloader



# Main program function defined below
def main():

    # This function retrieves 7 Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args_train()

    # load training data
    train_data, test_data, valid_data, trainloader, testloader, validloader = load_data(in_arg.data_directory)
    
    # build and train the model as well as saving the model checkpoint
    build_train_model(train_data, test_data, trainloader, testloader, 
                      in_arg.save_dir, in_arg.arch, in_arg.learning_rate, 
                      in_arg.hidden_units, in_arg.epochs, in_arg.gpu)
    
    
# Call to main function to run the program
if __name__ == "__main__":
    main()