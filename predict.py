#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Johannes SB
# DATE CREATED: 05/20/21
# REVISED DATE:
# PURPOSE:
# Uses a trained network to predict the class for an input image.
##

# Imports python modules
import torch
import json
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Imports functions created for this program
from get_input_args import get_input_args_predict
from process_image import process_image
from save_load_checkpoint import load_checkpoint

def predict(image_path, model, topk, gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # if function call is GPU check if GPU is available and if yes run on GPU
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # code to predict the class from an image file
    classes = []
    l_indices = []
    l_probs = []

    # load and process image
    image_ar = process_image(image_path)
    # add batch dimension if missing
    if len(image_ar.shape) == 3:
        image_ar.unsqueeze_(0)
    # load checkpoint
    model_l, epoch = load_checkpoint(model)
    # Move model to the default device
    model_l.to(device)
    # set model to evaluation mode
    model_l.eval()
    # switch off gradient calculation
    with torch.no_grad():
        # Move input to the default device
        image_ar = image_ar.to(device)
        # Cast to float
        image_ar = image_ar.float()
        # calculate outputs by running images through the network
        logps = model_l(image_ar)
        # Calculate accuracy
        ps = torch.exp(logps)

        # find top-𝐾 most probable indices to classes
        probs, indices  = ps.topk(topk)

    # invert the dictionary to get a mapping from index to class
    idx_to_class = dict([(value, key) for key, value in model_l.class_to_idx.items()])
    # convert probs tensor to list
    for x in np.nditer(np.array(probs)):
        l_probs.append(float(x))

    # get classes
    for x in np.nditer(np.array(indices)):
        l_indices.append(int(x))
    for i in l_indices:
        classes.append(idx_to_class[i])

    return l_probs, classes

# Main program function defined below
def main():

    # This function retrieves Command Line Arugments from user as input from
    # the user running the program from a terminal window. This function returns
    # the collection of these command line arguments from the function call as
    # the variable in_arg
    in_arg = get_input_args_predict()

    # Predict image category of given image
    probs, classes = predict(in_arg.image, in_arg.checkpoint, in_arg.top_k, in_arg.gpu)

    # if name to category is dictonary is passed along map name to classes
    if in_arg.category_names is not None:
        classes_names = []
        with open(in_arg.category_names, 'r') as f:
            cat_to_name = json.load(f)
            # map categories to names
            for i in classes:
                classes_names.append(cat_to_name[i])
        print('The given Image is classified as category {}, with the probability of {}'.format(classes[0],probs[0]))
        print('The category {} is mapped to the name {}'.format(classes[0], classes_names[0]))
    else:
        print('The given Image is classified as category {}, with the probability of {}'.format(classes[0],probs[0]))

# Call to main function to run the program
if __name__ == "__main__":
    main()
