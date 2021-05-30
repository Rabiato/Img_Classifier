#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Johannes SB
# DATE CREATED: 05/20/21
# REVISED DATE: 
# PURPOSE: 
# Provides a image processing function to bring input image in the right format 
# for the model
##

# Imports python modules
import torch
import numpy as np
from PIL import Image

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    with Image.open(image) as im:
        width, height = im.size
        aspect_ratio = width / height
        if width < height:
            new_width = 256
            new_height = int(new_width / aspect_ratio)
        elif height < width:
            new_height = 256
            new_width = int(width * aspect_ratio)
        else: # when both sides are equal
            new_width = 256
            new_height = 256
        image = im.resize((new_width, new_height))

        # crop out the center 224x224 portion of the image
        # if width or hight is odd number casting will cause an off center
        # crop by .5 pixel which is acceptable for this use case
        sidelength = 224
        left = int((width - sidelength)/2)
        top = int((height - sidelength)/2)
        right = int((width + sidelength)/2)
        bottom = int((height + sidelength)/2)

        im = im.crop((left, top, right, bottom))

        # convert image to numpy array an adapt color channels
        # to values 0-1
        np_image = np.array(im)/255 # np_image: height x width x channel
        # normalized imgage
        means = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - means) / std
        # transpose color channel to be first and retain order of the other two dimensions
        np_image_t = np_image.transpose((2, 0, 1))

        image_tensor = torch.from_numpy(np_image_t)

    return image_tensor