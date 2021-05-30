#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Johannes SB
# DATE CREATED: 05/20/21
# REVISED DATE: 
# PURPOSE: 
# Provide Save and load function for model checkpoint
##

# Imports python modules
import torch
import os
import numpy as np
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Imports functions created for this program
# ---

def load_checkpoint(filepath):
    ''' Function that loads a checkpoint and rebuilds the model
    '''
    # load filepath
    checkpoint = torch.load(filepath)
    
    # load pretrained model
    arch = checkpoint['arch']
    model = getattr(models, arch)(pretrained=True)
    
    # freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # load mapping of classes to indices
    model.class_to_idx = checkpoint['class_to_idx']
    # define classifier
    hidden_units = checkpoint['hidden_units']
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, len(model.class_to_idx)),
                                     nn.LogSoftmax(dim=1))
    
    # load model state
    model.load_state_dict(checkpoint['state_dict'])
    learning_rate = checkpoint['learning_rate']
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    # load optimizer state 
    optimizer.load_state_dict(checkpoint['optimizer_dic'])
    
    epoch = checkpoint['epoch']
    
    return model, epoch

def save_checkpoint(filepath, model, optimizer, epochs, arch, learning_rate, hidden_units):
    ''' Function that saves a checkpoint of the model
    '''
    if filepath is not None:
        path_filename = path = os.path.join(filepath , 'checkpoint.pth')
    else:
        path_filename = 'checkpoint.pth'
        
    checkpoint = {'epoch': epochs,
                  'arch': arch,
                  'learning_rate': learning_rate,
                  'hidden_units': hidden_units,
                  'state_dict': model.state_dict(),
                  'optimizer_dic': optimizer.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, path_filename)