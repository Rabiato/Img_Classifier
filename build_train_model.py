#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# PROGRAMMER: Johannes SB
# DATE CREATED: 05/20/21
# REVISED DATE:
# PURPOSE:
# Builds and trains a pretrained model together with a classifier
##

# Imports python modules
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models

# Imports functions created for this program
from save_load_checkpoint import save_checkpoint

def build_train_model(train_data, test_data, trainloader, testloader,
                      checkp_save, arch, learning_rate, hidden_units, epoch, gpu):

    # if function call is GPU check if GPU is available and if yes run on GPU
    if gpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # get pretrained model
    #model = models.vgg19(pretrained=True)
    model = getattr(models, arch)(pretrained=True)

    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False

    # assign mapping from class to index to modle
    model.class_to_idx = train_data.class_to_idx
    # define classifier
    model.classifier = nn.Sequential(nn.Linear(25088, hidden_units),
                                     nn.ReLU(),
                                     nn.Dropout(0.2),
                                     nn.Linear(hidden_units, len(model.class_to_idx)),
                                     nn.LogSoftmax(dim=1))



# define criterion. NLLLoss expected log-probabilitie
    criterion = nn.NLLLoss()

    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    model.to(device)

    epochs = epoch
    steps = 0
    running_loss = 0
    print_every = 5

    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            # calculate log-probabilitie
            logps = model.forward(inputs)
            # calculat loss
            loss = criterion(logps, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for inputs, labels in testloader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        test_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.2f}.. "
                      f"Test loss: {test_loss/len(testloader):.2f}.. "
                      f"Test accuracy: {accuracy/len(testloader)*100:.2f}%")
                running_loss = 0
                model.train()

    # save checkpoint
    save_checkpoint(checkp_save, model, optimizer, epochs, arch, learning_rate, hidden_units)
