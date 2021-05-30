#!/usr/bin/env python3
#
# PROGRAMMER: Johannes SB
# DATE CREATED: 05/20/21
# REVISED DATE: 
# PURPOSE: Creates two function that retrieves command line inputs
#          from the user. 
#          One for train function and one for test function
# Command Line Arguments for train function:
#          0) Non-Optional: path to training data
#          1) Set directory to save checkpoints: --save_dir save_directory
#          2) Choose architecture: --arch "vgg13"
#          3) Set hyperparameters:
#          3.1) --learning_rate 0.01
#          3.2) --hidden_units 512
#          3.3) --epochs 20
#          4) Use GPU for training --gpu
#
# Command Line Arguments for predict function:
#          0) Non-Optional: path to image 
#          0.1) Non-Optional: model checkpoint
#          1) Return top K most likely classes: --top_k 3
#          2) Use a mapping of categories to real names: --category_names cat_to_name.json
#          3) Use GPU for inference: --gpu   
##

# Imports python modules
import argparse

def get_input_args_train():
    """
    Function that retrieves command line inputs from the user for train data function. 
    Command Line Arguments:
          0) Non-Optional: path to training data
          1) Set directory to save checkpoints: --save_dir save_directory
          2) Choose architecture: --arch "vgg13"
          3) Set hyperparameters:
          3.1) --learning_rate 0.01
          3.2) --hidden_units 512
          3.3) --epochs 20
          4) Use GPU for training --gpu
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Argument 0 non-optional: that's a path to a folder with the training data
    parser.add_argument('data_directory', type = str, help = 'path to data directory') 
    # Argument 1: that's a path to a folder to save the model checkpoint
    parser.add_argument('--save_dir', type = str, help = 'path to directory to save checkpoints')
    # Argument 2: that's the CNN Model Architecture
    parser.add_argument('--arch', type = str, default = 'vgg19', help = 'CNN Model Architecture')
    # Argument 3.1: that's the hyperparameter learning rate
    parser.add_argument('--learning_rate', type = float, default = 0.001, help = 'hyperparameter learning rate')
    # Argument 3.2: that's the hyperparameter hidden_units
    parser.add_argument('--hidden_units', type = int, default = 512, help = 'hyperparameter hidden_units')
    # Argument 3.3: that's the hyperparameter epochs
    parser.add_argument('--epochs', type = int, default = 3, help = 'hyperparameter epochs')
    # Argument 4: use GPU for training
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU for training')
                        
    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()

    return in_args
                        
def get_input_args_predict():
    """
    Function that retrieves command line inputs from the user for predict data function. 
    Command Line Arguments:
    0) Non-Optional: path to image 
    0.1) Non-Optional: model checkpoint
    1) Return top K most likely classes: --top_k 3
    2) Use a mapping of categories to real names: --category_names cat_to_name.json
    3) Use GPU for inference: --gpu
    
    This function returns these arguments as an ArgumentParser object.
    Parameters:
     None - simply using argparse module to create & store command line arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """
    # Create Parse using ArgumentParser
    parser = argparse.ArgumentParser()

    # Argument 0 non-optional: that's a path to the image the model should give prediction 
    parser.add_argument('image', type = str, help = 'path to image') 
    # Argument 0.1 non-optional: model checkpoint
    parser.add_argument('checkpoint', type = str, help = 'model checkpoint') 
    # Argument 1: Return top K most likely classes
    parser.add_argument('--top_k', type = int, default = 5, help = 'top K most likely classes')
    # Argument 2: Use a mapping of categories to real names
    parser.add_argument('--category_names', type = str, help = 'Use a mapping of categories to real names')
    # Argument 3: use GPU for training
    parser.add_argument('--gpu', action = 'store_true', help = 'Use GPU for training')

    # Assigns variable in_args to parse_args()
    in_args = parser.parse_args()

    return in_args