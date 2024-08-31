"""
This will be moduel docstrings

This file is to train a new network on a dataset and save the model as a 
checkpoint.

Requirements:
Train a new network on a data set with train.py

Basic usage: python train.py data_directory

- Prints out training loss, validation loss, and validation accuracy as the network trains
- Options:
    Set directory to save checkpoints
        python train.py data_dir --save_dir save_directory

    Choose architecture
        python train.py data_dir --arch "vgg13"

    Set hyperparameters
        python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20

    Use GPU for training
        python train.py data_dir --gpu

"""
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import DataLoader

# Initialize the DataLoader with the data directory
data_loader = DataLoader('/data/flowers')

# Accessing the train, valid, and test loaders
train_loader = data_loader.train_loader
valid_loader = data_loader.valid_loader
test_loader = data_loader.test_loader

# Accessing the class_to_idx mapping
class_to_idx = data_loader.class_to_idx

# Print out some details to verify
print(f"Number of training batches: {len(train_loader)}")
print(f"Number of validation batches: {len(valid_loader)}")
print(f"Number of test batches: {len(test_loader)}")
print(f"Class to index mapping: {class_to_idx}")