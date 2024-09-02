
"""
Module: train.py

This module is used to train a neural network model using the specified 
architecture, hyperparameters, and dataset.

Usage:
    $ python train.py [options]

Options:
    --arch (str): Model architecture (vgg16 or densenet121). Default is vgg16.
    --learning_rate (float): Learning rate for the optimizer. Default is 0.001.
    --hidden_units (int): Number of hidden units in the model. Default is 4096.
    --epochs (int): Number of training epochs. Default is 5.
    --gpu: Use GPU if available. This is a flag option.

Example:
    $ python train.py --arch vgg16 --learning_rate 0.001 --hidden_units 4096 --epochs 5 --gpu
"""

import argparse
from torchvision import models
from model import ModelTrainer
from data_loader import DataLoader
from utils import CheckpointManager, get_device

def main():
    parser = argparse.ArgumentParser(description='Train a neural network.')

    # Model architecture
    parser.add_argument('--arch', type=str, default='vgg16',
                        help='Model architecture (vgg16 or densenet121)')

    # Hyperparameters
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--running_loss', type=int, default=0, help='Running loss')

    # Device
    parser.add_argument('--device', type=str, default='gpu', help='Device (cpu or gpu)')

    # Data path
    parser.add_argument('--data_dir', type=str, default='data/flowers', help='Path to the dataset')

    # Parse arguments
    args = parser.parse_args()

    # Select model architecture
    if args.arch == 'vgg16':
        model = model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif args.arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Architecture not supported. Choose either 'vgg16' or 'densenet121'.")

    # Create instance of DataLoader
    data_loader = DataLoader(data_dir=args.data_dir)
    
    # Load data
    train_loader, valid_loader, test_loader = data_loader.load_data()

    # Initialize the trainer
    trainer = ModelTrainer(model, hidden_units=args.hidden_units, epochs=args.epochs, learning_rate=args.learning_rate, running_loss=args.running_loss)

    # Select the device
    device = get_device() if args.device == 'gpu' else trainer.device = 'cpu'

    # Train the model
    trainer.train(train_loader, valid_loader, epochs=args.epochs)

    # Save checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.save_checkpoint(model, trainer.optimiser, args.epochs, 'checkpoint.pth')

if __name__ == '__main__':
    main()
