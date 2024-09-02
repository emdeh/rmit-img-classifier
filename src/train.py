
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

def confirm_options(args):
    print("\nPlease confirm your training options:")
    print(f"Model architecture: {args.arch}")
    print(f"Number of hidden units: {args.hidden_units}")
    print(f"Number of epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Device: {args.device}")
    print(f"Data directory: {args.data_dir}")
    
    confirm = input("\nAre these settings correct? (y/n): ").strip().lower()
    
    if confirm == 'y':
        print("Training the model with the specified options.\n"
              "The next prompt will be confirm hardware availability.\n"
              "Then the Epoch status and statistics will be displayed.")
        return args
    else:
        # Optionally allow the user to re-enter options
        print("Let's modify the options.")
        return modify_options(args)

def modify_options(args):
    # Allow users to modify the settings
    args.arch = input(f"Model architecture (vgg16 or densenet121) [Press enter for default: {args.arch}]: ") or args.arch

    hidden_units = input(f"Number of hidden units [Press enter for default: {args.hidden_units}]: ")
    args.hidden_units = int(hidden_units) if hidden_units else args.hidden_units

    epochs = input(f"Number of epochs [Press enter for default: {args.epochs}]: ")
    args.epochs = int(epochs) if epochs else args.epochs

    learning_rate = input(f"Learning rate [Press enter for default: {args.learning_rate}]: ")
    args.learning_rate = float(learning_rate) if learning_rate else args.learning_rate

    args.device = input(f"Device (cpu or gpu) [Press enter for default: {args.device}]: ") or args.device
    args.data_dir = input(f"Data directory [Press enter for default: {args.data_dir}]: ") or args.data_dir

    confirm_options(args)

    return args

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

    # Confirm options with the user
    args = confirm_options(args)

    # Select model architecture
    if args.arch == 'vgg16':
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif args.arch == 'densenet121':
        model = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
    else:
        raise ValueError("Architecture not supported. Choose either 'vgg16' or 'densenet121'.")

    # Create instance of DataLoader
    data_loader = DataLoader(data_dir=args.data_dir)

    # Load data
    train_loader, valid_loader, test_loader = data_loader.load_data()

    # Initialize the trainer
    trainer = ModelTrainer(model, hidden_units=args.hidden_units, epochs=args.epochs, learning_rate=args.learning_rate)

    # Select the device
    trainer.device = get_device() if args.device == 'gpu' else trainer.device('cpu')

    # Train the model
    trainer.train(train_loader, valid_loader)

    # Save checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.save_checkpoint(model, trainer.optimiser, trainer.epochs, 'checkpoint.pth')

if __name__ == '__main__':
    main()
