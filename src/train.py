import sys
import os
import argparse
import torch
from src.model import ModelTrainer, CheckpointManager
from src.utils import get_device
from src.data_loader import DataLoader

def main():
    parser = argparse.ArgumentParser(description='Train a neural network.')
    
    # Model architecture
    parser.add_argument('--arch', type=str, default='vgg16', 
                        help='Model architecture (vgg16 or densenet121)')
    
    # Hyperparameters
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    
    # Device
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    # Parse arguments
    args = parser.parse_args()

    # Select model architecture
    if args.arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif args.arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        raise ValueError("Architecture not supported. Choose either 'vgg16' or 'densenet121'.")
    
    # Load data
    data_loader = DataLoader('path/to/dataset')
    train_loader, valid_loader, test_loader = data_loader.load_data()

    # Initialize the trainer
    trainer = ModelTrainer(model, lr=args.learning_rate, hidden_units=args.hidden_units)

    # Train the model
    trainer.train(train_loader, valid_loader, epochs=args.epochs)

    # Save checkpoint
    checkpoint_manager = CheckpointManager()
    checkpoint_manager.save_checkpoint(model, trainer.optimiser, args.epochs, 'checkpoint.pth')

if __name__ == '__main__':
    main()
