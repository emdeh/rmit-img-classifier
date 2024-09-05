"""
Module docstring placeholder
"""
import argparse
from model import ModelManager
from utils import DataLoader

def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, device_type):
    """
    Function docstring placeholder
    """
    # Initialize DataLoader class
    data_loader = DataLoader(data_dir)
    dataloaders, class_to_idx = data_loader.load_data()

    # Initialize ModelManager class
    model_manager = ModelManager(arch, hidden_units, learning_rate, class_to_idx, device_type)

    # Train the model
    model_manager.train(dataloaders, epochs)

    # Save checkpoint
    model_manager.save_checkpoint(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a new network on a dataset.',
        epilog='''
        Example usage:
            python train.py --data_dir /path/to/data --save_dir /path/to/save_dir --arch resnet50 --learning_rate 0.001 --hidden_units 512 --epochs 20 --device cpu
        '''
    )

    # Required arguments with short flags
    parser.add_argument(
        '-dir', '--data_dir', 
        required=True,
        help='Directory of training data. Example: -dir /path/to/data'
    )
    parser.add_argument(
        '-s', '--save_dir', 
        required=True,
        help='Directory to save checkpoint. Example: -s /path/to/save_dir'
    )
    parser.add_argument(
        '-a', '--arch', 
        default='vgg16',
        choices=['vgg16', 'resnet50'],
        help='Model architecture (vgg16 or resnet50). Example: -a vgg16 (default)'
    )
    parser.add_argument(
        '-l', '--learning_rate',
        type=float,
        default=0.002,
        help='Learning rate. Example: -l 0.002 (default)'
    )
    parser.add_argument(
        '-u', '--hidden_units',
        type=int,
        default=4096,
        help='Number of hidden units.  Example: -u 4096 (default)'
    )
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help='Number of training epochs. Example: -e 5 (default)'
    )
    parser.add_argument(
        '-d', '--device',
        default='gpu',
        type=str,
        choices=['cpu', 'gpu'],
        help='Device to use for training: "cpu" or "gpu". Example: -g gpu (default)'
    )

    # Parse arguments
    args = parser.parse_args()

    # Call main function
    main(
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        arch=args.arch,
        learning_rate=args.learning_rate,
        hidden_units=args.hidden_units,
        epochs=args.epochs,
        device_type=args.device
    )
