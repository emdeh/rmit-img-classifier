"""
This module provides functionality to train a new deep learning model on a dataset. 
It allows the user to specify parameters such as the architecture, learning rate, 
hidden units, and number of training epochs. The model is then saved as a checkpoint 
for later use.

The script initialises a DataLoader object to load the training and validation datasets, 
sets up the model using a ModelManager object, and handles the training process. 
The final trained model is saved to the specified directory.

Functions:
    main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, device_type): 
    Trains a model using the specified parameters and saves the trained model checkpoint.

Command-line arguments:
    - --data_dir / -dir: Path to the directory containing the training data.
    - --save_dir / -s: Path to the directory where the model's checkpoint will be saved.
    - --arch / -a: The model architecture to use (default is vgg16).
    - --learning_rate / -l: The learning rate for training (default is 0.002).
    - --hidden_units / -u: The number of hidden units in the classifier (default is 4096).
    - --epochs / -e: The number of training epochs (default is 5).
    - --device / -d: Device to use for training, either "cpu" or "gpu" (default is "gpu").

Usage example:
    python train.py --data_dir /path/to/data --save_dir /path/to/save_dir --arch resnet50 \
    --learning_rate 0.001 --hidden_units 512 --epochs 20 --device cpu

Dependencies:
    - torch: A deep learning framework for defining and training models.
    - argparse: For parsing command-line arguments.
    - sys: Provides system-specific functions and handles command-line input.
"""

import sys
import os
import argparse
import logging
import time
import traceback

from model import ModelManager
from utils import DataLoader

# Set up logging
logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, device_type):
    """
    Trains a deep learning model on the provided dataset and saves the trained model 
    checkpoint to a specified directory.

    Args:
        data_dir (str): Path to the directory containing the training and validation datasets.
        save_dir (str): Directory where the trained model checkpoint will be saved.
        arch (str): Model architecture to use (e.g., 'vgg16', 'resnet50').
        learning_rate (float): Learning rate for the optimizer.
        hidden_units (int): Number of hidden units in the model's classifier.
        epochs (int): Number of training epochs.
        device_type (str): Device to use for training ('cpu' or 'gpu').

    Returns:
        None
    """
    # Log the start time
    start_time = time.time()
    logger.info("Training process started...")

    #Check the data directory exists
    if not os.path.exists(data_dir):
        logger.error("The specified data directory does not exist: %s", data_dir)
        raise FileNotFoundError(
            f"The specified data directory does not exist: {data_dir}"
            )

    # Check/create the save directory
    if not os.path.exists(save_dir):
        try:
            logger.info("The checkpoint directory passed %s does not exist", save_dir)
            logger.info("Attemptinmg to create the directory...")
            os.makedirs(save_dir)
            logger.info("Created checkpoint directory: %s", save_dir)
        except Exception as dir_error:
            raise OSError(
                f"Could not create save directory: {save_dir}. Error: {dir_error}"
                ) from dir_error

    # Initialise DataLoader class
    data_loader = DataLoader(data_dir)
    dataloaders, class_to_idx = data_loader.load_data()

    # Initialise ModelManager class
    model_manager = ModelManager(arch, hidden_units, learning_rate, class_to_idx, device_type)

    # Train the model
    model_manager.train(dataloaders, epochs)

    # Save checkpoint
    model_manager.save_checkpoint(save_dir)

    # Log the end time and calculate total runtime
    end_time = time.time()
    total_runtime = end_time - start_time
    logger.info("Script execution finished. Total runtime: %.2f seconds", total_runtime)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train a new network on a dataset.',
        epilog='''
        Example usage:
            python train.py --data_dir /path/to/data --save_dir /path/to/save_dir --arch resnet50 --learning_rate 0.001 --hidden_units 512 --epochs 20 --device cpu
        '''
    )
    # Arguments with short flags
    parser.add_argument(
        '-dir', '--data_dir',
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

    # If no args passed, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()

    try:
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
    except FileNotFoundError as file_error:
        logger.error("File not found: %s", file_error)
        sys.exit(1)
    except ValueError as value_error:
        logger.error("Invalid value encountered: %s", value_error)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user. Exiting gracefully...")
        sys.exit(0)
    except Exception as e: # pylint: disable=W0718
        logger.error("An unexpected error occurred: %s", e)
        # Log the full traceback to get more details about the error
        logger.debug(traceback.format_exc())
        sys.exit(1)
