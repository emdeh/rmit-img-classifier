"""
This module provides the functionality to predict the class of a flower from an image 
using a pre-trained model. The user can specify the image path, model 
checkpoint, the number of top predicted classes to return, and an optional JSON file 
for mapping class indices to human-readable names.

The script loads a model checkpoint, processes the input image, and predicts the top 
K classes along with their associated probabilities.

Functions:
    main(**kwargs): Processes the input arguments, loads the model and image, 
    performs the prediction, and displays the predicted classes and their probabilities.

Command-line arguments:
    - --image_path / -i: Path to the input image file.
    - --checkpoint / -c: Path to the model checkpoint file.
    - --top_k / -k: Number of top K predicted classes to return (default is 5).
    - --category_names / -n: Path to a JSON file for mapping class indices to flower names.
    - --device / -d: Device to use for inference, either "cpu" or "gpu" (default is "gpu").

Usage example:
    python predict.py --image_path /path/to/image.jpg --checkpoint /path/to/checkpoint.pth \
    --top_k 5 --category_names /path/to/cat_to_name.json --device gpu

Dependencies:
    - torch: A deep learning framework used for loading models and performing inference.
    - argparse: Handles parsing of command-line arguments.
    - sys: Provides system-specific functions for handling command-line input.
"""

import os
import sys
import argparse
import logging
import traceback

from model import ModelManager
from utils import ImageProcessor

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(**kwargs):
    """
    Performs the prediction of the top K classes for a given image using a pre-trained 
    model checkpoint.

    Args:
        **kwargs: Arbitrary keyword arguments, including:
            - image_path (str): Path to the image file.
            - checkpoint_path (str): Path to the model checkpoint file.
            - top_k (int): Number of top K classes to return.
            - category_names_path (str): Path to a JSON file mapping class indices to category names.
            - device (str): Device type to use for inference ('cpu' or 'gpu').

    Returns:
        None
    """
    # Map arguments to variables
    image_path = kwargs['image_path']
    checkpoint_path = kwargs['checkpoint_path']
    top_k = kwargs['top_k']
    category_names_path = kwargs['category_names_path']
    device_type = kwargs['device']

    # Check if image file exists
    if not os.path.isfile(image_path):
        logger.error("Image file not found: %s", image_path)
        raise FileNotFoundError(f"Image file not found: {image_path}")

    # Check if checkpoint file exists
    if not os.path.isfile(checkpoint_path):
        logger.error("Checkpoint file not found: %s", checkpoint_path)
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    # Load a model from checkpoint
    try:
        logger.info("Loading model checkpoint from: %s", checkpoint_path)
        model_manager = ModelManager.load_checkpoint(checkpoint_path, device_type)
    except Exception as load_error:
        logger.error("Failed to load model checkpoint: %s", load_error)
        raise RuntimeError(f"Failed to load model checkpoint: {load_error}") from load_error

    # Load category names
    category_names = None
    if category_names_path:
        if not os.path.isfile(category_names_path):
            logger.error("Category names file not found: %s", category_names_path)
            raise FileNotFoundError(f"Category names file not found: {category_names_path}") from category_names_path
        
        try:
            logger.info("Loading category names from: %s", category_names_path)
            category_names = model_manager.load_category_names(category_names_path)
        except Exception as category_error:
            logger.error("Failed to load category names from file: %s", category_error)
            raise RuntimeError(f"Failed to load category names from file: {category_error}") from category_error

    # Process the image
    image_processor = ImageProcessor()
    logger.info("Processing image: %s", image_path)
    image = image_processor.process_image(image_path)

    # Predict the top K classes
    try:
        probs, class_indices = model_manager.predict(image, top_k)
    except Exception as prediction_error:
        logger.error("Failed to make prediction: %s", prediction_error)
        raise RuntimeError(f"Failed to make prediction: {prediction_error}") from prediction_error

    # Map class indices to flower names
    if category_names:
        try:
            class_names = model_manager.map_class_to_name(class_indices, category_names)
            logger.info("Predicted Classes: %s", class_names)
        except Exception as map_error:
            logger.error("Failed to map class indices to category names: %s", map_error)
            raise RuntimeError(f"Failed to map class indices to category names: {map_error}") from map_error
    else:
        logger.info("Predicted Class Indices: %s", class_indices)

    logger.info("Predicted Probabilities: %s", probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image using a trained model.',
        epilog='''
        Example usage:
            python predict.py --image_path /path/to/image --checkpoint /path/to/checkpoint --top_k 3 --category_names /path/to/cat_to_name.json --device cpu
        '''
    )

    # Arguments with flags
    parser.add_argument(
        '-i', '--image_path', 
        required=True,
        help='Path to the input image file. Example: -i /path/to/flower/image.jpg'
    )
    parser.add_argument(
        '-c', '--checkpoint', 
        required=True,
        help='Path to the model checkpoint file to load. Example: -c /path/to/checkpoint.pth'
    )
    parser.add_argument(
        '-k', '--top_k', 
        type=int,
        default=5,
        help='Return the top K most likely classes. Example: -k 5 (default)'
    )
    parser.add_argument(
        '-n', '--category_names', 
        required=True,
        type=str,
        help='Path to a JSON file mapping categories to flower names. Example: -n /path/to/cat_to_name.json'
    )
    parser.add_argument(
        '-d', '--device',
        default='gpu',
        type=str,
        choices=['cpu', 'gpu'],
        help='Device to use for inference: "cpu" or "gpu". Example: -d gpu'
    )

    # If no args passed, print help
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    # Parse arguments
    args = parser.parse_args()

    # Call main function
    try:
        main(
            image_path=args.image_path,
            checkpoint_path=args.checkpoint,
            top_k=args.top_k,
            category_names_path=args.category_names,
            device=args.device
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
