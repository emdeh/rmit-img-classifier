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

import sys
import argparse
from model import ModelManager
from utils import ImageProcessor

def main(**kwargs):
    """
    Function docstring placeholder
    """
    image_path = kwargs['image_path']
    checkpoint_path = kwargs['checkpoint_path']
    top_k = kwargs['top_k']
    category_names_path = kwargs['category_names_path']
    device_type = kwargs['device']

    # Load model from checkpoint
    model_manager = ModelManager.load_checkpoint(checkpoint_path, device_type)

    # Load category names (if provided)
    category_names = None
    if category_names_path:
        category_names = model_manager.load_category_names(category_names_path)

    # Process the image
    image_processor = ImageProcessor()
    image = image_processor.process_image(image_path)

    # Predict the top K classes
    probs, class_indices = model_manager.predict(image, top_k)

    # Map class indices to flower names if category names are provided
    if category_names:
        class_names = model_manager.map_class_to_name(class_indices, category_names)
        print(f"Predicted Classes: {class_names}")
    else:
        print(f"Predicted Classes: {class_indices}")

    print(f"Probabilities: {probs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Predict flower name from an image using a trained model.',
        epilog='''
        Example usage:
            python predict.py --image_path /path/to/image --checkpoint /path/to/checkpoint --top_k 3 --category_names /path/to/cat_to_name.json --device cpu
        '''
    )

    # Optional arguments with short flags
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

    # Call main with kwargs
    main(
        image_path=args.image_path,
        checkpoint_path=args.checkpoint,
        top_k=args.top_k,
        category_names_path=args.category_names,
        device=args.device
    )
