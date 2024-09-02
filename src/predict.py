'''
This file uses a trained network to predict the class for an input image.
'''
import sys
import os
import json
import argparse
from src.model import ImageClassifier
from src.utils import ImageProcessor, get_device, CheckpointManager
from src.data_loader import DataLoader


# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Predict the class of an image.
    Args:
        image_path (str): Path to the image.
        checkpoint (str): Path to the checkpoint.
        top_k (int, optional): Return top K most likely classes. Defaults to 5.
        category_names (str, optional): Path to category names JSON file. Defaults to 'data/cat_to_name.json'.
        gpu (bool, optional): Use GPU for inference. Defaults to False.
    Returns:
        None
    """

    parser = argparse.ArgumentParser(description='Predict the class of an image.')
    
    parser.add_argument('image_path', type=str, help='Path to the image')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='data/cat_to_name.json', 
                        help='Path to category names JSON file')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    # Load the model
    model = CheckpointManager.load_checkpoint(args.checkpoint)

    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    # Load the device
    device = get_device()
    
    probs, classes = model.predict(args.image_path, args.top_k)
    
    # Map classes to names
    flower_names = [cat_to_name[cls] for cls in classes]
    
    print(f"Predicted classes: {flower_names}")
    print(f"Probabilities: {probs}")

if __name__ == '__main__':
    main()
