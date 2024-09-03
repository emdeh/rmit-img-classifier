'''
This file uses a trained network to predict the class for an input image.
'''
import os
import sys
import argparse
from utils import CheckpointManager, load_label_mapping
from model import ImageClassifier

def confirm_options(args):
    print("\nPlease confirm your prediction options:")
    print(f"Image path: {args.image_path}")
    print(f"Checkpoint path: {args.checkpoint}")
    print(f"Top K classes: {args.top_k}")
    print(f"Category names path: {args.category_names}")
    print(f"Device: {args.device}")

    confirm = input("\nAre these settings correct? (y/n): ").strip().lower()

    if confirm == 'y':
        print("Proceeding with the specified options.\n")
        return args
    else:
        print("Let's modify the options.")
        return modify_options(args)

def modify_options(args):
    # Allow users to modify the settings
    args.image_path = input(f"Image path [Press enter for default: {args.image_path}]: ") or args.image_path

    # Modify other options if needed
    top_k = input(f"Top K classes [Press enter for default: {args.top_k}]: ")
    args.top_k = int(top_k) if top_k else args.top_k

    args.category_names = input(f"Category names path [Press enter for default: {args.category_names}]: ") or args.category_names
    args.device = input(f"Device (cpu or gpu) [Press enter for default: {args.device}]: ") or args.device

    return confirm_options(args)

def select_checkpoint(args):
    # List available checkpoints and prompt user to select one
    available_checkpoints = CheckpointManager.list_checkpoints()
    if available_checkpoints:
        print("\nAvailable checkpoints:")
        for idx, checkpoint in enumerate(available_checkpoints, start=1):
            print(f"{idx}: {checkpoint}")
        
        selected_idx = input(f"Select checkpoint by number [Press enter for default: {args.checkpoint}]: ")
        if selected_idx.isdigit() and 1 <= int(selected_idx) <= len(available_checkpoints):
            args.checkpoint = available_checkpoints[int(selected_idx) - 1]
    else:
        print("No checkpoints available in the 'checkpoints' directory.")
        sys.exit("Exiting...\n"
                 "Please train a model by using train.py before attempting\n"
                 "to predict an image.")

    return args

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

    parser.add_argument('--image_path', type=str, default='data/flowers/valid/1/image_06739.jpg', help='Path to the image')
    parser.add_argument('--checkpoint', type=str, help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='data/cat_to_name.json',
                        help='Path to category names JSON file')
    
    # Device
    parser.add_argument('--device', type=str, default='gpu', help='Device (cpu or gpu)')

    args = parser.parse_args()

    # First, select the checkpoint
    args = select_checkpoint(args)

    # Then confirm all other options with the user
    args = confirm_options(args)

    # Load the model, optimiser, and epochs
    model, _, _ = CheckpointManager.load_checkpoint(args.checkpoint)

    # Load category names
    cat_to_name = load_label_mapping(args.category_names)

    # Load Classifier
    classifier = ImageClassifier(model, cat_to_name)

    print("Model's class_to_idx mapping:", model.class_to_idx) # Debugging
    print("Loaded cat_to_name keys:", cat_to_name.keys()) # Debugging

    # Predict the class of the image
    probs, classes = classifier.predict(args.image_path, args.top_k)

    # Map classes to names using data_loader's cat_to_name attribute
    flower_names = [cat_to_name.get(cls, "Unknown") for cls in classes]

    print(f"Predicted classes: {flower_names}")
    print(f"Probabilities: {probs}")

if __name__ == '__main__':
    main()