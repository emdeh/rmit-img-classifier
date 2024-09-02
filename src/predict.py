'''
This file uses a trained network to predict the class for an input image.
'''

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
    args.checkpoint = input(f"Checkpoint path [Press enter for default: {args.checkpoint}]: ") or args.checkpoint
    top_k = input(f"Top K classes [Press enter for default: {args.top_k}]: ")
    args.top_k = int(top_k) if top_k else args.top_k
    args.category_names = input(f"Category names path [Press enter for default: {args.category_names}]: ") or args.category_names
    args.device = input(f"Device (cpu or gpu) [Press enter for default: {args.device}]: ") or args.device

    return confirm_options(args)

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

    parser.add_argument('--image_path', type=str, help='Path to the image')
    parser.add_argument('--checkpoint', type=str, default='/checkpoints/checkpoint.pth', help='Path to the checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='data/cat_to_name.json',
                        help='Path to category names JSON file')
    parser.add_argument('--device', type=str, default='gpu', help='Device (cpu or gpu)')

    args = parser.parse_args()

    # Load the model, optimiser, and epochs
    model = CheckpointManager.load_checkpoint(args.checkpoint)

    # Load category names
    cat_to_name = load_label_mapping(args.category_names)

    # Load Classifier
    classifier = ImageClassifier(model, cat_to_name)

    # Predict the class of the image
    probs, classes = classifier.predict(args.image_path, args.top_k)

    # Map classes to names using data_loader's cat_to_name attribute
    flower_names = [cat_to_name[cls] for cls in classes]

    print(f"Predicted classes: {flower_names}")
    print(f"Probabilities: {probs}")

if __name__ == '__main__':
    main()
