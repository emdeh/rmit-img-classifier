import argparse
from model import ModelManager
from utils import ImageProcessor

def main(**kwargs):
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
        required=True,
        help='Return the top K most likely classes. Example: -k 5'
    )
    parser.add_argument(
        '-n', '--category_names', 
        required=True,
        type=str, 
        help='Path to a JSON file mapping categories to flower names. Example: -n /path/to/cat_to_name.json'
    )
    parser.add_argument(
        '-d', '--device', 
        required=True,
        default='gpu',
        type=str, 
        choices=['cpu', 'gpu'], 
        help='Device to use for inference: "cpu" or "gpu". Example: -d gpu'
    )

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
