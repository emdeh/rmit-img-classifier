import argparse
from model import ModelManager
from utils import ImageProcessor

def main(image_path, checkpoint_path, top_k, category_names, gpu):
    # Load model from checkpoint
    model_manager = ModelManager.load_checkpoint(checkpoint_path, gpu)
    
    # Load category names (if provided)
    if category_names:
        category_names = model_manager.load_category_names(category_names)

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
    parser = argparse.ArgumentParser(description='Predict flower name from an image.')
    parser.add_argument('image_path', help='Path to image file')
    parser.add_argument('checkpoint', help='Model checkpoint to load')
    parser.add_argument('--top_k', type=int, default=1, help='Return top K most likely classes')
    parser.add_argument('--category_names', help='Path to JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    main(args.image_path, args.checkpoint, args.top_k, args.category_names, args.gpu)
