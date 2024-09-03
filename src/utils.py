"""
This will be a module docstring.

This file is for utility functions used to load data and pre-process images.

Predict flower name from an image with predict.py along with the probability 
of that name. That is, you'll pass in a single image /path/to/image and return 
the flower name and class probability.

Basic usage: python predict.py /path/to/image checkpoint
Options:
    Return top K most likely classes: 
        python predict.py input checkpoint --top_k 3

    Use a mapping of categories to real names:
        python predict.py input checkpoint --category_names cat_to_name.json

    Use GPU for inference:
        python predict.py input checkpoint --gpu

The best way to get the command line input into the scripts is with the 
argparse module(opens in a new tab) in the standard library. 
You can also find a nice tutorial for argparse here(opens in a new tab).
"""
import os
import datetime
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision import models
from torchvision.models.vgg import VGG16_Weights

class ImageProcessor:
    """
    A class for preprocessing and visualising images for inference.

    This class handles the image processing required to prepare images for 
    input into the model and provides visualization utilities to display images 
    alongside their predicted classes and probabilities.

    Attributes
    ----------
    image_path : str
        Path to the image file to be processed.
    
    Methods
    -------
    process_image(self):
        Preprocesses the image for model inference.
    
    imshow(self, image, ax=None, title=None):
        Displays an image after converting it from a PyTorch tensor.
    
    visualise_prediction(self, image_tensor, probs, classes, flower_names, ax_img, ax_bar):
        Visualises the image with predicted classes and probabilities.
    """
    def __init__(self, image_path=None):
        self.image_path = image_path

    @staticmethod
    def process_image(image_path):
        """
        Scales, crops, and normalises a PIL image for a PyTorch model,
        returns a PyTorch tensor.

        Args:
        - image_path (str): Path to the image file.

        Returns:
        - image_tensor (torch.Tensor): Processed image as a PyTorch tensor.
        """
        # Load the image
        pil_image = Image.open(image_path)

        # Resize the image so the shortest side is 256 pixels
        size = 256
        aspect_ratio = pil_image.size[0] / pil_image.size[1]
        if aspect_ratio > 1:
            new_size = (int(aspect_ratio * size), size)
        else:
            new_size = (size, int(size / aspect_ratio))

        pil_image = pil_image.resize(new_size, Image.LANCZOS)

        # Center crop the image to 224x224
        width, height = pil_image.size
        left = (width - 224) / 2
        top = (height - 224) / 2
        right = left + 224
        bottom = top + 224
        pil_image = pil_image.crop((left, top, right, bottom))

        # Convert image to Numpy array and normalise
        np_image = np.array(pil_image) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        np_image = (np_image - mean) / std

        # Reorder dimensions so that color channel is first
        np_image = np_image.transpose((2, 0, 1))

        return np_image

    @staticmethod
    def imshow(image, ax=None, title=None):
        """
        Displays an image after converting it from a PyTorch tensor.

        Args:
        - image (torch.Tensor or np.ndarray): Image to display.
        - ax (matplotlib.axes._axes.Axes, optional): Axes on which to display the image.
        - title (str, optional): Title of the image.

        Returns:
        - ax (matplotlib.axes._axes.Axes): The axes with the image.
        """
        if ax is None:
            ax = plt.subplots()

        # PyTorch tensors assume the color channel is the first dimension
        # but matplotlib assumes it is the third dimension
        if isinstance(image, torch.Tensor):
            image = image.numpy().transpose((1, 2, 0))

        # Undo preprocessing
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)

        ax.imshow(image)

        if title:
            ax.set_title(title)

        return ax

    def visualise_prediction(self, image_tensor, probs, flower_names, ax_img, ax_bar):
        """
        Visualises the image with predicted classes and probabilities.

        Args:
        - image_tensor (torch.Tensor): The input image tensor.
        - probs (list of float): The probabilities of the top predicted classes.
        - classes (list of str): The top predicted classes.
        - flower_names (list of str): The corresponding flower names for the classes.
        - ax_img: Matplotlib Axes for the image.
        - ax_bar: Matplotlib Axes for the bar chart.
        """
        # Display the image
        self.imshow(image_tensor, ax=ax_img)
        ax_img.set_title(flower_names[0])  # Title with the top predicted flower name

        # Plot the probabilities
        y_pos = np.arange(len(flower_names))
        ax_bar.barh(y_pos, probs, align='center')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(flower_names)
        ax_bar.set_xlabel('Probability')
        ax_bar.invert_yaxis()  # Invert y-axis so the highest probability is at the top

class CheckpointManager:
    """
    A class to manage the saving and loading of model checkpoints.

    This class provides methods to save a model's state to a file and load it 
    back into the model, enabling training to resume or evaluation at a later 
    time.

    Methods
    -------
    save_checkpoint(self, model, optimiser, epochs, filepath):
        Saves the model checkpoint to the specified file.
    
    load_checkpoint(filepath):
        Loads the model checkpoint from the specified file and rebuilds the model.
    """

    def save_checkpoint(self, model, optimiser, epochs, filepath="YYYYMMDD-HHMM-checkpoint.pth"):
        """
        Saves the model checkpoint to the specified file.

        Parameters
        ----------
        model : torch.nn.Module
            The trained model to be saved.
        optimiser : torch.optim.Optimizer
            The optimizer associated with the model.
        epochs : int
            The number of epochs the model was trained for.
        filepath : str
            The path where the checkpoint will be saved.
        """
        if filepath is None:
            # Generate a default filename with current date/time
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M")
            filepath = f"checkpoints/{current_time}-checkpoint.pth"

        # Ensure the checkpoint directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Attach the class_to_idx mapping to the model
        # model.class_to_idx = model.class_to_idx # I think this is redundant

        # Create the checkpoint dictionary
        checkpoint = {
            'model_architecture': type(model).__name__,
            'input_size': 25088,  # TODO This should match the input size of the model
            'output_size': 102,  # TODO This should match the output size of the model
            'hidden_layers': [4096],  # List of hidden layer sizes, if applicable
            'state_dict': model.state_dict(),
            'class_to_idx': model.class_to_idx,
            'classifier': model.classifier,
            'optimiser_state_dict': optimiser.state_dict(),
            'epochs': epochs
        }

        # Save the checkpoint
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")

    @staticmethod
    def load_checkpoint(filepath, model_architecture='vgg16', load_optimiser=True):
        """
        Loads the model checkpoint from the specified file and rebuilds the model.

        Parameters
        ----------
        filepath : str
            The path to the checkpoint file.
        model_architecture : str
            The architecture of the model to load (e.g., 'vgg16').
        load_optimiser : bool
            Whether to load the optimiser state from the checkpoint.

        Returns
        -------
        model : torch.nn.Module
            The model rebuilt from the checkpoint.
        optimiser : torch.optim.Optimizer or None
            The optimizer associated with the model, or None if not loaded.
        epochs : int
            The number of epochs the model was trained for.
        """
        # Determine the device to be used
        device = get_device()

        # Load the checkpoint, specify weights_only=False to load the entire checkpoint
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)

        # Rebuild the model based on the architecture in the checkpoint
        if model_architecture == 'vgg16':
            model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            model.classifier = checkpoint['classifier']
        else:
            raise ValueError(f"Model architecture '{model_architecture}' is not supported.")

        # Load the state dict into the model
        model.load_state_dict(checkpoint['state_dict'])

        # Attach the class_to_idx mapping to the model
        model.class_to_idx = checkpoint['class_to_idx']

        # Rebuild optimiser if needed
        optimiser = None
        if load_optimiser:
            try:
                optimiser = torch.optim.Adam(model.parameters())
                optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
            except ValueError as e:
                print(
                    "Warning: Failed to load optimiser state due to a" 
                    "parameter mismatch.\n" 
                    f"Reason: {e}\n"
                    "\n"
                    "This could affect training in the following ways:\n"
                    "1. If you are resuming training after a long session or" 
                    "with a complex model,\n" 
                    "   starting with a fresh optimiser might cause the model" 
                    "to lose some momentum.\n"
                    "   This could lead to slower convergence or different" 
                    "training dynamics.\n"
                    "\n"
                    "2. If you are fine-tuning or performing inference, this" 
                    "generally won't matter as much,\n"
                    "   and the model's weights are still loaded correctly.\n"
                    "\n"
                    "A new optimiser has been reinitialised with default " 
                    "settings."
                )
                # Reinitialise optimiser if loading failed
                optimiser = torch.optim.Adam(model.parameters())

        # Load the number of epochs
        epochs = checkpoint['epochs'] # TODO: Shoudl this be dynamic?

        print(f"Checkpoint loaded from {filepath}")

        return model, optimiser, epochs

    @staticmethod
    def list_checkpoints(directory='checkpoints/'):
        """
        List all the checkpoint files in the specified directory.

        Args:
        - directory (str): The directory containing the checkpoint files.

        Returns:
        - checkpoints (list): List of checkpoint file paths.
        """
        try:
            checkpoints = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pth')]
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Directory not found: {directory}") from exc

        return checkpoints

def get_device():
    """
    Determines and returns the device (CPU or GPU) to be used for training 
    or evaluation.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU with CUDA.")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using AMD GPU with ROCm.")
    else:
        device = torch.device("cpu")
        print("GPU not found or unavailable. Using CPU.")

    return device


def load_label_mapping(label_map_path):
    """
    Loads the mapping of class indices to class labels.

    Args:
    - label_map_path (str): Path to the JSON file containing the label mapping.

    Returns:
    - dict: A dictionary mapping class indices to class labels.
    """
    try:
        with open(label_map_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Label mapping file not found at {label_map_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in label mapping file at {label_map_path}") from exc


def sanity_check(image_paths, model, cat_to_name, topk=5):
    """
    Perform a sanity check by visualizing the model's top K predictions
    alongside the actual images.

    Args:
    - image_paths (list): List of paths to image files.
    - model (torch.nn.Module): Trained PyTorch model for prediction.
    - cat_to_name (dict): Mapping from class indices to flower names.
    - topk (int): Number of top most likely classes to visualize.

    Returns:
    - fig (matplotlib.figure.Figure): The matplotlib figure object containing the plots.
    """
    image_processor = ImageProcessor()

    # Determine the number of rows needed (each img has 1 row, 2 columns)
    n_images = len(image_paths)
    nrows = n_images
    ncols = 2

    # Create a figure with the dynamic number of rows
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(12, 5 * nrows))

    # Flatten the axes array for easier indexing
    axes = axes.flatten()

    for i, image_path in enumerate(image_paths):
        ax_img = axes[2 * i]
        ax_bar = axes[2 * i + 1]

        # Process the image
        image_tensor = image_processor.process_image(image_path)

        # Set model to evaluation mode
        model.eval()

        # Disable gradients for inference
        with torch.no_grad():
            output = model(torch.from_numpy(image_tensor).unsqueeze(0))

        # Get the probabilities and classes
        probs = torch.softmax(output, dim=1).cpu().numpy().flatten()
        top_probs, top_indices = torch.topk(torch.tensor(probs), topk)
        top_probs = top_probs.numpy().flatten()
        top_indices = top_indices.numpy().flatten()

        # Convert indices to classes
        idx_to_class = {val: key for key, val in model.class_to_idx.items()}
        top_classes = [idx_to_class[idx] for idx in top_indices]

        # Map classes to names
        flower_names = [cat_to_name[cls] for cls in top_classes]

        # Display the image
        image_processor.imshow(image_tensor, ax=ax_img)
        ax_img.set_title(flower_names[0])  # Title with the top predicted flower name

        # Plot the probabilities
        y_pos = np.arange(len(flower_names))
        ax_bar.barh(y_pos, top_probs, align='center')
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(flower_names)
        ax_bar.set_xlabel('Probability')
        ax_bar.invert_yaxis()  # Invert y-axis so the highest probability is at the top

    plt.tight_layout()

    # Return the figure object
    return fig