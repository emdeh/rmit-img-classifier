"""
This module provides utility classes for handling data loading and image processing
tasks.

The `DataLoader` class is responsible for loading and transforming training and
validation datasets, while the `ImageProcessor` class provides methods for processing
images to prepare them for input into deep learning models.

Classes:
    DataLoader: Handles loading and transforming datasets for training and validation.
    ImageProcessor: Provides image preprocessing functionality for input into models.

Dependencies:
    - torch: Provides deep learning framework functionality.
    - torchvision: Contains popular datasets and transforms for image data.
    - PIL: Python Imaging Library, used for opening and manipulating image files.
"""
import os
import torch
from torchvision import datasets, transforms

from PIL import Image, UnidentifiedImageError


class DataLoader:
    """
    A class for loading and transforming image datasets for training and validation.

    Attributes:
        data_dir (str): The directory where the dataset is stored.

    Methods:
        load_data(): Loads the training and validation datasets, applies transformations,
                        and returns the dataloaders and class-to-index mapping.
    """
    def __init__(self, data_dir):
        """
        Initialises the DataLoader with the specified data directory and sets up paths 
        for loading the training and validation datasets.

        Args:
            data_dir (str): Directory containing the dataset (with 'train' and 'valid' subdirectories).

        Returns:
            None
        """
        self.data_dir = data_dir

        # Check the data dir exists
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def load_data(self):
        """
        Loads the training and validation datasets from the specified directory, applies the 
        appropriate transformations for each dataset, and returns dataloaders for use in model 
        training and validation.

        Args:
            None

        Returns:
            tuple: 
                - dataloaders (dict): Dataloaders for the 'train' and 'valid' datasets.
                - class_to_idx (dict): Mapping of class labels to indices.
        """
        train_dir = os.path.join(self.data_dir, 'train')
        valid_dir = os.path.join(self.data_dir, 'valid')

        # Check if train and valid directories exist
        if not os.path.isdir(train_dir):
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(valid_dir):
            raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

        print(f"Loading training data from {train_dir}")
        print(f"Loading validation data from {valid_dir}")

        # Define transforms
        try:
            data_transforms = {
                'train': transforms.Compose([
                    transforms.RandomRotation(30),
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
                ]),
                'valid': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            }
            print("Data trainsformations complete...")
        except Exception as e:
                raise RuntimeError(f"Error defining data transformations: {e}")

        # Load datasets
        try:
            image_datasets = {
                x: datasets.ImageFolder(f"{self.data_dir}/{x}", transform=data_transforms[x])
                for x in ['train', 'valid']
            }
            # Check if they are empty
            for x in ['train', 'valid']:
                if len(image_datasets[x]) == 0:
                    raise ValueError(f"No images found in {x} dataset at {os.path.join(self.data_dir, x)}")
        except Exception as e:
            raise RuntimeError(f"Error loading image datasets: {e}")
        
        # Create dataloaders
        try:
            dataloaders = {
                x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
                for x in ['train', 'valid']
            }
        except Exception as e:
            raise RuntimeError(f"Error creating dataloaders: {e}")
            
        print("Data loaded")
        return dataloaders, image_datasets['train'].class_to_idx

class ImageProcessor:
    """
    A class for handling the preprocessing of images for input into deep learning models.

    This class provides static methods to process image files, applying transformations
    such as resizing, cropping, converting the image to a tensor, and normalising the pixel values 
    to match the requirements for model inference or training.

    Methods:
        process_image(image_path): Loads an image from the specified file path and applies 
                                   necessary preprocessing steps to prepare it for input 
                                   into a model.
    """

    def process_image(self, image_path):
        """
        Processes an image for use in the model by applying transformations such as 
        resizing, cropping, converting to a tensor, and normalising the pixel values.

        Args:
            image_path (str): Path to the image file to be processed.

        Returns:
            torch.Tensor: A tensor representation of the processed image, ready for input 
            to the model.
        """
        # Check if image file exists
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Open image
        try:
            image = Image.open(image_path)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Image file not found: {e}")
        except UnidentifiedImageError as e:
            raise UnidentifiedImageError(f"Cannot identify image file: {e}")
        except Exception as e:
            raise RuntimeError(f"Error opening image file: {e}")

        print("Image pre-processing starting...")

        # Apply transformations
        try:
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225])
            ])
            processed_image = preprocess(image)
            print("Image preprocessing complete.")
        except Exception as e:
            raise RuntimeError(f"Error procesing image: {e}")
        
        return processed_image
