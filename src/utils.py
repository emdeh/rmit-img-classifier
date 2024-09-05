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

import torch
from torchvision import datasets, transforms
from PIL import Image

class DataLoader:
    """
    class docstring placeholder
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
        train_dir = f"{self.data_dir}"
        valid_dir = f"{self.data_dir}"

        print(f"Loading training data from {train_dir}/train")
        print(f"Loading validation data from {valid_dir}/valid")

        # Define transforms
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

        # Load datasets
        image_datasets = {
            x: datasets.ImageFolder(f"{self.data_dir}/{x}", transform=data_transforms[x])
            for x in ['train', 'valid']
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
            for x in ['train', 'valid']
        }
        print("Data loaded")
        return dataloaders, image_datasets['train'].class_to_idx

class ImageProcessor:
    """
    Class docstring placeholder
    """
    @staticmethod
    def process_image(image_path):
        """
        Processes an image for use in the model by applying transformations such as 
        resizing, cropping, converting to a tensor, and normalising the pixel values.

        Args:
            image_path (str): Path to the image file to be processed.

        Returns:
            torch.Tensor: A tensor representation of the processed image, ready for input 
            to the model.
        """
        image = Image.open(image_path)
        print("Image pre-processing starting...")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
        ])
        print("Image preprocessing complete.")
        return preprocess(image)
