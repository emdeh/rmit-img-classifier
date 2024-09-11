"""
This module provides utility classes for handling data loading, image processing
tasks, and other system-wide utilities such as logging.

The `DataLoader` class is responsible for loading and transforming training and
validation datasets, while the `ImageProcessor` class provides methods for 
processing images to prepare them for input into deep learning models. The 
`Logger` class is used to configure and return logger instances for use in the 
application.

Classes:
    DataLoader: Handles loading and transforming datasets for training and validation.
    ImageProcessor: Provides image preprocessing functionality for input into models.
    Logger: Configures and returns logger instances for use in the application.

Dependencies:
    - logging: Provides logging functionality for the application.
    - torch: Provides deep learning framework functionality.
    - torchvision: Contains popular datasets and transforms for image data.
    - PIL: Python Imaging Library, used for opening and manipulating image files.
"""
import os
import logging
import time
import torch
from torchvision import datasets, transforms

from PIL import Image, UnidentifiedImageError


class DataLoader:
    """
    A class for loading and transforming image datasets for training and validation.

    Attributes:
        data_dir (str): The directory where the dataset is stored.
        logger (logging.Logger): Logger instance for the DataLoader class.

    Methods:
        load_data(): Loads the training and validation datasets, applies transformations,
                        and returns the dataloaders and class-to-index mapping.
    """
    def __init__(self, data_dir):
        """
        Initialises the DataLoader with the specified data directory and sets up paths 
        for loading the training and validation datasets.

        Args:
            data_dir (str): 
            Directory containing the dataset (with 'train' and 'valid' subdirectories).

        Returns:
            None
        """
        self.data_dir = data_dir
        self.logger = logging.getLogger(__name__) # Initialise logger for this class

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
        # Log start time
        start_time = time.time()
        self.logger.info("Loading data...")

        train_dir = os.path.join(self.data_dir, 'train')
        valid_dir = os.path.join(self.data_dir, 'valid')

        # Check if train and valid directories exist
        if not os.path.isdir(train_dir):
            self.logger.error("Training directory not found: %s", train_dir)
            raise FileNotFoundError(f"Training directory not found: {train_dir}")
        if not os.path.isdir(valid_dir):
            self.logger.error("Validation directory not found: %s", valid_dir)
            raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

        self.logger.info("Loading training data from %s", train_dir)
        self.logger.info("Loading validation data from %s", valid_dir)

        # Define transforms
        # TODO: See note in readme.md for more info on potential augmentations
        # on the training set to enlarge the data
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
            self.logger.info("Data transformations complete...")
        except Exception as transform_error:
            self.logger.error("Error defining data transformations: %s", transform_error)
            raise RuntimeError(
                f"Error defining data transformations: {transform_error}"
                ) from transform_error

        # Load datasets
        try:
            image_datasets = {
                x: datasets.ImageFolder(f"{self.data_dir}/{x}", transform=data_transforms[x])
                for x in ['train', 'valid']
            }
            # Check if they are empty
            for x in ['train', 'valid']:
                if len(image_datasets[x]) == 0:
                    self.logger.error(
                        "No images found in %s dataset at %s", x, os.path.join(self.data_dir, x)
                        )

                    raise ValueError(
                        f"No images found in {x} dataset at {os.path.join(self.data_dir, x)}"
                        )

        except Exception as dataset_error:
            raise RuntimeError(
                f"Error loading image datasets: {dataset_error}"
                ) from dataset_error

        # Create dataloaders
        try:
            dataloaders = {
                x: torch.utils.data.DataLoader(
                    image_datasets[x],
                    batch_size=64,
                    shuffle=True,
                    num_workers=4,
                    pin_memory=True
                    )
                for x in ['train', 'valid']
            }
            self.logger.info("Data loaders created successfully...")
        except Exception as dataloader_error:
            raise RuntimeError(
                f"Error creating dataloaders: {dataloader_error}"
                ) from dataloader_error

        # Log end time
        end_time = time.time()
        total_runtime = end_time - start_time
        self.logger.info("Data loaded. Total load time: %.2f seconds", total_runtime)
        self.logger.info("Data loaded")
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
        # Get logger for this method
        logger = logging.getLogger(__name__)

        # Log start time
        start_time = time.time()
        logger.info("Processing image: %s", image_path)

        # Check if image file exists
        if not os.path.isfile(image_path):
            logger.error("Image not found: %s", image_path)
            raise FileNotFoundError(f"Image file not found: {image_path}")

        # Open image
        try:
            image = Image.open(image_path)

        except FileNotFoundError as img_error:
            logger.error("Image file not found: %s", img_error)
            raise FileNotFoundError(
                f"Image file not found: {img_error}"
                ) from img_error

        except UnidentifiedImageError as img_frmt_error:
            logger.error("Cannot identify image file: %s", img_frmt_error)
            raise UnidentifiedImageError(
                f"Cannot identify image file: {img_frmt_error}"
                ) from img_frmt_error

        except Exception as run_error:
            logger.error("Error opening image file: %s", run_error)
            raise RuntimeError(
                f"Error opening image file: {run_error}"
                ) from run_error

        logger.info("Image pre-processing starting...")

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

            logger.info("Image pre-processing complete.")

        except Exception as img_process_error:
            raise RuntimeError(
                f"Error procesing image: {img_process_error}"
                ) from img_process_error

        # Log end time
        end_time = time.time()
        total_runtime = end_time - start_time
        logger.info(
            "Total image process time: %.4f seconds", 
            round(total_runtime)
            )

        return processed_image

class Logger:
    """
    A class for creating and configuring loggers for use in the application.
    """
    @staticmethod
    def get_logger(name, level=logging.DEBUG, log_to_file=False, log_file='app.log'):
        """
        Returns a logger instance for the specified name.
        
        :param name: Name for the logger (typically the class name)
        :param level: Logging level (default: DEBUG)
        :param log_to_file: If True, log messages will also be saved to a file (default: False)
        :param log_file: File to log messages if log_to_file is True (default: 'app.log')
        :return: Configured logger
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Create console handler
        ch = logging.StreamHandler()
        ch.setLevel(level)

        # Create a formatter with time, name, level, and message
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch.setFormatter(formatter)

        # Add the console handler if it's not already added
        if not logger.hasHandlers():
            logger.addHandler(ch)

        # Optional: log to file if log_to_file is True
        if log_to_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

        return logger
