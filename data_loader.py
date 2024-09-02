"""
Module: data_loader.py
This module provides a DataLoader class for handling data loading and 
transformations for training and evaluating a model. It includes methods for 
loading the data, applying transformations, and creating data loaders for 
training, validation, and testing.

Classes:
- DataLoader: A class to handle data loading and transformations.

Methods:
- __init__(self, data_dir, label_map_path): Initializes the DataLoader with 
    the specified data directory and label mapping file path.

- load_data(self): Loads the data, applies transformations, and returns data 
    loaders.

- load_label_mapping(self): Loads the mapping of class indices to class labels.

Attributes:
- train_loader: DataLoader for the training dataset.
- valid_loader: DataLoader for the validation dataset.
- test_loader: DataLoader for the test dataset.
- class_to_idx: Mapping of class names to indices based on the training dataset.

"""

import json
from torch.utils.data import DataLoader as TorchDataLoader
from torchvision import datasets, transforms

class DataLoader:
    """
    A class to handle data loading and transformations for training and 
    evaluating a model.

    This class is responsible for applying transformations to the dataset,
    loading the dataset, and creating data loaders for training, validation, 
    and testing.

    Attributes
    ----------
    train_loader : torch.utils.data.DataLoader
        DataLoader for the training dataset.
    valid_loader : torch.utils.data.DataLoader
        DataLoader for the validation dataset.
    test_loader : torch.utils.data.DataLoader
        DataLoader for the test dataset.
    class_to_idx : dict
        Mapping of class names to indices based on the training dataset.

    Methods
    -------
    __init__(self, data_dir):
        Initializes the DataLoader with the specified data directory.
    
    load_data(self):
        Loads the data, applies transformations, and returns data loaders.
    """
    def __init__(self, data_dir, label_map_path):
        self.data_dir = data_dir # TODO: Will need a way to edit this if training remote vs locally. Was /home/ubuntu/flowers for remote
        self.train_dir = f"{data_dir}/train"
        self.valid_dir = f"{data_dir}/valid"
        self.test_dir = f"{data_dir}/test"
        self.label_map_path = label_map_path

        # Placeholder for data loaders and class_to_idx
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None
        self.class_to_idx = None
        self.cat_to_name = self.load_label_mapping()

        # Loaad the data
        self.load_data()

    def load_data(self):
        """
        Loads data, applies transforms, validations and tests.
        """
        # Define your transforms for the training, validation, and testing sets
        data_transforms = {
            'train': transforms.Compose([
                transforms.RandomRotation(30),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'valid': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }

        # Load the datasets with ImageFolder
        image_datasets = {
            'train': datasets.ImageFolder(self.train_dir, transform=data_transforms['train']),
            'valid': datasets.ImageFolder(self.valid_dir, transform=data_transforms['valid']),
            'test': datasets.ImageFolder(self.test_dir, transform=data_transforms['test'])
        }

        # Using the image datasets and the transforms, define the dataloaders
        self.train_loader = TorchDataLoader(image_datasets['train'], batch_size=64, shuffle=True)
        self.valid_loader = TorchDataLoader(image_datasets['valid'], batch_size=64)
        self.test_loader = TorchDataLoader(image_datasets['test'], batch_size=64)
        # Save the class_to_idx mapping
        self.class_to_idx = image_datasets['train'].class_to_idx

    def load_label_mapping(self):
        """
        Loads the mapping of class indices to class labels.
        """
        try:
            with open(self.label_map_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Label mapping file not found at {self.label_map_path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in label mapping file at {self.label_map_path}") from exc
