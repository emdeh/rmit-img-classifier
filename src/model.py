"""
This module provides the `ModelManager` class for managing deep learning model
operations, including model creation, training, checkpoint saving and loading, and
making predictions.

The `ModelManager` class allows flexibility in selecting different archiectectures, hidden units, 
learning rate, and devices (CPU/GPU) for training. It also handles saving and 
loading checkpoints, so models can be used for inferences.

Classes:
    ModelManager: Handles creating, training, saving, loading, and predicting 
    with deep learning models using various architectures like VGG16 and ResNet50.

Dependencies:
    - torch: A deep learning framework used for defining models and performing computations.
    - torchvision: Contains model architectures.
    - json: For handling JSON files, in this case class-to-category mapping.
    - warnings: For handling warnings during model loading.
"""
import os
import json
import warnings
import logging
import time

import torch
from torch import nn, optim
from torchvision import models


class ModelManager:
    """
    A class for managing the creation, training, saving, loading, and predicting 
    of models using architectures like VGG16 and ResNet50.

    The ModelManager provides methods to build a model with a specific architecture 
    and hidden units, train it on a dataset, save the trained model checkpoint, and 
    load a saved checkpoint for future use. It also supports making predictions using 
    the trained model.

    Attributes:
        arch (str): Model architecture to use (e.g., 'vgg16' or 'resnet50').
        hidden_units (int): Number of hidden units for the model's classifier.
        learning_rate (float): Learning rate for the optimiser.
        class_to_idx (dict): Mapping of class labels to indices.
        device_type (str): Device to use for training ('cpu' or 'gpu').
        logger (logging.Logger): Logger instance for logging messages.

    Methods:
        _create_model(arch, hidden_units): 
            Builds the model with the specified architecture and hidden units.
        train(dataloaders, epochs, print_every):
            Trains the model and prints training/validation statistics.
        save_checkpoint(save_dir): S
            aves the model checkpoint.
        load_checkpoint(checkpoint_path, device_type): 
            Loads a model checkpoint from the specified path.
        predict(image, top_k):
            Predicts the top K classes for a given image.
        load_category_names(json_file):
            Loads a JSON file that maps class indices to category names.
        map_class_to_name(class_indices, category_names):
            Maps predicted class indices to category names.
    """
    def __init__(self, arch, hidden_units, learning_rate, class_to_idx, device_type):
        """
        Initialises the ModelManager with the specified architecture, hidden units, learning rate,
        class-to-index mapping, and device type (CPU or GPU). Also sets up the model, criterion, 
        and optimiser for training.

        Args:
            arch (str): Model architecture to use (e.g., 'vgg16' or 'resnet50').
            hidden_units (int): Number of hidden units for the model's classifier.
            learning_rate (float): Learning rate for the optimiser.
            class_to_idx (dict): Mapping of class labels to indices.
            device_type (str): Device to use for training ('cpu' or 'gpu').

        Returns:
            None
        """
        # Assign logger to class attribute
        self.logger = logging.getLogger(__name__)

        if arch not in ['vgg16', 'resnet50']:
            self.logger.error("Unsupported architecture: %s", arch)
            raise ValueError(f"Unsupported architecture: {arch}")

        if not isinstance(hidden_units, int) or hidden_units <= 0:
            self.logger.error("Hidden units should be a positive integer.")
            raise ValueError("Hidden units should be a positive integer.")

        if not isinstance(learning_rate, float) or learning_rate <= 0:
            self.logger.error("Learning rate should be a positive float.")
            raise ValueError("Learning rate should be a positive float.")

        if not isinstance(class_to_idx, dict):
            self.logger.error("class_to_idx should be a dictionary.")
            raise ValueError("class_to_idx should be a dictionary.")

        if device_type not in ['cpu', 'gpu']:
            self.logger.error("Unsupported device type: %s", device_type)
            raise ValueError(f"Unsupported device type: {device_type}")

        # Set the device based on device_type (cpu or gpu)
        if device_type == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                self.logger.info("GPU selected and available. Running on GPU.")
            else:
                self.device = torch.device("cpu")
                self.logger.warning("GPU selected but not available. Falling back to CPU.")
        else:
            self.device = torch.device("cpu")
            self.logger.info("CPU explicitly selected. Running on CPU")

        # Model setup
        self.class_to_idx = class_to_idx
        self.arch = arch
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = self._create_model(arch, hidden_units)
        self.criterion = nn.NLLLoss()

        # Setup optimiser based on the architecture selected
        if arch == 'vgg16':
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        elif arch == 'resnet50':
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def _create_model(self, arch, hidden_units):
        """
        Creates and returns a deep learning model based on the specified architecture 
        and hidden units, with pretrained weights and a new classifier for VGG16 or ResNet50.

        Args:
            arch (str): Model architecture to use (e.g., 'vgg16' or 'resnet50').
            hidden_units (int): Number of hidden units for the model's classifier.

        Returns:
            torch.nn.Module: The model with the updated classifier.
        """
        # Log start time
        start_time = time.time()
        self.logger.info("Loading model...")

        # TODO: Potential redundant code that could be refactored.
        try:
            if arch == 'vgg16':
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            elif arch == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

            # Freeze parameters to stop backpropagation
            for param in model.parameters():
                param.requires_grad = False

            # Set up classifier based on architecture
            if arch == 'vgg16':

                # Create a new classifier for VGG16
                classifier = nn.Sequential(
                    nn.Linear(model.classifier[0].in_features, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_units, len(self.class_to_idx)),
                    nn.LogSoftmax(dim=1)
                )
                model.classifier = classifier

            elif arch == 'resnet50':
                # ResNet50 uses `fc` instead of `classifier`
                model.fc = nn.Sequential(
                    nn.Linear(model.fc.in_features, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_units, len(self.class_to_idx)),
                    nn.LogSoftmax(dim=1)
                )

            self.logger.info(
                "Classifier model loaded with architecture %s and hidden units %d."
                , arch, hidden_units)
            return model

        except Exception as model_error:
            self.logger.error("Failed to create model: %s", model_error)
            raise RuntimeError(
                f"Failed to create model: {model_error}"
                ) from model_error

    def train(self, dataloaders, epochs, print_every=5):
        """
        Trains the model using the specified dataloaders for the given number of epochs. 
        The function will also periodically evaluate the model on the validation set and prints 
        training/validation statistics as it progresses through epochs.

        Args:
            dataloaders (dict): Dictionary containing 'train' and 'valid' dataloaders.
            epochs (int): Number of training epochs.
            print_every (int): Number of steps between printing training/validation statistics.

        Returns:
            None
        """
        if not isinstance(dataloaders, dict) or 'train' not in dataloaders or 'valid' not in dataloaders:
            self.logger.error(
                "Dataloaders must be a dictionary containing 'train' and 'valid' keys.")
            raise ValueError(
                "Dataloaders must be a dictionary containing 'train' and 'valid' keys.")

        if not isinstance(epochs, int) or epochs <= 0:
            self.logger.error("Epochs should be a positive integer.")
            raise ValueError("Epochs should be a positive integer.")

        self.logger.info("Training commencing...")

        steps = 0
        running_loss = 0
        try:
            for epoch in range(epochs):
                self.logger.info("Commencing epoch: %d", epoch+1)
                for inputs, labels in dataloaders['train']:
                    steps += 1

                    # Move input and label tensors to the appropriate device (GPU/CPU)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    # Zero the gradients
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)

                    # Backward pass and optimise
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()

                    # Perform validation every `print_every` steps
                    if steps % print_every == 0:
                        # Set model to evaluation mode
                        self.model.eval()
                        validation_loss = 0
                        accuracy = 0

                        # Disable gradient calculation for validation
                        with torch.no_grad():
                            for inputs, labels in dataloaders['valid']:
                                inputs, labels = inputs.to(self.device), labels.to(self.device)
                                outputs = self.model(inputs)
                                loss = self.criterion(outputs, labels)
                                validation_loss += loss.item()

                                # Calculate accuracy
                                ps = torch.exp(outputs)
                                top_p, top_class = ps.topk(1, dim=1)
                                equals = top_class == labels.view(*top_class.shape)
                                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                                # print("Debug: Calculated accuracy")

                        # Print statistics
                        self.logger.info(
                            "Epoch %d/%d.. "
                            "Train loss: %.3f.. "
                            "Validation loss: %.3f.. "
                            "Validation accuracy: %.3f"
                            , epoch+1, epochs, running_loss/print_every, validation_loss/len(dataloaders['valid']), accuracy/len(dataloaders['valid']))

                        running_loss = 0

                        # Set model back to training mode
                        self.model.train()
            self.logger.info("Training complete!")

        except RuntimeError as cuda_error:
            if 'CUDA' in str(cuda_error):
                self.logger.error(
                    "Training failed due to a CUDA-related issue (out of memory, etc.).")
                raise RuntimeError(
                    "Training failed due to a CUDA-related issue (out of memory, etc.)."
                    ) from cuda_error

            self.logger.error("Training failed: %s", cuda_error)
            raise RuntimeError(f"Training failed: {cuda_error}") from cuda_error

        except Exception as general_error:
            self.logger.error("An unexpected error occurred during training: %s", general_error)
            raise RuntimeError(f"An unexpected error occurred during training: {general_error}"
            ) from general_error


    def save_checkpoint(self, save_dir):
        """
        Saves the trained model into a checkpoint in the specified directory, including the model 
        state, architecture, class-to-index mapping, hidden units, and learning rate.

        Args:
            save_dir (str): Directory where the checkpoint file will be saved.

        Returns:
            None
        """

        self.logger.info("Saving checkpoint to: %s", save_dir)
        if not os.path.isdir(save_dir):
            self.logger.error("Save directory does not exist: %s", save_dir)
            raise FileNotFoundError(f"Save directory does not exist: {save_dir}")

        #Save the appropriate classifier depending on the architecture
        if self.arch == 'vgg16':
            classifier = self.model.classifier
        elif self.arch == 'resnet50':
            classifier = self.model.fc
        else:
            self.logger.error("Architecture %s not supported for saving checkpoints.", self.arch)
            raise ValueError(f"Architecture {self.arch} not supported for saving checkpoints.")

        try:
        # Save checkpoint with necessary metadata
            checkpoint = {
                'state_dict': self.model.state_dict(),
                'class_to_idx': self.class_to_idx,
                'architecture': self.arch,  # Save the architecture correctly
                'hidden_units': self.hidden_units,
                'learning_rate': self.learning_rate,
                'classifier': classifier
            }

            # Save checkpoint
            torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
            self.logger.info("Checkpoint saved!")

        except Exception as save_error:
            self.logger.error("Failed to save checkpoint: %s", save_error)
            raise RuntimeError(
                f"Failed to save checkpoint: {save_error}"
                ) from save_error

    @classmethod
    # Decorator needed because the method is invoked on the class itself instead
    # of an object of the class. This allows it to return a new instance of the class.

    def load_checkpoint(cls, checkpoint_path, device_type):
        # TODO: This needs work...
        """
        Loads a model checkpoint from the specified path and restores the model's state, 
        architecture, and other parameters. Also loads the class-to-index mapping.

        Args:
            checkpoint_path (str): Path to the model checkpoint file.
            device_type (str): Device to load the model on ('cpu' or 'gpu').

        Returns:
            ModelManager: An instance of the ModelManager with the loaded model.
        """

                # Get logger for this method
        logger = logging.getLogger(__name__)

        if not os.path.isfile(checkpoint_path):
            logger.error("Checkpoint file not found: %s", checkpoint_path)
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

        # Determine map_location based on device_type
        if device_type == 'gpu' and torch.cuda.is_available():
            map_location = 'cuda'
        else:
            map_location = 'cpu'

        logger.info("Loading checkpoint from: %s", checkpoint_path)

        warnings.simplefilter("ignore")
        # TODO: Implement logging

        # Load the checkpoint from file
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            print("\nCheckpoint loaded.")

        except Exception as checkpoint_error:
            logger.error("Failed to load checkpoint: %s", checkpoint_path)
            raise RuntimeError(f"Failed to load checkpoint: {checkpoint_error}"
            ) from checkpoint_error

        # Extract necessary information
        class_to_idx = checkpoint['class_to_idx']
        arch = checkpoint.get('architecture', 'vgg16')  # Default to vgg16 if not found

        logger.info("Loaded architecture from checkpoint: %s", arch)

        # Normalise the architecture name
        # TODO: May not be required after all.
        if arch.lower() == 'vgg':
            arch = 'vgg16'

        hidden_units = checkpoint.get('hidden_units', 4096)
        learning_rate = checkpoint.get('learning_rate', 0.001)

        # Create a new ModelManager instance
        model_manager = cls(arch, hidden_units, learning_rate, class_to_idx, device_type)

        # Load the state_dict from the checkpoint
        model_manager.model.load_state_dict(checkpoint['state_dict'])

        return model_manager

    def predict(self, image, top_k):
        """
        Predicts the top K classes for the given image using the trained model. Returns 
        the probabilities and class indices of the top K predicted classes.

        Args:
            image (torch.Tensor): Preprocessed image tensor for prediction.
            top_k (int): Number of top predictions to return.

        Returns:
            tuple: A tuple containing:
                - probs (numpy.ndarray): Probabilities of the top K predicted classes.
                - classes (numpy.ndarray): Indices of the top K predicted classes.
        """
        if not isinstance(top_k, int) or top_k <= 0:
            self.logger.error("top_k must be a positive integer.")
            raise ValueError("top_k must be a positive integer.")

        # Set model to evaluation mode.
        try:
            self.model.eval()

            # Move image to device
            image = image.to(self.device)
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))
                probs, classes = output.topk(top_k, dim=1)
        except Exception as predict_error:
            self.logger.error("Prediction failed: %s", predict_error)
            raise RuntimeError(f"Prediction failed: {predict_error}"
            ) from predict_error

        return probs.exp().cpu().numpy()[0], classes.cpu().numpy()[0]

    def load_category_names(self, json_file):
        """
        Loads a JSON file that maps class indices to category names.

        Args:
            json_file (str): Path to the JSON file containing class-to-name mappings.

        Returns:
            dict: A dictionary mapping class indices to category names.
        """
        if not os.path.isfile(json_file):
            self.logger.error("JSON file not found: %s", json_file)
            raise FileNotFoundError(f"JSON file not found: {json_file}")

        # Load mapping from class index to category names
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                category_names = json.load(f, strict=False)
        except json.JSONDecodeError as json_load_error:
            self.logger.error("Error reading JSON file: %s", json_load_error)
            raise RuntimeError(f"Error reading JSON file: {json_load_error}"
            ) from json_load_error

        return category_names

    def map_class_to_name(self, class_indices, category_names):
        # TODO : Error handling.
        """
        Maps the predicted class indices to their corresponding category names using 
        the provided category names dictionary.

        Args:
            class_indices (list): List of predicted class indices.
            category_names (dict): Dictionary mapping class indices to category names.

        Returns:
            list: A list of category names corresponding to the predicted class indices.
        """

        # Invert class_to_idx to get idx_to_class mapping
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}


        # Map the predicted class indices to the actual category names
        class_names = [category_names[idx_to_class[i]] for i in class_indices]

        return class_names
