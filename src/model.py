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

    Methods:
        _create_model(arch, hidden_units): Builds the model with the specified architecture and hidden units.
        train(dataloaders, epochs, print_every): Trains the model and prints training/validation statistics.
        save_checkpoint(save_dir): Saves the model checkpoint.
        load_checkpoint(checkpoint_path, device_type): Loads a model checkpoint from the specified path.
        predict(image, top_k): Predicts the top K classes for a given image.
        load_category_names(json_file): Loads a JSON file that maps class indices to category names.
        map_class_to_name(class_indices, category_names): Maps predicted class indices to category names.
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
        if arch not in ['vgg16', 'resnet50']:
            raise ValueError(f"Unsupported architecture: {arch}")
        
        if not isinstance(hidden_units, int) or hidden_units <= 0:
            raise ValueError("Hidden units should be a positive integer.")
        
        if not isinstance(learning_rate, float) or learning_rate <= 0:
            raise ValueError("Learning rate should be a positive float.")
        
        if not isinstance(class_to_idx, dict):
            raise ValueError("class_to_idx should be a dictionary.")
    
        if device_type not in ['cpu', 'gpu']:
            raise ValueError(f"Unsupported device type: {device_type}")
        
        # Set the device based on device_type (cpu or gpu)
        if device_type == 'gpu':
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print("GPU selected and available. Running on GPU.")
            else:
                self.device = torch.device("cpu")
                print("GPU selected but not available. Falling back to CPU.")
        else:
            self.device = torch.device("cpu")
            print("CPU explicitly selected. Running on CPU.")

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
        # Normalise the architecture name to handle variations like 'VGG16'
        # and 'vgg16'.
        # TODO: May be redundant but currently to scared to change anything...
        #arch = arch.lower()


        #if hasattr(models, arch):
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

            print(f"Classifier model loaded with architecture {arch} and hidden units {hidden_units}.")
            return model
            
        except Exception as e:
            raise RuntimeError(f"Failed to create model: {e}")

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
            raise ValueError("Dataloaders must be a dictionary containing 'train' and 'valid' keys.")
        
        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs should be a positive integer.")
    
        print("Training commencing...")

        steps = 0
        running_loss = 0
        try:
            for epoch in range(epochs):
                print(f"Commencing epoch: {epoch+1}")
                for inputs, labels in dataloaders['train']:
                    steps += 1

                    # Move input and label tensors to the appropriate device (GPU/CPU)
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    # print("Debug: inputs and labels moved to device")

                    # Zero the gradients
                    self.optimizer.zero_grad()
                    # print("Debug: gradients zeroed")

                    # Forward pass
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    # print("Debug: forward pass")

                    # Backward pass and optimise
                    loss.backward()
                    self.optimizer.step()
                    # print("Debug: backward pass and optimise")

                    running_loss += loss.item()

                    # Perform validation every `print_every` steps
                    if steps % print_every == 0:
                        # Set model to evaluation mode
                        self.model.eval()
                        validation_loss = 0
                        accuracy = 0
                        # print("Debug: in validation if statement...")

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
                        print(f'Epoch {epoch+1}/{epochs}.. '
                            f'Train loss: {running_loss/print_every:.3f}.. '
                            f'Validation loss: {validation_loss/len(dataloaders["valid"]):.3f}.. '
                            f'Validation accuracy: {accuracy/len(dataloaders["valid"]):.3f}')

                        running_loss = 0

                        # Set model back to training mode
                        self.model.train()
            print("Training complete!")
        except RuntimeError as e:
            if 'CUDA' in str(e):
                raise RuntimeError("Training failed due to a CUDA-related issue (out of memory, etc.).")
            raise RuntimeError(f"Training failed: {e}")
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during training: {e}")

    def save_checkpoint(self, save_dir):
        """
        Saves the trained model into a checkpoint in the specified directory, including the model 
        state, architecture, class-to-index mapping, hidden units, and learning rate.

        Args:
            save_dir (str): Directory where the checkpoint file will be saved.

        Returns:
            None
        """

        print(f"Saving checkpoint to: {save_dir}")
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Save directory does not exist: {save_dir}")
        
        #Save the appropriate classifier depending on the architecture
        if self.arch == 'vgg16':
            classifier = self.model.classifier
        elif self.arch == 'resnet50':
            classifier = self.model.fc
        else:
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
            print("Checkpoint saved!")

        except Exception as e:
            raise RuntimeError(f"Failed to save checkpoint: {e}")

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
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
        
        # Determine map_location based on device_type
        if device_type == 'gpu' and torch.cuda.is_available():
            map_location = 'cuda'
        else:
            map_location = 'cpu'

        print(f"Loading checkpoint from: {checkpoint_path}")

        warnings.simplefilter("ignore")
        # TODO: Implement logging

        # Load the checkpoint from file
        try:
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            print("\nCheckpoint loaded.")

        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")

        # Extract necessary information
        class_to_idx = checkpoint['class_to_idx']
        arch = checkpoint.get('architecture', 'vgg16')  # Default to vgg16 if not found

        print(f"Loaded architecture from checkpoint: {arch}")

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
            raise ValueError("top_k must be a positive integer.") 
        
        # Set model to evaluation mode.
        try:
            self.model.eval()

            # Move image to device
            image = image.to(self.device)
            with torch.no_grad():
                output = self.model(image.unsqueeze(0))
                probs, classes = output.topk(top_k, dim=1)
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
        
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
            raise FileNotFoundError(f"JSON file not found: {json_file}")
    
        # Load mapping from class index to category names
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                category_names = json.load(f)
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Error reading JSON file: {e}")
    
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