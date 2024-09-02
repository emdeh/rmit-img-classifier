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
import json
import torch
import numpy as numpy
from PIL import Image
from torch import nn, optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader as TorchDataLoader



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
    
    visualise_prediction(self, model, topk=5):
        Visualises the image with predicted classes and probabilities.
    """
    def __init__(self, image_path):
        pass

    def process_image(self):
        pass

    def visualise_prediction(self, model, topk=5):
        pass

class ModelTrainer:
    """
    A class to manage the training and evaluation of a deep learning model.

    This class provides methods to train a model on a dataset, validate the 
    model's performance, and evaluate the model on a test dataset.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model used for training and evaluation.
    criterion : torch.nn.Module
        The loss function used for training the model.
    optimiser : torch.optim.Optimizer
        The optimiser used for training the model.
    device : torch.device
        The device (CPU or GPU) on which the model is trained or evaluated.

    Methods
    -------
    __init__(self, model, criterion, optimiser):
        Initialises the ModelTrainer with the specified model, criterion, 
        optimiser, and device.
    
    freeze_parameters(self):
        Freezes the parameters of the pre-trained model.
    
    define_classifier(self):
        Defines and attaches a new classifier to the model.
    
    train(self, train_loader, valid_loader, epochs=5):
        Trains the model using the provided data.
    
    evaluate(self, test_loader):
        Evaluates the model on a test dataset.
    
    get_device():
        Determines and returns the device (CPU or GPU) to be used for training 
        or evaluation.
    """

    def __init__(self, model, lr=0.001):
        self.model = model
        self.device = get_device()

        # Call methods to prep model
        self.freeze_parameters()
        self.define_classifier()

        # Move model to device
        self.model = self.model.to(self.device)

        # Specify loss function and optimiser
        self.criterion = nn.NLLLoss()
        self.optimiser = optim.Adam(self.model.classifier.parameters(), lr=lr)

    def freeze_parameters(self):
        """
        Freeze the parameters of the pre-trained model to avoid 
        backpropagation through them
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def define_classifier(self):
        """
        Define a new, untrained feed-forward network as a classifier that
        replaces the pre-trained model's classifier with a new one.
        """
        classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 102),
            nn.LogSoftmax(dim=1)
        )

        self.model.classifier = classifier
    


    def train(self, train_loader, valid_loader, epochs=5):
        # Training logic here
        """
        Trains the model using the provided training and validation data loaders.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        valid_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        epochs : int
            The number of epochs to train the model.
        """
        self.model.train()
        steps = 0
        running_loss = 0
        print_every = 5

        for epoch in range(epochs):
            for inputs, labels in train_loader:
                steps += 1

                # Move input and label tensors to the appropriate device
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Zero the gradients
                self.optimiser.zero_grad()

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                self.optimiser.step()

                running_loss += loss.item()

                if steps % print_every == 0:
                    # Validate the model
                    self.model.eval()
                    accuracy = 0
                    validation_loss = 0
                    with torch.no_grad():
                        for inputs, labels in valid_loader:
                            inputs, labels = inputs.to(self.device), labels.to(self.device)
                            outputs = self.model(inputs)
                            validation_loss += self.criterion(outputs, labels).item()

                            # Calculate the accuracy
                            ps = torch.exp(outputs)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    print(f'Epoch {epoch+1}/{epochs}.. '
                          f'Train loss: {running_loss/print_every:.3f}.. '
                          f'Validation loss: {validation_loss/len(valid_loader):.3f}.. '
                          f'Validation accuracy: {accuracy/len(valid_loader):.3f}')
                    
                    running_loss = 0
                    self.model.train()

    def evaluate(self, test_loader):
        """
        Evaluates the model on a test dataset.
        """
        # Set the model to evaluation mode
        self.model.eval()

        # Initialise test accuracy and number of samples
        test_accuracy = 0
        num_samples = 0

                # Disable gradient computation as not required for validation
        with torch.no_grad():
            for inputs, labels in test_loader:
                # Move inputs and labels to the appropriate device
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Calculate probabilities
                ps = torch.exp(outputs)
                
                # Get the top class
                top_p, top_class = ps.topk(1, dim=1)
                
                # Compare predicted classes with true labels
                equals = top_class == labels.view(*top_class.shape)
                
                # Calculate accuracy for the batch and accumulate
                batch_accuracy = torch.mean(equals.type(torch.FloatTensor)).item()
                test_accuracy += batch_accuracy * inputs.size(0)
                
                # Accumulate the number of samples
                num_samples += inputs.size(0)

        # Calculate the overall accuracy on the test set
        test_accuracy = test_accuracy / num_samples
        print(f"Test Accuracy: {test_accuracy:.3f}")

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

    def save_checkpoint(self, model, optimiser, epochs, filepath='checkpoint.pth'):
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
        # Attach the class_to_idx mapping to the model
        model.class_to_idx = model.class_to_idx

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

        """
        Note: If the model was trained remotely as described in the
        remote_gpu_training.md model, then it will save on the remote instance.
        Be sure to download the model to your local machine using the following
        command: 

        scp -i "your-key.pem" ubuntu@your-remote-ip:/path/to/checkpoint.pth /path/to/save/checkpoint.pth

        or scp -r -i ... for directories.

        Replace "your-key.pem" with the path to your private key file,
        "your-remote-ip" with the IP address of your remote instance,
        "/path/to/checkpoint.pth" with the path to the model on the remote instance,
        and "/path/to/save/checkpoint.pth" with the path where you want to save the model on your local machine.

        If you don't do this, then the model will be lost when the remote 
        instance is terminated (if there is no persistent storage).
        """

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
            model = models.vgg16(pretrained=True)
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
                    "Warning: Failed to load optimiser state due to a parameter mismatch.\n" 
                    f"Reason: {e}\n"
                    "\n"
                    "This could affect training in the following ways:\n"
                    "1. If you are resuming training after a long session or with a complex model,\n" 
                    "   starting with a fresh optimiser might cause the model to lose some momentum.\n"
                    "   This could lead to slower convergence or different training dynamics.\n"
                    "\n"
                    "2. If you are fine-tuning or performing inference, this generally won't matter as much,\n" 
                    "   and the model's weights are still loaded correctly.\n"
                    "\n"
                    "A new optimiser has been reinitialised with default settings."
                )
                # Reinitialise optimiser if loading failed
                optimiser = torch.optim.Adam(model.parameters())

        # Load the number of epochs
        epochs = checkpoint['epochs'] # TODO: Shoudl this be dynamic?

        print(f"Checkpoint loaded from {filepath}")

        return model, optimiser, epochs

class ImageClassifer:
    """
    A class to manage the classification process using a trained model.

    this will house the code equiv of:
        # Load a pre-trained model
        model = models.vgg16(pretrained=True)
    """

    def __init__(self, model, label_mapping):
        self.model = model
    
    def classify(self, image):
        """
        Classifies an image using the model.
        """
        self.model.eval() # Set model to evaluation mode
        with torch.no_grad():
            output = self.model(image)
        return output


"""STANDALONE FUNCTIONS"""

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
            print("Using CPU.")
        
        return device