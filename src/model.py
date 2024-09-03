'''
This file is for functions and classes relating to the model.
'''

import torch
from torch import nn, optim
from utils import ImageProcessor, get_device

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

    def __init__(self, model, data_loader, hidden_units=4096, epochs=5, learning_rate=0.001, running_loss=0, device='cpu'):
        self.model = model
        self.hidden_units = hidden_units
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.running_loss = running_loss

        self.device = get_device()

        # Call methods to prep model
        self.freeze_parameters()
        self.define_classifier()

        # Move model to device
        self.model = self.model.to(self.device)

        # Specify loss function and optimiser
        self.criterion = nn.NLLLoss()
        self.optimiser = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)

        # Set class_to_idx from the DataLoader instance
        self.model.class_to_idx = data_loader.class_to_idx

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
        input_size = self.model.classifier[0].in_features
        classifier = nn.Sequential(
            nn.Linear(input_size, self.hidden_units),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.hidden_units, 102),
            nn.LogSoftmax(dim=1)
        )

        self.model.classifier = classifier

    def train(self, train_loader, valid_loader):
        """
        Trains the model using the provided training and validation data loaders.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training dataset.
        valid_loader : torch.utils.data.DataLoader
            DataLoader for the validation dataset.
        """
        self.model.train()

        steps = 0
        running_loss = 0
        print_every = 5

        for epoch in range(self.epochs):
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

                    print(f'Epoch {epoch+1}/{self.epochs}.. '
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
                top_class = ps.topk(1, dim=1)
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

class ImageClassifier:
    """
    A class to manage the classification process using a trained model.

    Attributes
    ----------
    model : torch.nn.Module
        The trained PyTorch model used for classification.
    
    Methods
    -------
    predict(self, image_path, topk=5):
        Predicts the top K classes for a given image.
    
    sanity_check(self, image_paths, cat_to_name, topk=5):
        Visualizes the predictions for a list of images.
    """

    def __init__(self, model, label_mapping, device='cpu'):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.class_to_idx = label_mapping

    def predict(self, image_path, topk=5):
        """
        Predict the class (or classes) of an image using a trained deep learning model.
        
        Args:
        - image_path (str): Path to the image file.
        - topk (int): Number of top most likely classes to return.
        
        Returns:
        - probs (list): Probabilities of the top K classes.
        - classes (list): Corresponding classes for the top K probabilities.
        """
        # Process the image
        image_processor = ImageProcessor()
        np_image = image_processor.process_image(image_path)

        # Convert to PyTorch tensor and add batch dimension
        image_tensor = torch.from_numpy(np_image).unsqueeze(0).float().to(self.device)

        # Move tensor to the same device as the model
        image_tensor = image_tensor.to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Disable gradients for inference
        with torch.no_grad():
            output = self.model.forward(image_tensor)

        # Apply softmax to get probabilities
        probs = torch.softmax(output, dim=1)

        # Get the top K probabilities and classes
        top_probs, top_indices = torch.topk(probs, topk)
        top_probs = top_probs.cpu().numpy().flatten()
        top_indices = top_indices.cpu().numpy().flatten()

        # Convert indices to classes
        idx_to_class = {v: k for k, v in self.model.class_to_idx.items()}
        print(f"idx_to_class: {idx_to_class}")  # Debugging
        print(f"top_indices: {top_indices}")    # Debugging

        try:
            top_classes = [idx_to_class[idx] for idx in top_indices]
        except KeyError as e:
            print(f"Error: Index {e} not found in idx_to_class mapping.")
            raise

        return top_probs, top_classes

