import torch
from torch import nn, optim
from torchvision import models
import json
import warnings
from utils import setup_logging
import logging

class ModelManager:
    def __init__(self, arch, hidden_units, learning_rate, class_to_idx, device_type):
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
        self.class_to_idx = class_to_idx  # Class index mapping
        self.arch = arch
        self.hidden_units = hidden_units
        self.learning_rate = learning_rate
        self.model = self._create_model(arch, hidden_units)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def _create_model(self, arch, hidden_units):
        # For newer versions of torchvision
        if hasattr(models, arch):
            # Check if weights need to be loaded explicitly for newer versions
            if arch == 'vgg16':
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
            elif arch == 'resnet50':
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                raise ValueError(f"Architecture {arch} not supported or weights not available.")
        else:
            # For older versions of torchvision that use 'pretrained=True' instead of weights
            model = getattr(models, arch)(pretrained=True)

        
        # Freeze parameters so we don't backpropagate through them
        for param in model.parameters():
            param.requires_grad = False

        # Create a new classifier
        classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, len(self.class_to_idx)),  # Use self.class_to_idx for size
            nn.LogSoftmax(dim=1)
        )

        model.classifier = classifier
        print("Classifier model loaded with following hyperparameters:")
        print(f"{classifier}")
        return model

    def train(self, dataloaders, epochs, print_every=5):
        print("Training commencing...")
        steps = 0
        running_loss = 0

        for epoch in range(epochs):
            print(f"Commencing epoch: {epoch+1}")
            for inputs, labels in dataloaders['train']:
                steps += 1

                # Move input and label tensors to the appropriate device (GPU/CPU)
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                #print("Debug: inputs and labels moved to device")

                # Zero the gradients
                self.optimizer.zero_grad()
                #print("Debug: gradients zeroed")

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                #print("Debug: forward pass")

                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                #print("Debug: backward pass and optimise")

                running_loss += loss.item()

                # Perform validation every `print_every` steps
                if steps % print_every == 0:
                    # Set model to evaluation mode
                    self.model.eval()
                    validation_loss = 0
                    accuracy = 0
                    #print("Debug: in validation if statement...")

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
                            #print("Debug: Calculated accuracy")

                    # Print statistics
                    print(f'Epoch {epoch+1}/{epochs}.. '
                          f'Train loss: {running_loss/print_every:.3f}.. '
                          f'Validation loss: {validation_loss/len(dataloaders["valid"]):.3f}.. '
                          f'Validation accuracy: {accuracy/len(dataloaders["valid"]):.3f}')
                    
                    running_loss = 0

                    # Set model back to training mode
                    self.model.train()
        print("Training complete!")

    def save_checkpoint(self, save_dir):
        print(f"Saving checkpoint to: {save_dir}")
        # Save checkpoint with necessary metadata
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx,
            'architecture': self.model.__class__.__name__,
            'hidden_units': self.hidden_units,
            'learning_rate': self.learning_rate,
            'classifier': self.model.classifier
        }
        torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
        print("Checkpoint saved!")

    @classmethod
    def load_checkpoint(cls, checkpoint_path, device_type):
        # Setup logger to explain message
        logger = setup_logging()

        # Determine map_location based on device_type
        if device_type == 'gpu' and torch.cuda.is_available():
            map_location = 'cuda'
        else:
            map_location = 'cpu'

        with warnings.catch_warnings(record=True,) as w:
            warnings.simplefilter("ignore")

            # Load the checkpoint from file
            checkpoint = torch.load(checkpoint_path, map_location=map_location)

        logger.info("\nCheckpoint loaded.")

        # Extract necessary information
        class_to_idx = checkpoint['class_to_idx']
        arch = checkpoint.get('architecture', 'vgg16')
        hidden_units = checkpoint.get('hidden_units', 4096)
        learning_rate = checkpoint.get('learning_rate', 0.001)

        # Create a new ModelManager instance
        model_manager = cls(arch, hidden_units, learning_rate, class_to_idx, device_type)

        # Load the state_dict from the checkpoint
        model_manager.model.load_state_dict(checkpoint['state_dict'])

        return model_manager

    def predict(self, image, top_k):
        self.model.eval()
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            probs, classes = output.topk(top_k, dim=1)
            return probs.exp().cpu().numpy()[0], classes.cpu().numpy()[0]

    def load_category_names(self, json_file):
        # Load mapping from class index to category names
        with open(json_file, 'r') as f:
            return json.load(f)

    def map_class_to_name(self, class_indices, category_names):
        # Invert class_to_idx to get idx_to_class mapping
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Map the predicted class indices to the actual category names
        class_names = [category_names[idx_to_class[i]] for i in class_indices]
        
        return class_names