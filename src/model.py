"""
Module docstring placeholder
"""
import json
import warnings
import torch
from torch import nn, optim
from torchvision import models


class ModelManager:
    """
    Class docstring placeholder
    """
    def __init__(self, arch, hidden_units, learning_rate, class_to_idx, device_type):
        """
        Class docstring placeholder
        """
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
        # Setup optimizer based on the architecture
        if arch == 'vgg16':
            self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        elif arch == 'resnet50':
            self.optimizer = optim.Adam(self.model.fc.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def _create_model(self, arch, hidden_units):
        """
        Function docstring placeholder
        """
        # Normalize the architecture name to handle variations like 'VGG16'
        # and 'vgg16'
        arch = arch.lower()

        # For newer versions of torchvision
        if hasattr(models, arch):
            # Check if weights need to be loaded explicitly for newer versions
            if arch == 'vgg16':
                model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
                # Freeze parameters so we don't backpropagate through them
                for param in model.parameters():
                    param.requires_grad = False

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
                model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                # Freeze parameters so we don't backpropagate through them
                for param in model.parameters():
                    param.requires_grad = False

                # ResNet50 uses `fc` instead of `classifier`
                model.fc = nn.Sequential(
                    nn.Linear(model.fc.in_features, hidden_units),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_units, len(self.class_to_idx)),
                    nn.LogSoftmax(dim=1)
                )
            else:
                raise ValueError(f"Architecture {arch} not supported or weights not available.")
        else:
            # For older versions of torchvision that use 'pretrained=True'
            # instead of weights
            model = getattr(models, arch)(pretrained=True)

        print(f"Classifier model loaded with architecture {arch} and hidden units {hidden_units}.")

        return model


    def train(self, dataloaders, epochs, print_every=5):
        """
        Function docstring placeholder
        """
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
                            top_class = ps.topk(1, dim=1)
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
        """
        Function docstring placeholder
        """
        print(f"Saving checkpoint to: {save_dir}")

        # Save the appropriate classifier depending on the architecture
        if self.arch == 'vgg16':
            classifier = self.model.classifier
        elif self.arch == 'resnet50':
            classifier = self.model.fc
        else:
            raise ValueError(f"Architecture {self.arch} not supported for saving checkpoints.")

        # Save checkpoint with necessary metadata
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx,
            'architecture': self.arch,  # Save the architecture correctly
            'hidden_units': self.hidden_units,
            'learning_rate': self.learning_rate,
            'classifier': classifier  # Save the correct classifier
        }

        torch.save(checkpoint, f"{save_dir}/checkpoint.pth")
        print("Checkpoint saved!")


    @classmethod
    def load_checkpoint(cls, checkpoint_path, device_type):
        """
        Function docstring placeholder
        """

        # Determine map_location based on device_type
        if device_type == 'gpu' and torch.cuda.is_available():
            map_location = 'cuda'
        else:
            map_location = 'cpu'

        print(f"Loading checkpoint from: {checkpoint_path}")

        warnings.simplefilter("ignore")
        # TODO: Implement logging

        # Load the checkpoint from file
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

        print("\nCheckpoint loaded.")

        # Extract necessary information
        class_to_idx = checkpoint['class_to_idx']
        arch = checkpoint.get('architecture', 'vgg16')  # Default to vgg16 if not found

        # Debugging step: Print out the architecture to make sure it is correct
        print(f"Loaded architecture from checkpoint: {arch}")

        # Normalize the architecture name
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
        Function docstring placeholder
        """
        self.model.eval()
        image = image.to(self.device)
        with torch.no_grad():
            output = self.model(image.unsqueeze(0))
            probs, classes = output.topk(top_k, dim=1)
            return probs.exp().cpu().numpy()[0], classes.cpu().numpy()[0]

    def load_category_names(self, json_file):
        """
        Function docstring placeholder
        """
        # Load mapping from class index to category names
        with open(json_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def map_class_to_name(self, class_indices, category_names):
        """
        Function docstring placeholder
        """
        # Invert class_to_idx to get idx_to_class mapping
        idx_to_class = {v: k for k, v in self.class_to_idx.items()}

        # Map the predicted class indices to the actual category names
        class_names = [category_names[idx_to_class[i]] for i in class_indices]

        return class_names
