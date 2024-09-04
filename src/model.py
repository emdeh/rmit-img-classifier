import torch
from torch import nn, optim
from torchvision import models
import json

class ModelManager:
    def __init__(self, arch, hidden_units, learning_rate, class_to_idx, gpu):
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.class_to_idx = class_to_idx  # Class index mapping
        self.model = self._create_model(arch, hidden_units)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        self.model.to(self.device)

    def _create_model(self, arch, hidden_units):
        # Load a pre-trained model
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
        return model

    def train(self, dataloaders, epochs):
        for epoch in range(epochs):
            self.model.train()
            running_loss = 0
            for inputs, labels in dataloaders['train']:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloaders['train'])}")

    def save_checkpoint(self, save_dir):
        # Save checkpoint with necessary metadata
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.class_to_idx,
            'architecture': self.model.__class__.__name__,
            'classifier': self.model.classifier
        }
        torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

    @classmethod
    def load_checkpoint(cls, checkpoint_path, gpu):
        # Load a checkpoint from a file
        checkpoint = torch.load(checkpoint_path)
        class_to_idx = checkpoint['class_to_idx']
        model_manager = cls('vgg16', 512, 0.001, class_to_idx, gpu)
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
