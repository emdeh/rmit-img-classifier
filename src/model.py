import torch
from torch import nn, optim
from torchvision import models
import json

class ModelManager:
    def __init__(self, arch, hidden_units, learning_rate, class_to_idx, gpu):
        self.device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
        self.model = self._create_model(arch, hidden_units)
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        self.model.class_to_idx = class_to_idx
        self.model.to(self.device)

    def _create_model(self, arch, hidden_units):
        model = getattr(models, arch)(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        # Create classifier
        classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_units, len(self.model.class_to_idx)),
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
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'class_to_idx': self.model.class_to_idx
        }
        torch.save(checkpoint, f"{save_dir}/checkpoint.pth")

    @classmethod
    def load_checkpoint(cls, checkpoint_path, gpu):
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
        with open(json_file, 'r') as f:
            return json.load(f)
