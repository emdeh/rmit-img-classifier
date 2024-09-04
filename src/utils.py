import torch
from torchvision import datasets, transforms
from PIL import Image

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_data(self):
        train_dir = f"{self.data_dir}/flowers/train"
        valid_dir = f"{self.data_dir}/flowers/valid"

        print(f"Loading training data from {train_dir}")
        print(f"Loading validation data from {valid_dir}")

        # Define transforms
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
            ])
        }

        print("Data trainsformations complete...")

        # Load datasets
        image_datasets = {
            x: datasets.ImageFolder(f"{self.data_dir}/{x}", transform=data_transforms[x])
            for x in ['train', 'valid']
        }
        dataloaders = {
            x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
            for x in ['train', 'valid']
        }
        print("Data loaded")
        return dataloaders, image_datasets['train'].class_to_idx

class ImageProcessor:
    @staticmethod
    def process_image(image_path):
        image = Image.open(image_path)
        print("Image pre-processing starting...")
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        print("Image preprocessing complete.")
        return preprocess(image)
