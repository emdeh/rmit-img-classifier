import argparse
from model import ModelManager
from utils import DataLoader

def main(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Initialize DataLoader class
    data_loader = DataLoader(data_dir)
    dataloaders, class_to_idx = data_loader.load_data()

    # Initialize ModelManager class
    model_manager = ModelManager(arch, hidden_units, learning_rate, class_to_idx, gpu)

    # Train the model
    model_manager.train(dataloaders, epochs)

    # Save checkpoint
    model_manager.save_checkpoint(save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a new network on a dataset.')
    parser.add_argument('data_dir', help='Directory of training data')
    parser.add_argument('--save_dir', default='checkpoint', help='Directory to save checkpoint')
    parser.add_argument('--arch', default='vgg16', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

    args = parser.parse_args()

    main(args.data_dir, args.save_dir, args.arch, args.learning_rate, args.hidden_units, args.epochs, args.gpu)
