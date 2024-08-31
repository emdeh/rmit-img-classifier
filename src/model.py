'''
This file is for functions and classes relating to the model.
'''

class ImageClassifier:
    """
    A class to represent an image classifier model.

    This class encapsulates the creation, training, and inference of a deep 
    learning model for image classification.

    Attributes
    ----------
    model : torch.nn.Module
        The neural network model used for image classification.
    device : torch.device
        The device (CPU or GPU) on which the model is trained or evaluated.
    criterion : torch.nn.Module
        The loss function used for training the model.
    optimizer : torch.optim.Optimizer
        The optimizer used for training the model.

    Methods
    -------
    __init__(self, architecture='vgg16', hidden_units=4096, output_size=102):
        Initialises the ImageClassifier with the specified architecture.
    
    train(self, train_loader, valid_loader, epochs=5):
        Trains the model using the provided data.
    
    save_checkpoint(self, save_path):
        Saves the model's state to a checkpoint file.
    
    load_checkpoint(self, checkpoint_path):
        Loads a model from a checkpoint file.
    
    predict(self, image_path):
        Predicts the class of an input image.
    """
    
    def __init__(self, architecture='vgg16', hidden_units=4096, output_size=102):
        pass

    def train(self, train_loader, valid_loader, epochs=5):
        pass

    def save_checkpoint(self, save_path):
        pass

    def load_checkpoint(self, checkpoint_path):
        pass

    def predict(self, image_path):
        pass
