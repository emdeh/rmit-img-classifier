'''
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

'''

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
    def __init__(self, data_dir):
        pass

    def load_data(self):
        pass

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

class CheckpointManager:
    """
    A class to manage the saving and loading of model checkpoints.

    This class provides methods to save a model's state to a file and load it 
    back into the model, enabling training to resume or evaluation at a later 
    time.

    Attributes
    ----------
    checkpoint_dir : str
        Directory where checkpoints are saved or loaded from.

    Methods
    -------
    save(self, model, save_path):
        Saves the model's state to a checkpoint file.
    
    load(self, checkpoint_path):
        Loads the model's state from a checkpoint file.
    """
    
    def __init__(self, checkpoint_dir):
        pass

    def save(self, model, save_path):
        pass

    def load(self, checkpoint_path):
        pass
