# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Geting started

To setup the environment, do the following:

```bash
wget https://raw.githubusercontent.com/emdeh/rmit-img-classifier/main/scripts/setup.sh
```

Then make the file executable with:
```bash
chmod +x setup.sh
```

Then run the file with source:
```bash
source setup.sh
```

>**There is no need to clone the repo, this will be done for you**

## Usage

Once the environment is setup, activate it with:
```bash
conda activate img-classifier
```

To train the image classifier, run the following command:
```bash
        # Example usage:
            python src/train.py --data_dir data/ --save_dir checkpoints/ --arch vgg16 --learning_rate 0.02 --hidden_units 4096 --epochs 5 --device gpu

        # For help:
            python src/train.py -h
```

To predict the class of an image using the trained model, run the following command:
```bash
        # Example usage:
            python src/predict.py --image_path data/valid/12/image_03997.jpg --checkpoint checkpoints/checkpoint.pth --top_k 3 --category_names cat_to_name.json --device cpu

        # For help:
            python src/predict.py -h
```

## Other information

If you train on a remote GPU and if you want to monitor it, use the following command:
```bash
watch -n 1 nvidia-smi
```

## Project structure

```bash
Image-Classifier/
├── assets/ # Not required for general usage
│   ├── Flowers.png 
│   └── inference_example.png
├── data/
│   ├── train/ # Install with wget from setup.sh
│   ├── valid/ # Install with wget from setup.sh
│   ├── test/ # Install with wget from setup.sh
│   └── cat_to_name.json
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── predict.py
│   ├── utils.py
│   ├── data_loader.py
│   └── model.py
├── checkpoints/ # Created with setup.sh - .gitignored due to file size.
│   └── [Saved checkpoint files]
├── scripts/
│   ├── update.sh # Used to update /src/* of remote implementations from GitHub.
│   └── setup.sh # Used to setup the environment - see above
├── env.yaml # Called by setup.sh, contains dependencies
├── README.md
├── Image Classifier Project.ipynb # Initial development; not needed for general usage.
├── Documentation # Other documentation.
└── .gitignore
```


# TO-DO LIST
- Update prints with logging statements.
    - predict.py done.
    
- Update to distribute workload across multiple remote GPUs
- Review TODOs
- see baseline approach/cheatsheet including links to case studies for next CNN model (https://cs231n.github.io/convolutional-networks/)
- See [pytorch computer vision tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)(opens in a new tab) for additional insights, and optional features like [learning rate scheduler](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)(opens in a new tab) 💡

## Other notes
-  Finding the correct hyperparameters can be a time-consuming undertaking, there are several general approaches towards trying to have an optimised approach, see this [blogpost](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/)(opens in a new tab) that gives an overview of such.

## Note on augmentations (See TODO in utils.py ~line 84)
See https://github.com/aleju/imgaug for what is possible in regard to augmentations and see https://pytorch.org/vision/stable/transforms.html?highlight=augmentations which can be freely chosen from

1. Increased Data Diversity: Augmentations help create a more diverse and representative dataset. By applying various transformations to the original data, such as adding noise, changing word order, or replacing words with synonyms, we can generate new examples that capture different variation

3. Regularization: Augmentations act as a form of regularization by introducing noise or perturbations to the training data. This helps prevent overfitting, where the model becomes too specialized to the training data and performs poorly on unseen examples. By exposing the model to different variations of the data, augmentations encourage it to learn more robust and generalisable representations

3. Data Scarcity: In some cases, obtaining a large labeled dataset can be challenging or expensive. Augmentations can help alleviate the problem of data scarcity by generating additional training examples from the limited available data. This allows the model to learn from a larger and more diverse set of examples, leading to improved performance.

4. Addressing Biases: Augmentations can also be used to address biases present in the training data. By applying transformations that modify sensitive attributes or introduce variations in the data, we can reduce the impact of biases and ensure fairer and more unbiased predictions.

