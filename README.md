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