# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Geting started

To setup the environment, run one of the setup scripts that is appropriate for your environment.

- For instructions on how you can use a remote GPU, please see [remote-use.md](/path/to/remote-training.md).
- For instructions on how to use a local GPU (or CPU), please see [local-use.md](/path/to/local-use.md).

## Usage

To train the image classifier, run the following command:
```bash
        Example usage:
            python train.py --data_dir /path/to/data --save_dir /path/to/save_dir --arch resnet50 --learning_rate 0.001 --hidden_units 512 --epochs 20 --device cpu
```

To predict the class of an image using the trained model, run the following command:
```bash
        Example usage:
            python predict.py --image_path /path/to/image --checkpoint /path/to/checkpoint --top_k 3 --category_names /path/to/cat_to_name.json --device cpu
```

### More information

For instructions on how you can use a remote GPU, please see [remote-training.md](/path/to/remote-training.md).

## Jupyter Notebook

