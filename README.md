# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.

## Geting started

To setup the environment, run one of the setup scripts that is appropriate for your environment.

- For instructions on how you can use a remote GPU, please see [remote-use.md](/path/to/remote-training.md).
- For instructions on how to use a local GPU (or CPU), please see [local-use.md](/path/to/local-use.md).

## Usage

To train the image classifier, run the following command:
```bash
python train.py --data_dir <path_to_data_directory> --save_dir <path_to_save_directory> --arch <model_architecture> --learning_rate <learning_rate> --hidden_units <hidden_units> --epochs <num_epochs> --gpu
```

To predict the class of an image using the trained model, run the following command:
```bash
python predict.py --image_path <path_to_image> --checkpoint <path_to_checkpoint> --top_k <num_top_predictions> --category_names <path_to_category_names> --gpu
```

### More information

For instructions on how you can use a remote GPU, please see [remote-training.md](/path/to/remote-training.md).

## Jupyter Notebook

