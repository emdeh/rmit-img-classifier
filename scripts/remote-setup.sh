#!/bin/bash

# Variables
ENV_NAME="remote-env"
REPO_URL="https://github.com/emdeh/rmit-img-classifier.git"
DATA_URL="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
PROJECT_DIR="$HOME/img-classifier"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"
DATA_DIR="$PROJECT_DIR/data"
ENV_FILE="$PROJECT_DIR/remote-env.yaml"

# Move to the user's home directory
cd $HOME

# Check if Miniconda is installed
if ! command -v conda &> /dev/null; then
    echo "Miniconda not found, installing Miniconda..."
    curl -O $MINICONDA_URL
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm $HOME/Miniconda3-latest-Linux-x86_64.sh
    echo "Miniconda installed."
else
    echo "Miniconda is already installed."
fi

# Clone the project repository
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning the project repository..."
    git clone $REPO_URL $PROJECT_DIR
else
    echo "Project repository already exists."
fi

# Download and extract the dataset into the data directory
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory and downloading dataset..."
    mkdir -p $DATA_DIR
    wget $DATA_URL -O "$DATA_DIR/flower_data.tar.gz"
    tar -xzf "$DATA_DIR/flower_data.tar.gz" -C $DATA_DIR
    rm "$DATA_DIR/flower_data.tar.gz"
else
    if [ ! -d "$DATA_DIR/flowers" ]; then
        echo "Flowers dataset not found. Downloading dataset..."
        wget $DATA_URL -O "$DATA_DIR/flower_data.tar.gz"
        tar -xzf "$DATA_DIR/flower_data.tar.gz" -C $DATA_DIR
        rm "$DATA_DIR/flower_data.tar.gz"
    else
        echo "Flowers dataset already exists in the data directory."
    fi
fi

# Create the checkpoints directory
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoints directory..."
    mkdir -p $CHECKPOINT_DIR
else
    echo "Checkpoints directory already exists."
fi

# Create and activate the Conda environment
if [ -f "$ENV_FILE" ]; then
    echo "Creating Conda environment from $ENV_FILE..."
    conda init
    source ~/.bashrc
    conda env create -f $ENV_FILE
    echo "Activating Conda environment..."
    conda activate $ENV_NAME
else
    echo "$ENV_FILE not found. Please ensure the env.yaml file is present in the project directory."
    exit 1
fi

echo "Environment setup is complete. The Conda environment is now activated."
