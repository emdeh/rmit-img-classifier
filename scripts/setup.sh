#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Echo each command being executed
set -x

# Define variables
REPO_URL="https://github.com/emdeh/rmit-img-classifier.git"
DATA_URL="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
REPO_DIR="$HOME/img-classifier"
DATA_DIR="$REPO_DIR/data"
CHECKPOINT_DIR="$REPO_DIR/checkpoints"

# Install Miniconda if not already installed
if [ ! -d "$HOME/miniconda" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    # Install Miniconda silently and accept the EULA by using the `-b` option
    bash ~/miniconda.sh -b -p $HOME/miniconda
    rm ~/miniconda.sh
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Initialize conda
source "$HOME/miniconda/etc/profile.d/conda.sh"

# Create the conda environment from the env.yaml file
echo "Creating conda environment from env.yaml..."
conda env create -f env.yaml

# Activate the environment
echo "Activating conda environment..."
conda activate fnl-prj-img-class

# Install any additional dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Download and extract dataset
mkdir -p data/flowers
wget https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz -O data/flower_data.tar.gz
tar -xzvf data/flower_data.tar.gz -C data/flowers
rm data/flower_data.tar.gz

# Create checkpoints directory
mkdir -p checkpoints

# Inform the user that setup is complete
echo "Setup is complete. To activate the environment in the future, run: conda activate fnl-prj-img-class"