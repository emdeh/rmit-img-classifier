#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Echo each command being executed
set -x

# Define variables
PROJECT_DIR="$HOME/img-classifier"
REPO_URL="https://github.com/emdeh/rmit-img-classifier.git"
DATA_URL="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
DATA_DIR="$PROJECT_DIR/data"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# Install Miniconda if not already installed
if [ ! -d "$HOME/miniconda" ]; then
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # Install Miniconda silently and accept the EULA by using the `-b` option
    bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    rm Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc
    export PATH="$HOME/miniconda/bin:$PATH"
fi

# Initialize conda
source "$HOME/miniconda/etc/profile.d/conda.sh"

# Clone the repository
if [ ! -d "$PROJECT_DIR" ]; then
    echo "Cloning the repository..."
    git clone $REPO_URL $PROJECT_DIR
fi

# Change to the project directory
cd $PROJECT_DIR

# Download and extract dataset
echo "Creating data directory and downloading dataset..."
mkdir -p $DATA_DIR
wget $DATA_URL -O $DATA_DIR/flower_data.tar.gz
echo "Extracting dataset..."
tar -xzvf $DATA_DIR/flower_data.tar.gz -C $DATA_DIR
echo "Removing compressed dataset..."
rm $DATA_DIR/flower_data.tar.gz

# Create the conda environment from the env.yaml file
echo "Creating conda environment from env.yaml..."
conda env create -f env.yaml

# Activate the environment
echo "Activating conda environment..."
conda activate fnl-prj-img-class

# Install any additional dependencies from requirements.txt
#echo "Installing dependencies from requirements.txt..."
#pip install -r requirements.txt

# Create checkpoints directory
mkdir -p $CHECKPOINT_DIR

# Inform the user that setup is complete
echo "Setup is complete. To activate the environment in the future, run: conda activate fnl-prj-img-class"
