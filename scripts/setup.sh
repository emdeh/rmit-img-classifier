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

# Install Miniconda if not installed
if ! command -v conda &> /dev/null; then
    echo "Conda not found. Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    export PATH="$HOME/miniconda/bin:$PATH"
    rm miniconda.sh
fi

# Clone the repository
if [ ! -d "$REPO_DIR" ]; then
    echo "Cloning the repository..."
    git clone $REPO_URL $REPO_DIR
fi

# Navigate to the repository directory
cd $REPO_DIR

# Create data directory if it doesn't exist
if [ ! -d "$DATA_DIR" ]; then
    echo "Creating data directory..."
    mkdir -p $DATA_DIR
fi

# Download and extract the dataset
echo "Downloading and extracting the dataset..."
wget $DATA_URL -O "$DATA_DIR/flower_data.tar.gz"
tar -xzvf "$DATA_DIR/flower_data.tar.gz" -C $DATA_DIR
rm "$DATA_DIR/flower_data.tar.gz"

# Create checkpoints directory if it doesn't exist
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Creating checkpoints directory..."
    mkdir -p $CHECKPOINT_DIR
fi

# Get the environment name from the env.yaml file
ENV_NAME=$(grep 'name:' env.yaml | cut -d' ' -f2)

# Check if the environment already exists
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' already exists. Skipping creation."
else
    # Create the conda environment from the env.yaml file
    echo "Creating conda environment from env.yaml..."
    conda env create -f env.yaml
fi

# Activate the environment in the current shell
echo "Activating conda environment..."
source $HOME/miniconda/bin/activate $ENV_NAME

# Ensure the environment is activated
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
    echo "Failed to activate conda environment '$ENV_NAME'. Exiting."
    exit 1
fi

# Install any additional dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

# Inform the user that setup is complete and the environment is activated
echo "Setup is complete. The environment '$ENV_NAME' is activated."
