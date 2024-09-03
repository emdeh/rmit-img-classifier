#!/bin/bash

"""
This script automates the setup of the image classifier project environment on a Lambda Labs instance.

Usage:
    source setup_env.sh

The script performs the following tasks:
1. Installs Miniconda if not already installed.
2. Clones the image classifier project repository from GitHub.
3. Downloads and extracts the required dataset.
4. Creates a Conda environment using the 'env.yaml' file.
5. Activates the Conda environment.
6. Creates the necessary directory structure for storing checkpoints.

To get started:
1. Provision a remote instance and ssh to it.

2. Run this command to retrieve this script:
    wget https://raw.githubusercontent.com/emdeh/rmit-img-classifier/main/scripts/setup.sh

3. Run this command to make it executable:
    chmod +x setup.sh

4. Then run this command to execute it:
    source setup.sh

**Important**:
- To ensure the Conda environment remains activated in your current shell, it is improtant to run this script using the `source` command:
  `source setup_env.sh`

- After running the script, the environment will be activated.

Directory Structure:
- ~/img-classifier/: Root directory of the cloned project.
- ~/img-classifier/data/: Directory where the dataset will be downloaded and extracted.
- ~/img-classifier/checkpoints/: Directory for storing model checkpoints.

Environment:
- Miniconda is used to manage dependencies and environments.
- The environment configuration is provided in the 'env.yaml' file.

Dependencies:
- The 'env.yaml' file should be present in the project directory for the environment setup.

Output:
- The script outputs messages to the console, indicating the progress of each task.

Author:
- [emdeh]

"""

# Exit immediately if a command exits with a non-zero status
set -e

# Echo each command being executed
set -x

# cd to root directory
cd ~

# Define variables
PROJECT_DIR="$HOME/img-classifier"
REPO_URL="https://github.com/emdeh/rmit-img-classifier.git"
DATA_URL="https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz"
DATA_DIR="$PROJECT_DIR/data/flowers"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# Install Miniconda if not already installed
if [ ! -d "$HOME/miniconda" ]; then
    echo "Installing Miniconda..."
    curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    # Install Miniconda silently and accept the EULA by using the `-b` option
    bash Miniconda3-latest-Linux-x86_64.sh -b
    rm Miniconda3-latest-Linux-x86_64.sh
    source ~/.bashrc
    export PATH="$HOME/miniconda3/bin:$PATH"
fi

# Initialize conda
source "$HOME/miniconda3/etc/profile.d/conda.sh"

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

# Create checkpoints directory
mkdir -p $CHECKPOINT_DIR

# Inform the user that setup is complete
echo "Setup is complete. To activate the environment in the future, run: conda activate fnl-prj-img-class"