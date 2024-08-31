#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Echo each command being executed
set -x

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda first."
    exit 1
fi

# Create the conda environment from the env.yaml file
echo "Creating conda environment from env.yaml..."
conda env create -f ../env.yaml

# Activate the environment
echo "Activating conda environment..."
source activate ai_trainer_predictor

# Install any additional dependencies from requirements.txt
echo "Installing dependencies from requirements.txt..."
pip install -r ../requirements.txt

# Inform the user that setup is complete
echo "Setup is complete. To activate the environment, run: conda activate ai_trainer_predictor"
