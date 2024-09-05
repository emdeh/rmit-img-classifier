#!/bin/bash

# Variables
PROJECT_DIR=$HOME/img-classifier
SRC_DIR="$PROJECT_DIR/src"
REPO_ZIP_URL="https://github.com/emdeh/rmit-img-classifier/archive/refs/heads/main.zip"
ZIP_FILE="$PROJECT_DIR/repo.zip"
REMOTE_ENV_FILE="$PROJECT_DIR/remote_env.yaml"
ENV_FILE="$PROJECT_DIR/env.yaml"
REMOTE_ENV_URL="https://raw.githubusercontent.com/emdeh/rmit-img-classifier/main/env.yaml"

# Move to the project directory
cd $PROJECT_DIR

# Download the latest version of the project repository as a zip file
echo "Downloading the latest project repository..."
curl -sL $REPO_ZIP_URL -o $ZIP_FILE

# Unzip and only extract the src/ directory, then overwrite the local src/
echo "Extracting and updating the src/ directory..."
unzip -o $ZIP_FILE "rmit-img-classifier-main/src/*" -d $PROJECT_DIR
mv $PROJECT_DIR/rmit-img-classifier-main/src/* $SRC_DIR/
rm -rf $PROJECT_DIR/rmit-img-classifier-main

# Clean up the zip file
rm $ZIP_FILE

# Check if the local env.yaml exists
if [ ! -f "$ENV_FILE" ]; then
    echo "Local env.yaml file not found. Exiting."
    exit 1
fi

# Download the latest env.yaml file from GitHub
curl -s -o "$REMOTE_ENV_FILE" "$REMOTE_ENV_URL"

# Compare the local and remote env.yaml files using md5sum
LOCAL_HASH=$(md5sum $ENV_FILE | awk '{ print $1 }')
REMOTE_HASH=$(md5sum $REMOTE_ENV_FILE | awk '{ print $1 }')

if [ "$LOCAL_HASH" != "$REMOTE_HASH" ]; then
    echo "env.yaml file has changed. Updating Conda environment..."
    conda env update -f $ENV_FILE --prune
else
    echo "env.yaml is up to date. No Conda environment update required."
fi

# Clean up the remote env.yaml file
rm "$REMOTE_ENV_FILE"

echo "Project update complete."
