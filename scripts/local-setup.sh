#!/bin/bash

read -p "Have you cloned the repo? (y/n): " answer

if [[ $answer == [Yy]* ]]; then
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
fi
else
    echo "Please clone the repo first!"
    exit 1
fi