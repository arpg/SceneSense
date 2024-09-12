#!/bin/bash

# Check if the user provided a directory as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

# Directory for setting up the virtual environment
DIR=$1

# Navigate to the specified directory
if [ ! -d "$DIR" ]; then
    echo "The directory $DIR does not exist."
    exit 1
fi

cd "$DIR"

#install venv 3.9
sudo apt install python3.9 python3.9-venv python3.9-dev


# Create a Python virtual environment
python3.9 -m venv .env

# Activate the virtual environment
source .env/bin/activate

# Install the required packages
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install diffusers["torch"] transformers
pip install --upgrade huggingface_hub
pip install natsort
pip install wandb
pip install spconv-cu120
pip install scipy
pip install matplotlib

pip install --upgrade pip

pip install open3d
pip install opencv-python
pip install clean-fid

# Install git-lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs

pip install -e .
echo "Setup completed in directory: $DIR"

