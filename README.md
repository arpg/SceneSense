Pointnet implimentation: https://github.com/yanx27/Pointnet_Pointnet2_pytorch 

## Installation

python -m venv .env

source .env/bin/activate

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install diffusers["torch"] transformers

pip install --upgrade huggingface_hub

pip3 install natsort

pip install wandb

pip install spconv

pip install scipy

pip install matplotlib

### git-lfs install: 
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash

sudo apt-get install git-lfs
