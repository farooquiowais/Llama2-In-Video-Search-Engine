#!/bin/bash

# Any preliminary setup
echo "Setting up environment..."

# Upgrade pip
pip install --upgrade pip

# Install packages from requirements.txt
pip install -r requirement.txt

# Install PyTorch, torchvision, and torchaudio for specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117 --upgrade

echo "Installation completed successfully!"
