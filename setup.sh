#!/bin/bash

PYTHON_VERSION="3.8.18"
VENV_NAME="myenv"

sudo apt update

# Check if Python 3.8 is already installed
if ! command -v python3.8 &>/dev/null; then
    # If Python 3.8 is not installed, install it
    echo "Installing Python 3.8.18..."
    sudo apt update
    sudo apt install -y python3.8 python3-pip python3.8-venv
fi

# Create a virtual environment with Python 3.8
echo "Creating virtual environment with Python 3.8.18..."
python3.8 -m venv $VENV_NAME

# Activate the virtual environment
echo "Activating virtual environment..."
source $VENV_NAME/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install project dependencies
echo "Installing project dependencies..."
pip install -r requirements.txt

# Deactivate the virtual environment
deactivate

echo "Setup completed. To activate the virtual environment, use 'source $VENV_NAME/bin/activate'."