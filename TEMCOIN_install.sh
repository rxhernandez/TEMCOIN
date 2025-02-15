#!/bin/bash

# Clone the TEMCOIN repository from GitHub
echo "Cloning TEMCOIN repository..."
git clone https://github.com/rxhernandez/TEMCOIN.git

# Change directory to the cloned repository
cd TEMCOIN

# Install or upgrade the build package
echo "Installing/upgrading the build package..."
pip install --upgrade build

# Build the Python wheel file
echo "Building the Python wheel..."
python -m build --wheel

# Install the generated wheel file
echo "Searching for and installing the latest wheel file..."
WHL_FILE=$(ls dist/TEMCOIN-*.whl)
pip install $WHL_FILE

echo "Installation completed!"