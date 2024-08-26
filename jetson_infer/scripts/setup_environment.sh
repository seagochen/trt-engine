#!/bin/bash

# Function to set up the environment for x86_64
setup_x86_64() {
    echo "Setting up environment for x86_64..."

    # Setup the CUDA and TensorRT Environment
    echo "export PATH=/usr/local/cuda/bin:$(pwd)/tensorrt/targets/x86_64-linux-gnu/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(pwd)/tensorrt/targets/x86_64-linux-gnu/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

    # Use ldconfig to update the library cache
    sudo ldconfig

    echo "Environment setup for x86_64 completed and added to .bashrc."
}

# Function to set up the environment for aarch64
setup_aarch64() {
    echo "Setting up environment for aarch64..."

    # Setup the CUDA and TensorRT Environment
    echo "export PATH=/usr/local/cuda/bin:$(pwd)/tensorrt/targets/aarch64-linux-gnu/bin:\$PATH" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$(pwd)/tensorrt/targets/aarch64-linux-gnu/lib:\$LD_LIBRARY_PATH" >> ~/.bashrc

    # Use ldconfig to update the library cache
    sudo ldconfig

    echo "Environment setup for aarch64 completed and added to .bashrc."
}

# Function to remove configuration from .bashrc
remove_config() {
    echo "Removing environment configuration from .bashrc..."

    # Remove lines containing the specific paths
    sed -i '/\/usr\/local\/cuda\/bin/d' ~/.bashrc
    sed -i '/\/usr\/local\/cuda\/lib64/d' ~/.bashrc
    sed -i '/tensorrt\/targets\/x86_64-linux-gnu\/bin/d' ~/.bashrc
    sed -i '/tensorrt\/targets\/x86_64-linux-gnu\/lib/d' ~/.bashrc
    sed -i '/tensorrt\/targets\/aarch64-linux-gnu\/bin/d' ~/.bashrc
    sed -i '/tensorrt\/targets\/aarch64-linux-gnu\/lib/d' ~/.bashrc

    echo "Configuration removed from .bashrc."
}

# Prompt the user to select the operation
echo "Select the operation:"
echo "1) Set up environment for x86_64"
echo "2) Set up environment for aarch64"
echo "3) Remove environment configuration from .bashrc"

read -p "Enter your choice (1, 2, or 3): " choice

case $choice in
    1)
        setup_x86_64
        ;;
    2)
        setup_aarch64
        ;;
    3)
        remove_config
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

# Reload .bashrc to apply changes
source ~/.bashrc
