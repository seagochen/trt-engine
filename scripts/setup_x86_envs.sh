#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
TARGET_DIR="$HOME/Projects/Serverlet"
LINK_NAME="/opt/Serverlet"

# Function to print messages
print_message() {
    echo "=============================="
    echo "$1"
    echo "=============================="
}

# Function to update package list
update_packages() {
    print_message "Updating package list..."
    sudo apt update
}

# Function to install system dependencies
install_dependencies() {
    print_message "Installing OpenCV and other system dependencies..."
    sudo apt install -y \
        build-essential \
        cmake \
        git \
        pkg-config \
        libgtk-3-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libv4l-dev \
        libxvidcore-dev \
        libx264-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        gfortran \
        openexr \
        libatlas-base-dev \
        python3-dev \
        python3-numpy \
        libtbbmalloc2 \
        libtbb-dev \
        libdc1394-dev
}

# Function to install OpenCV
install_opencv() {
    print_message "Installing OpenCV..."
    sudo apt install -y libopencv-dev
}

# Function to install Protobuf
install_protobuf() {
    print_message "Installing Protobuf..."
    sudo apt install -y protobuf-compiler libprotobuf-dev
}

# Function to install YAML library
install_yaml() {
    print_message "Installing YAML library..."
    sudo apt install -y libyaml-cpp-dev
}

# Function to install Mosquitto
install_mosquitto() {
    print_message "Installing Mosquitto..."
    sudo apt install -y mosquitto mosquitto-clients libmosquitto-dev
}

install_nvitop() {
    print_message "Installing nvitop for x86 platform..."

    # 更新系统包并安装 Python 和 pip
    sudo apt update
    sudo apt install -y python3 python3-pip

    # 安装 nvitop
    pip3 install -U nvitop

    # 测试 nvitop 是否安装成功
    print_message "Testing nvitop installation..."
    if nvitop --help >/dev/null 2>&1; then
        print_message "nvitop installed successfully! You can now run 'nvitop'."
    else
        print_message "nvitop installation failed. Please check for errors."
    fi
}

# Function to check installed versions
check_versions() {
    print_message "Checking installed versions..."

    # OpenCV
    if pkg-config --exists opencv4; then
        opencv_version=$(pkg-config --modversion opencv4)
        echo "OpenCV version: $opencv_version"
    else
        echo "OpenCV is not installed."
    fi

    # Protobuf
    if command -v protoc &> /dev/null; then
        protobuf_version=$(protoc --version)
        echo "Protobuf version: $protobuf_version"
    else
        echo "Protobuf is not installed."
    fi

    # YAML
    yaml_version=$(dpkg -s libyaml-cpp-dev 2>/dev/null | grep '^Version:')
    if [ -n "$yaml_version" ]; then
        echo "YAML version: $yaml_version"
    else
        echo "YAML library is not installed."
    fi

    # Mosquitto
    mosquitto_version=$(mosquitto -h 2>/dev/null | grep -i version || echo "Mosquitto is not installed.")
    echo "Mosquitto version: $mosquitto_version"
}

# Function to create a symbolic link in /opt
setup_symlink() {
    print_message "Setting up symbolic link: $LINK_NAME -> $TARGET_DIR"

    # Check if the target directory exists
    if [ ! -d "$TARGET_DIR" ]; then
        echo "Error: Target directory $TARGET_DIR does not exist."
        exit 1
    fi

    # Check if the symbolic link already exists
    if [ -L "$LINK_NAME" ]; then
        echo "Symbolic link $LINK_NAME already exists. Removing..."
        sudo rm -f "$LINK_NAME"
    elif [ -e "$LINK_NAME" ]; then
        echo "Error: $LINK_NAME exists but is not a symlink. Please remove it manually."
        exit 1
    fi

    # Create the symbolic link
    sudo ln -s "$TARGET_DIR" "$LINK_NAME"

    echo "Symbolic link created successfully."
}

# Main execution flow
main() {
    update_packages
    install_dependencies
    install_opencv
    install_protobuf
    install_yaml
    install_mosquitto
    install_nvitop
    check_versions

    setup_symlink
    print_message "Installation and setup complete!"
}

# Run the main function
main
