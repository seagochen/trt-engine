#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

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
        libtbb2 \
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

# Function to install Jetson-specific packages
install_jetson_packages() {
    print_message "Installing Jetson-specific packages..."
    sudo apt install -y jetson-stats
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
    yaml_version=$(dpkg -s libyaml-cpp-dev 2>/dev/null | grep '^Version: ')
    if [ -n "$yaml_version" ]; then
        echo "YAML version: $yaml_version"
    else
        echo "YAML library is not installed."
    fi

    # Mosquitto
    mosquitto_version=$(mosquitto -h 2>/dev/null | grep -i version || echo "Mosquitto is not installed.")
    echo "Mosquitto version: $mosquitto_version"
}

# Function to handle x86 installation
x86_installation() {
    print_message "Starting x86 installation..."
    update_packages
    install_dependencies
    install_opencv
    install_protobuf
    install_yaml
    install_mosquitto
    check_versions

    print_message "x86 Installation and setup complete!"
}

# Function to handle Jetson Nano installation
jetson_installation() {
    print_message "Starting Jetson Nano installation..."
    update_packages
    install_dependencies
    install_opencv
    install_protobuf
    install_yaml
    install_mosquitto
    install_jetson_packages
    check_versions

    print_message "Jetson Nano Installation and setup complete!"
}

# Main execution flow
main() {
    print_message "Select your installation target:"
    echo "1) x86"
    echo "2) Jetson Nano"
    read -p "Enter your choice (1 or 2): " choice

    case $choice in
        1)
            x86_installation
            ;;
        2)
            jetson_installation
            ;;
        *)
            echo "Invalid choice. Exiting..."
            exit 1
            ;;
    esac
}

# Run the main function
main
