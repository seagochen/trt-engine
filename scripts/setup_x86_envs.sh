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
    print_message "Installing system dependencies..."

    # Common dependencies
    COMMON_DEPS=(
        build-essential
        cmake
        git
        pkg-config
        libgtk-3-dev
        libavcodec-dev
        libavformat-dev
        libswscale-dev
        libv4l-dev
        libxvidcore-dev
        libx264-dev
        libjpeg-dev
        libpng-dev
        libtiff-dev
        gfortran
        openexr
        libatlas-base-dev
        python3-dev
        python3-numpy
        libdc1394-dev
    )

    # Attempt to install libtbbmalloc2 first
    TBB_DEPS=(libtbbmalloc2)
    print_message "Attempting to install libtbbmalloc2..."
    if ! sudo apt install -y "${COMMON_DEPS[@]}" "${TBB_DEPS[@]}"; then
        print_message "libtbbmalloc2 not found or failed to install. Trying libtbb-dev instead..."
        TBB_DEPS=(libtbb-dev)
        # Re-attempt installation with libtbb-dev
        if ! sudo apt install -y "${COMMON_DEPS[@]}" "${TBB_DEPS[@]}"; then
            print_message "Failed to install required TBB dependencies (libtbbmalloc2 or libtbb-dev). Please check your repositories."
            exit 1
        fi
    fi
    print_message "System dependencies installed successfully."
}

# Function to install OpenCV
install_opencv() {
    print_message "Installing OpenCV..."
    sudo apt install -y libopencv-dev
}

# Function to install nvitop
install_nvitop() {
    print_message "Installing nvitop for x86 platform..."

    # Install Python and pip (if not already installed, apt install is idempotent)
    sudo apt install -y python3 python3-pip

    # Install nvitop
    pip3 install -U nvitop

    # Test nvitop installation
    print_message "Testing nvitop installation..."
    if command -v nvitop &> /dev/null; then # More robust check for command existence
        print_message "nvitop installed successfully! You can now run 'nvitop'."
    else
        print_message "nvitop installation failed. Please check for errors."
        # Optionally, exit here if nvitop is a critical dependency
        # exit 1
    fi
}

# Function to check installed versions
check_versions() {
    print_message "Checking installed versions..."

    # OpenCV
    if pkg-config --exists opencv4; then
        opencv_version=$(pkg-config --modversion opencv4)
        echo "OpenCV version: $opencv_version"
    elif pkg-config --exists opencv; then # Fallback for older OpenCV
        opencv_version=$(pkg-config --modversion opencv)
        echo "OpenCV version: $opencv_version (using 'opencv' module)"
    else
        echo "OpenCV is not installed or pkg-config cannot find it."
    fi

    # TBB check
    if dpkg -s libtbbmalloc2 &>/dev/null; then
        echo "libtbbmalloc2 is installed."
    elif dpkg -s libtbb-dev &>/dev/null; then
        echo "libtbb-dev is installed, which typically provides TBB malloc libraries."
    else
        echo "Neither libtbbmalloc2 nor libtbb-dev appear to be installed."
    fi

    # Python
    echo "Python 3 version: $(python3 --version 2>&1)"
    echo "pip 3 version: $(pip3 --version 2>&1)"
}

# Main execution flow
main() {
    update_packages
    install_dependencies
    install_opencv # Note: install_dependencies already has many OpenCV prerequisites, but this installs the main libopencv-dev
    install_nvitop
    check_versions
    print_message "Installation and setup complete!"
}

# Run the main function
main