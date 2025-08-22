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

# Function to install Jetson Stats
install_jetson_stats() {
    # 更新系统包并安装 jetson-stats
    print_message "Installing jetson-stats..."
    sudo apt update
    sudo apt install -y python3-pip
    sudo pip3 install -U jetson-stats

    # 启动并设置 jtop.service 服务
    print_message "Setting up jtop service..."
    sudo systemctl enable jtop.service   # 确保服务开机自启
    sudo systemctl restart jtop.service  # 重启服务使其生效

    # 检查服务是否成功启动
    if systemctl is-active --quiet jtop.service; then
        print_message "jtop service is active. You can now use jtop."
    else
        print_message "jtop service failed to start. Checking logs..."
        journalctl -u jtop.service --no-pager
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
}

# Main execution flow
main() {
    update_packages
    install_dependencies
    install_opencv
    install_jetson_stats
    check_versions
    print_message "Installation and setup complete!"
}

# Run the main function
main
