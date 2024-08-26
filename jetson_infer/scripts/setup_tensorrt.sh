#!/bin/bash

# Function to list available tarballs in the tarballs directory
list_tarballs() {
    echo "Available TensorRT tarballs:"
    tarballs=($(ls tarballs/*.tar.gz))
    for i in "${!tarballs[@]}"; do
        echo "$((i + 1))) ${tarballs[$i]}"
    done
}

# Function to prompt user for selection
select_tarball() {
    read -p "Enter the number of the tarball you want to extract: " selection
    if [[ $selection -lt 1 || $selection -gt ${#tarballs[@]} ]]; then
        echo "Invalid selection. Exiting."
        exit 1
    fi
    tarfile=${tarballs[$((selection - 1))]}
}

# Function to extract and organize the TensorRT files
extract_and_organize() {
    # Create the tensorrt directory and clean it if it exists
    if [ ! -d tensorrt ]; then
        mkdir tensorrt
    else
        rm -rf tensorrt/*
    fi

    # Extract the tarball into the tensorrt directory
    tar -xvzf $tarfile -C tensorrt

    # Determine the untar folder name dynamically
    untar_folder=$(tar -tf $tarfile | head -n 1 | cut -f1 -d"/")

    # Move the subdirectories to the top level
    mv "tensorrt/$untar_folder"/* tensorrt

    # Remove the empty directory
    rmdir "tensorrt/$untar_folder"

    echo "TensorRT extracted and organized successfully."
}

# List available tarballs
list_tarballs

# Prompt the user to select a tarball
select_tarball

# Extract and organize the selected tarball
extract_and_organize
