#!/bin/bash
#
# TensorRT Engine Builder - Shell Wrapper
#
# This script wraps the Python build_engine.py script for easier usage.
# It provides a convenient shell interface to build TensorRT engines from ONNX models.
#
# Usage:
#   ./build_engine.sh <config_file.json>
#   ./build_engine.sh                     # Interactive mode
#
# Author: TrtEngineToolkits
# Date: 2025-11-10
#

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RESET='\033[0m'

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/build_engine.py"

# Function to print colored messages
print_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${RESET} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${RESET} $1"
}

# Function to check if Python script exists
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Function to check Python availability
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python3 is not installed or not in PATH"
        exit 1
    fi
}

# Function to display usage
show_usage() {
    cat << EOF
${GREEN}TensorRT Engine Builder${RESET}

${BLUE}Usage:${RESET}
  $0 <config_file.json>    Build engine from configuration file
  $0                       Interactive mode (browse and select config)
  $0 -h, --help            Show this help message

${BLUE}Examples:${RESET}
  $0 config/yolov8_pose.json
  $0 ../models/efficientnet.json

${BLUE}Configuration File Format:${RESET}
  The JSON configuration file should contain:
    - onnx_path: Path to input ONNX model
    - target_path: Path for output TensorRT engine
    - trt_config: TensorRT build options (precision, shapes, etc.)

  See build_engine.py for detailed configuration examples.

${BLUE}Features:${RESET}
  • Automatic NVDLA detection and fallback to GPU
  • Dynamic shape support
  • FP16/INT8 precision options
  • Real-time build progress display
  • Comprehensive error handling

EOF
}

# Function to list available config files
list_configs() {
    local config_dir="${SCRIPT_DIR}/../config"

    if [ -d "$config_dir" ]; then
        print_info "Searching for configuration files in: $config_dir"
        local configs=($(find "$config_dir" -name "*.json" 2>/dev/null))

        if [ ${#configs[@]} -eq 0 ]; then
            print_warning "No JSON configuration files found in $config_dir"
            return 1
        fi

        echo -e "\n${GREEN}Available configuration files:${RESET}"
        for i in "${!configs[@]}"; do
            echo "  [$((i+1))] ${configs[$i]}"
        done
        echo ""

        return 0
    else
        print_warning "Config directory not found: $config_dir"
        return 1
    fi
}

# Function to run the Python build script
run_build() {
    local config_file="$1"

    print_info "Starting TensorRT engine build process..."
    print_info "Configuration: $config_file"
    echo ""

    # Run the Python script
    python3 "$PYTHON_SCRIPT" "$config_file"
    local exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        print_success "Build process completed successfully!"
    else
        print_error "Build process failed with exit code: $exit_code"
    fi

    return $exit_code
}

# Main script logic
main() {
    # Check prerequisites
    check_python_script
    check_python

    # Handle command line arguments
    if [ $# -eq 0 ]; then
        # Interactive mode
        print_info "Interactive mode - No configuration file specified"

        if list_configs; then
            read -p "Enter the number of config file to use (or press Enter to exit): " selection

            if [ -z "$selection" ]; then
                print_info "No selection made. Exiting."
                exit 0
            fi

            # Get the selected config file
            local config_dir="${SCRIPT_DIR}/../config"
            local configs=($(find "$config_dir" -name "*.json" 2>/dev/null))
            local index=$((selection - 1))

            if [ $index -ge 0 ] && [ $index -lt ${#configs[@]} ]; then
                run_build "${configs[$index]}"
            else
                print_error "Invalid selection: $selection"
                exit 1
            fi
        else
            print_info "Please provide a configuration file as argument."
            echo ""
            show_usage
            exit 1
        fi
    elif [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
        show_usage
        exit 0
    else
        # Configuration file provided as argument
        local config_file="$1"

        # Check if file exists
        if [ ! -f "$config_file" ]; then
            print_error "Configuration file not found: $config_file"
            exit 1
        fi

        # Check if it's a JSON file
        if [[ ! "$config_file" =~ \.json$ ]]; then
            print_warning "File does not have .json extension: $config_file"
            read -p "Continue anyway? (y/N): " confirm
            if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
                print_info "Build cancelled."
                exit 0
            fi
        fi

        run_build "$config_file"
    fi
}

# Run main function
main "$@"
