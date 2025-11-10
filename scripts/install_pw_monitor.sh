#!/bin/bash
#
# Power Monitor Installation Script - Shell Wrapper
#
# This script wraps the Python install_pw_monitor.py script for easier usage.
# It automatically detects the platform (Jetson or x86) and installs the
# appropriate power monitoring tool:
#   - Jetson: jetson-stats (jtop)
#   - x86 with NVIDIA GPU: nvitop
#
# Usage:
#   ./install_pw_monitor.sh
#
# Author: TrtEngineToolkits
# Date: 2025-11-10
#

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_SCRIPT="${SCRIPT_DIR}/install_pw_monitor.py"

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

print_header() {
    echo -e "${CYAN}========================================${RESET}"
    echo -e "${CYAN}$1${RESET}"
    echo -e "${CYAN}========================================${RESET}"
}

# Function to check if running on Jetson
is_jetson() {
    if [ -f "/etc/nv_tegra_release" ]; then
        return 0
    fi

    if [ -f "/proc/device-tree/compatible" ]; then
        if grep -qi "tegra" /proc/device-tree/compatible 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

# Function to check Python script
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
${GREEN}Power Monitor Installation Tool${RESET}

${BLUE}Description:${RESET}
  Automatically detects your platform and installs the appropriate
  power monitoring tool:

  ${CYAN}Jetson Platform:${RESET}
    - Installs jetson-stats (jtop)
    - Provides GPU, CPU, memory, and power monitoring
    - Run with: jtop

  ${CYAN}x86 Platform with NVIDIA GPU:${RESET}
    - Installs nvitop
    - Provides GPU monitoring similar to nvidia-smi but interactive
    - Run with: nvitop

${BLUE}Usage:${RESET}
  $0 [OPTION]

${BLUE}Options:${RESET}
  --help, -h            Show this help message

${BLUE}Examples:${RESET}
  $0                    # Auto-detect platform and install

${BLUE}Notes:${RESET}
  " For Jetson: sudo privileges required (will prompt if needed)
  " For x86: regular user installation
  " Requires internet connection to download packages

EOF
}

# Main installation function
main() {
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        "")
            # No arguments, proceed with installation
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac

    # Check prerequisites
    check_python_script
    check_python

    # Display header
    print_header "Power Monitor Installation"

    # Detect platform
    if is_jetson; then
        print_info "Detected NVIDIA Jetson platform"
        print_info "Installing jetson-stats (jtop)..."
    else
        print_info "Detected x86 platform"
        print_info "Installing nvitop..."
    fi

    echo ""

    # Run the Python installation script
    if python3 "$PYTHON_SCRIPT"; then
        echo ""
        print_success "Installation completed successfully!"
        echo ""

        if is_jetson; then
            print_info "You can now run: ${GREEN}jtop${RESET}"
            print_warning "Note: You may need to log out and log back in for jtop to work"
        else
            print_info "You can now run: ${GREEN}nvitop${RESET}"
        fi
    else
        echo ""
        print_error "Installation failed!"
        exit 1
    fi
}

# Run main function
main "$@"
