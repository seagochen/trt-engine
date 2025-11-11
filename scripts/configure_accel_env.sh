#!/bin/bash
#
# CUDA & TensorRT Environment Configuration Script - Shell Wrapper
#
# This script wraps the Python configure_accel_env.py script for easier usage.
# It automatically detects CUDA and TensorRT installation locations and configures
# the necessary environment variables (PATH, LD_LIBRARY_PATH) in your shell rc file.
#
# Usage:
#   ./configure_accel_env.sh
#   ./configure_accel_env.sh --dry-run
#   ./configure_accel_env.sh --bashrc ~/.zshrc
#
# Author: TrtEngineToolkits
# Date: 2025-11-11
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
PYTHON_SCRIPT="${SCRIPT_DIR}/configure_accel_env.py"

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
${GREEN}CUDA & TensorRT Environment Configuration Tool${RESET}

${BLUE}Description:${RESET}
  Automatically detects CUDA and TensorRT installation locations and
  configures environment variables (PATH, LD_LIBRARY_PATH) in your
  shell configuration file.

  ${CYAN}What it does:${RESET}
    - Auto-detects CUDA installation (nvcc, libraries)
    - Auto-detects TensorRT installation (trtexec, libraries)
    - Writes idempotent environment variable exports to shell rc file
    - Supports both Jetson and x86 platforms

${BLUE}Usage:${RESET}
  $0 [OPTION]

${BLUE}Options:${RESET}
  --help, -h            Show this help message
  --dry-run             Preview changes without modifying files
  --bashrc FILE         Specify shell rc file to modify (default: ~/.bashrc)

${BLUE}Examples:${RESET}
  $0                           # Auto-configure with ~/.bashrc
  $0 --dry-run                 # Preview what would be configured
  $0 --bashrc ~/.zshrc         # Configure ~/.zshrc instead
  $0 --bashrc ~/.bashrc --dry-run    # Preview changes for ~/.bashrc

${BLUE}Notes:${RESET}
  • Idempotent: Safe to run multiple times
  • Creates marked blocks in your shell rc file for easy updates
  • After running, execute: source ~/.bashrc (or reopen terminal)
  • Requires CUDA and/or TensorRT to be installed on your system

${BLUE}Platform Support:${RESET}
  • Jetson (ARM64): Detects JetPack CUDA & TensorRT
  • x86_64: Detects standard CUDA & TensorRT installations

EOF
}

# Main configuration function
main() {
    local dry_run_flag=""
    local bashrc_flag=""
    local bashrc_file=""

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --help|-h)
                show_usage
                exit 0
                ;;
            --dry-run)
                dry_run_flag="--dry-run"
                shift
                ;;
            --bashrc)
                if [ -z "$2" ] || [[ "$2" == --* ]]; then
                    print_error "Option --bashrc requires an argument"
                    echo ""
                    show_usage
                    exit 1
                fi
                bashrc_file="$2"
                bashrc_flag="--bashrc $2"
                shift 2
                ;;
            *)
                print_error "Unknown option: $1"
                echo ""
                show_usage
                exit 1
                ;;
        esac
    done

    # Check prerequisites
    check_python_script
    check_python

    # Display header
    print_header "CUDA & TensorRT Environment Configuration"

    # Detect platform
    if is_jetson; then
        print_info "Detected NVIDIA Jetson platform"
    else
        print_info "Detected x86 platform"
    fi

    # Show configuration target
    if [ -n "$bashrc_file" ]; then
        print_info "Target shell rc file: $bashrc_file"
    else
        print_info "Target shell rc file: ~/.bashrc"
    fi

    if [ -n "$dry_run_flag" ]; then
        print_warning "Dry-run mode: No files will be modified"
    fi

    echo ""

    # Build Python command with flags
    local python_cmd="python3 \"$PYTHON_SCRIPT\""
    if [ -n "$bashrc_flag" ]; then
        python_cmd="$python_cmd $bashrc_flag"
    fi
    if [ -n "$dry_run_flag" ]; then
        python_cmd="$python_cmd $dry_run_flag"
    fi

    # Run the Python configuration script
    if eval "$python_cmd"; then
        echo ""
        if [ -z "$dry_run_flag" ]; then
            print_success "Configuration completed successfully!"
            echo ""
            if [ -n "$bashrc_file" ]; then
                print_info "To apply changes, run: ${GREEN}source $bashrc_file${RESET}"
            else
                print_info "To apply changes, run: ${GREEN}source ~/.bashrc${RESET}"
            fi
            print_info "Or simply restart your terminal"
        else
            print_success "Dry-run completed - review the output above"
        fi
    else
        echo ""
        print_error "Configuration failed!"
        exit 1
    fi
}

# Run main function
main "$@"
