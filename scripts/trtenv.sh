#!/bin/bash
#
# TrtEnv - TensorRT Environment Setup Tool (Shell Wrapper)
#
# This script wraps the Python trtenv.py script for easier usage.
# It provides comprehensive environment configuration for TensorRT and CUDA.
#
# Usage:
#   sudo ./trtenv.sh [OPTION]
#
# Options:
#   --setup-all           Configure everything (recommended)
#   --configure-env       Configure shell environment variables only
#   --ldconfig            Configure system ldconfig only
#   --install-monitor     Install performance monitoring tools only
#   --show                Show current configuration
#   --remove              Remove ldconfig configuration
#   --dry-run             Preview changes without modifying files
#   --help, -h            Show this help message
#
# Author: TrtEngineToolkits
# Date: 2025-11-14
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
PYTHON_SCRIPT="${SCRIPT_DIR}/trtenv.py"

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

# Function to check if running on Linux
check_linux() {
    if [[ "$(uname -s)" != "Linux" ]]; then
        print_error "This script only works on Linux systems"
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

# Function to check Python script
check_python_script() {
    if [ ! -f "$PYTHON_SCRIPT" ]; then
        print_error "Python script not found: $PYTHON_SCRIPT"
        exit 1
    fi
}

# Function to display usage
show_usage() {
    echo -e "${GREEN}TrtEnv - TensorRT Environment Setup Tool${RESET}"
    echo ""
    echo -e "${BLUE}Description:${RESET}"
    echo "  Comprehensive environment configuration for TensorRT and CUDA:"
    echo "    1. Configure shell environment variables (PATH, LD_LIBRARY_PATH)"
    echo "    2. Configure system library search paths (ldconfig)"
    echo "    3. Install performance monitoring tools (jtop/nvitop)"
    echo ""
    echo -e "${BLUE}Usage:${RESET}"
    echo "  sudo $0 [OPTION]"
    echo ""
    echo -e "${BLUE}Options:${RESET}"
    echo -e "  ${CYAN}Main Actions:${RESET}"
    echo "    --setup-all           Configure everything (recommended)"
    echo "    --configure-env       Configure shell environment variables only"
    echo "    --ldconfig            Configure system ldconfig only"
    echo "    --install-monitor     Install performance monitoring tools only"
    echo "    --show                Show current configuration"
    echo "    --remove              Remove ldconfig configuration"
    echo ""
    echo -e "  ${CYAN}Modifiers:${RESET}"
    echo "    --dry-run             Preview changes without modifying files"
    echo "    --bashrc PATH         Specify shell rc file (default: ~/.bashrc)"
    echo "    --lib-dirs DIR...     Specify library directories for ldconfig"
    echo ""
    echo -e "  ${CYAN}Help:${RESET}"
    echo "    --help, -h            Show this help message"
    echo ""
    echo -e "${BLUE}Examples:${RESET}"
    echo -e "  ${GREEN}Setup everything (recommended):${RESET}"
    echo "    sudo $0 --setup-all"
    echo ""
    echo -e "  ${GREEN}Preview changes:${RESET}"
    echo "    sudo $0 --setup-all --dry-run"
    echo ""
    echo -e "  ${GREEN}Configure shell environment only:${RESET}"
    echo "    $0 --configure-env"
    echo ""
    echo -e "  ${GREEN}Configure ldconfig only:${RESET}"
    echo "    sudo $0 --ldconfig"
    echo ""
    echo -e "  ${GREEN}Install monitoring tools:${RESET}"
    echo "    $0 --install-monitor"
    echo ""
    echo -e "  ${GREEN}Show current configuration:${RESET}"
    echo "    $0 --show"
    echo ""
    echo -e "  ${GREEN}Remove ldconfig configuration:${RESET}"
    echo "    sudo $0 --remove"
    echo ""
    echo -e "  ${GREEN}Use custom shell rc file:${RESET}"
    echo "    $0 --configure-env --bashrc ~/.zshrc"
    echo ""
    echo -e "  ${GREEN}Specify custom library directories:${RESET}"
    echo "    sudo $0 --ldconfig --lib-dirs /opt/tensorrt/lib /usr/local/cuda/lib64"
    echo ""
    echo -e "${BLUE}Notes:${RESET}"
    echo "  • Some operations require root privileges (sudo)"
    echo "  • --setup-all is the recommended way to configure everything"
    echo "  • Use --dry-run to preview changes before applying them"
    echo "  • After configuration, run 'source ~/.bashrc' or reopen terminal"
    echo ""
    echo -e "${BLUE}What does each module do:${RESET}"
    echo -e "  ${CYAN}1. Shell Environment (--configure-env):${RESET}"
    echo "     - Auto-detects CUDA and TensorRT installations"
    echo "     - Adds bin/ directories to PATH"
    echo "     - Adds lib/ directories to LD_LIBRARY_PATH"
    echo "     - Modifies ~/.bashrc (or specified shell rc file)"
    echo ""
    echo -e "  ${CYAN}2. System Ldconfig (--ldconfig):${RESET}"
    echo "     - Registers library paths in /etc/ld.so.conf.d/tensorrt.conf"
    echo "     - Runs ldconfig to update system library cache"
    echo "     - Requires root privileges"
    echo ""
    echo -e "  ${CYAN}3. Monitoring Tools (--install-monitor):${RESET}"
    echo "     - Installs jetson-stats (jtop) on Jetson platforms"
    echo "     - Installs nvitop on x86 platforms"
    echo "     - Provides GPU/CPU monitoring capabilities"
    echo ""
}

# Function to run Python script
run_python_script() {
    python3 "$PYTHON_SCRIPT" "$@"
}

# Main script logic
main() {
    # Check prerequisites
    check_linux
    check_python_script
    check_python

    # If no arguments, show help
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi

    # Parse for help
    for arg in "$@"; do
        if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
            show_usage
            exit 0
        fi
    done

    # Check if root is needed
    needs_root=false
    for arg in "$@"; do
        case "$arg" in
            --setup-all|--ldconfig|--remove)
                needs_root=true
                break
                ;;
        esac
    done

    # Warn if root needed but not running as root (unless dry-run)
    if $needs_root && [ "$EUID" -ne 0 ]; then
        # Check if --dry-run is present
        has_dry_run=false
        for arg in "$@"; do
            if [[ "$arg" == "--dry-run" ]]; then
                has_dry_run=true
                break
            fi
        done

        if ! $has_dry_run; then
            print_warning "Some operations require root privileges"
            print_info "Please run with sudo: sudo $0 $*"
        fi
    fi

    # Run Python script with all arguments
    print_header "TrtEnv - TensorRT Environment Setup"
    run_python_script "$@"
    exit_code=$?

    echo ""
    if [ $exit_code -eq 0 ]; then
        print_success "Operation completed successfully"
    else
        print_error "Operation failed with exit code: $exit_code"
    fi

    exit $exit_code
}

# Run main function
main "$@"
