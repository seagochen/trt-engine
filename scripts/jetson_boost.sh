#!/bin/bash
#
# Jetson Boost - Performance & Fan Control Tool
#
# This script wraps the Python jetson_boost.py script for easier usage.
# It provides quick access to common Jetson performance tuning operations.
#
# Usage:
#   sudo ./jetson_boost.sh [OPTION]
#
# Options:
#   --interactive, -i     Interactive menu (default)
#   --maxn                Set to MAXN mode (maximum performance)
#   --max-clocks          Maximize clocks (CUDA cores)
#   --restore             Restore to saved clock state
#   --fan-on              Force fan to maximum speed
#   --fan-off             Restore fan to auto mode
#   --fan-status          Show current fan status
#   --enable-autostart    Enable fan auto-start at boot
#   --disable-autostart   Disable fan auto-start at boot
#   --show                Show current clock status
#   --help, -h            Show this help message
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
PYTHON_SCRIPT="${SCRIPT_DIR}/jetson_boost.py"

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
check_jetson() {
    if [ -f "/etc/nv_tegra_release" ]; then
        return 0
    fi

    if [ -f "/proc/device-tree/compatible" ]; then
        if grep -qi "tegra" /proc/device-tree/compatible; then
            return 0
        fi
    fi

    print_error "This script is designed for NVIDIA Jetson platforms only."
    exit 1
}

# Function to check root privileges
check_root() {
    if [ "$EUID" -ne 0 ]; then
        print_error "This script must be run with root privileges."
        echo "Please use: sudo $0"
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
${GREEN}Jetson Boost - Performance & Fan Control${RESET}

${BLUE}Usage:${RESET}
  sudo $0 [OPTION]

${BLUE}Options:${RESET}
  ${CYAN}Interactive Mode:${RESET}
    --interactive, -i       Launch interactive menu (default)

  ${CYAN}Power Mode Control:${RESET}
    --maxn                  Set power mode to MAXN (maximum performance)

  ${CYAN}Clock Control (CUDA Cores):${RESET}
    --max-clocks            Maximize CPU/GPU clocks and enable all CUDA cores
    --restore               Restore clocks to previously saved state
    --show                  Display current clock status

  ${CYAN}Fan Control:${RESET}
    --fan-on                Force fan to maximum speed (255 PWM)
    --fan-off               Restore fan to automatic control mode
    --fan-status            Show current fan speed and PWM value

  ${CYAN}Auto-Start Control:${RESET}
    --enable-autostart      Enable fan auto-start at system boot
    --disable-autostart     Disable fan auto-start at system boot

  ${CYAN}Help:${RESET}
    --help, -h              Show this help message

${BLUE}Examples:${RESET}
  ${GREEN}Interactive menu:${RESET}
    sudo $0

  ${GREEN}Maximum performance setup:${RESET}
    sudo $0 --maxn              # Set power mode to MAXN
    sudo $0 --max-clocks        # Maximize clocks/CUDA cores
    sudo $0 --fan-on            # Force fan on

  ${GREEN}Fan control:${RESET}
    sudo $0 --fan-on            # Turn fan on (max speed)
    sudo $0 --fan-off           # Return to auto mode
    sudo $0 --fan-status        # Check fan status

  ${GREEN}Auto-start at boot:${RESET}
    sudo $0 --enable-autostart  # Fan starts automatically at boot
    sudo $0 --disable-autostart # Disable auto-start

  ${GREEN}Restore normal state:${RESET}
    sudo $0 --restore           # Restore clocks
    sudo $0 --fan-off           # Fan to auto mode

${BLUE}Notes:${RESET}
  • This script requires root privileges (sudo)
  • Only works on NVIDIA Jetson platforms
  • Power mode changes (--maxn) may require a reboot
  • Fan and clock controls are independent
  • Use --restore to return to normal clock state

${BLUE}Requirements:${RESET}
  • nvpmodel (power mode control)
  • jetson_clocks (clock control)
  • PWM fan interface in /sys/devices/pwm-fan or /sys/class/hwmon

EOF
}

# Function to run interactive mode
run_interactive() {
    print_header "Jetson Boost - Interactive Mode"
    python3 "$PYTHON_SCRIPT"
}

# Function to set MAXN mode
set_maxn() {
    print_header "Setting MAXN Performance Mode"

    if ! command -v nvpmodel &> /dev/null; then
        print_error "nvpmodel command not found"
        print_info "Install with: sudo apt-get install -y nvpmodel"
        exit 1
    fi

    print_info "Setting nvpmodel to mode 0 (MAXN)..."
    nvpmodel -m 0

    if [ $? -eq 0 ]; then
        print_success "Power mode set to MAXN"
        echo ""
        print_warning "A system reboot is recommended for changes to take full effect."
        read -p "Reboot now? (y/N): " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Rebooting system..."
            reboot
        fi
    else
        print_error "Failed to set power mode"
        exit 1
    fi
}

# Function to maximize clocks
maximize_clocks() {
    print_header "Maximizing Clocks and Enabling All CUDA Cores"

    if ! command -v jetson_clocks &> /dev/null; then
        print_error "jetson_clocks command not found"
        print_info "This utility is part of JetPack. Install Jetson developer tools via SDK Manager."
        exit 1
    fi

    # Store current settings first if not already stored
    local config_file="/root/.jetsonclocks_conf.txt"
    if [ ! -f "$config_file" ]; then
        print_info "Storing current clock settings..."
        jetson_clocks --store
    fi

    print_info "Maximizing CPU/GPU clocks and enabling all CUDA cores..."
    jetson_clocks

    if [ $? -eq 0 ]; then
        print_success "Clocks maximized and all CUDA cores enabled"
        echo ""
        print_info "Current clock status:"
        jetson_clocks --show
    else
        print_error "Failed to maximize clocks"
        exit 1
    fi
}

# Function to force fan on
fan_on() {
    print_header "Forcing Fan to Maximum Speed"
    print_info "Calling Python script to force fan on..."
    python3 "$PYTHON_SCRIPT" --fan-on
}

# Function to turn fan off (auto mode)
fan_off() {
    print_header "Restoring Fan to Automatic Control"
    print_info "Calling Python script to restore fan to auto mode..."
    python3 "$PYTHON_SCRIPT" --fan-off
}

# Function to show fan status
fan_status() {
    print_header "Fan Status"
    python3 "$PYTHON_SCRIPT" --fan-status
}

# Function to enable fan autostart
enable_autostart() {
    print_header "Enabling Fan Auto-Start at Boot"
    print_info "This will create a systemd service to start the fan automatically at boot..."
    python3 "$PYTHON_SCRIPT" --interactive << 'EOF'
9
0
EOF
    print_success "Fan auto-start configuration completed"
    print_info "The fan will automatically start at maximum speed on next boot"
}

# Function to disable fan autostart
disable_autostart() {
    print_header "Disabling Fan Auto-Start at Boot"
    print_info "This will disable the systemd service for fan auto-start..."
    python3 "$PYTHON_SCRIPT" --interactive << 'EOF'
a
0
EOF
    print_success "Fan auto-start disabled"
}

# Function to restore clocks
restore_clocks() {
    print_header "Restoring Clock Settings"

    if ! command -v jetson_clocks &> /dev/null; then
        print_error "jetson_clocks command not found"
        exit 1
    fi

    local config_file="/root/.jetsonclocks_conf.txt"
    if [ ! -f "$config_file" ]; then
        print_error "No saved clock configuration found at $config_file"
        print_info "Run '$0 --max-clocks' first to create a baseline configuration"
        exit 1
    fi

    print_info "Restoring clocks to saved state..."
    jetson_clocks --restore

    if [ $? -eq 0 ]; then
        print_success "Clocks restored successfully"
    else
        print_error "Failed to restore clocks"
        exit 1
    fi
}

# Function to show clock status
show_status() {
    print_header "Current Clock Status"

    # Show nvpmodel status
    if command -v nvpmodel &> /dev/null; then
        print_info "Power Mode (nvpmodel):"
        nvpmodel -q
        echo ""
    fi

    # Show jetson_clocks status
    if command -v jetson_clocks &> /dev/null; then
        print_info "Clock Status (jetson_clocks):"
        jetson_clocks --show
    else
        print_warning "jetson_clocks not available"
    fi
}

# Main script logic
main() {
    # Check prerequisites
    check_jetson
    check_root
    check_python_script
    check_python

    # Parse command line arguments
    case "${1:-}" in
        --interactive|-i|"")
            run_interactive
            ;;
        --maxn)
            set_maxn
            ;;
        --max-clocks)
            maximize_clocks
            ;;
        --restore)
            restore_clocks
            ;;
        --fan-on)
            fan_on
            ;;
        --fan-off)
            fan_off
            ;;
        --fan-status)
            fan_status
            ;;
        --enable-autostart)
            enable_autostart
            ;;
        --disable-autostart)
            disable_autostart
            ;;
        --show)
            show_status
            ;;
        --help|-h)
            show_usage
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
