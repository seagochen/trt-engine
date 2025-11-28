#!/bin/bash
#
# PyEngine Installer
#
# Install or update the pyengine Python package from the local project.
# Supports both development mode (editable) and standard installation.
#
# Usage:
#   ./install_pyengine.sh              # Install in editable mode (default)
#   ./install_pyengine.sh --editable   # Install in editable/development mode
#   ./install_pyengine.sh --release    # Install in standard mode
#   ./install_pyengine.sh --uninstall  # Uninstall pyengine
#   ./install_pyengine.sh --check      # Check installation status
#   ./install_pyengine.sh -h, --help   # Show help
#
# Author: TrtEngineToolkits
# Date: 2025-11-28
#

# ----------------------------
# Color codes and utilities
# ----------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
CYAN='\033[0;36m'
RESET='\033[0m'

print_info() {
    echo -e "${BLUE}[INFO]${RESET} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${RESET} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${RESET} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${RESET} $1" >&2
}

print_header() {
    echo -e "${CYAN}==================================================${RESET}"
    echo -e "${CYAN}$1${RESET}"
    echo -e "${CYAN}==================================================${RESET}"
}

# ----------------------------
# Get project root directory
# ----------------------------
get_project_root() {
    # Default installation path (highest priority)
    local default_path="/opt/TrtEngineToolkits"
    if [[ -d "$default_path/pyengine" ]] && [[ -f "$default_path/pyproject.toml" ]]; then
        echo "$default_path"
        return 0
    fi

    # Try script's parent directory (scripts/../)
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local parent_dir
    parent_dir="$(cd "$script_dir/.." && pwd)"

    if [[ -f "$parent_dir/pyproject.toml" ]]; then
        echo "$parent_dir"
        return 0
    fi

    # Try current working directory
    if [[ -f "$(pwd)/pyproject.toml" ]]; then
        echo "$(pwd)"
        return 0
    fi

    # Fallback to default path even if not found (will show error later)
    echo "$default_path"
}

# ----------------------------
# Check Python and pip
# ----------------------------
check_python() {
    local python_cmd=""

    # Try python3 first, then python
    if command -v python3 &>/dev/null; then
        python_cmd="python3"
    elif command -v python &>/dev/null; then
        python_cmd="python"
    else
        print_error "Python is not installed or not in PATH."
        return 1
    fi

    # Check Python version (need >= 3.8)
    local version
    version=$($python_cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)

    if [[ -z "$version" ]]; then
        print_error "Could not determine Python version."
        return 1
    fi

    local major minor
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)

    if [[ $major -lt 3 ]] || { [[ $major -eq 3 ]] && [[ $minor -lt 8 ]]; }; then
        print_error "Python >= 3.8 is required. Found: $version"
        return 1
    fi

    print_info "Found Python $version ($python_cmd)"
    PYTHON_CMD="$python_cmd"
    return 0
}

check_pip() {
    local pip_cmd=""

    # Try pip3 first, then pip, then python -m pip
    if command -v pip3 &>/dev/null; then
        pip_cmd="pip3"
    elif command -v pip &>/dev/null; then
        pip_cmd="pip"
    elif $PYTHON_CMD -m pip --version &>/dev/null; then
        pip_cmd="$PYTHON_CMD -m pip"
    else
        print_error "pip is not installed. Please install pip first."
        return 1
    fi

    print_info "Found pip: $pip_cmd"
    PIP_CMD="$pip_cmd"
    return 0
}

# ----------------------------
# Check pyproject.toml exists
# ----------------------------
check_pyproject() {
    local project_root="$1"
    local pyproject_file="$project_root/pyproject.toml"

    if [[ ! -f "$pyproject_file" ]]; then
        print_error "pyproject.toml not found at: $pyproject_file"
        return 1
    fi

    # Check pyengine directory exists
    if [[ ! -d "$project_root/pyengine" ]]; then
        print_error "pyengine directory not found at: $project_root/pyengine"
        return 1
    fi

    print_info "Found pyproject.toml: $pyproject_file"
    return 0
}

# ----------------------------
# Install pyengine
# ----------------------------
install_pyengine() {
    local project_root="$1"
    local editable="$2"

    print_header "Installing PyEngine"

    # Check prerequisites
    if ! check_python; then
        return 1
    fi

    if ! check_pip; then
        return 1
    fi

    if ! check_pyproject "$project_root"; then
        return 1
    fi

    echo ""

    # Build install command
    local -a install_cmd=($PIP_CMD install)

    if [[ "$editable" == "true" ]]; then
        install_cmd+=("-e")
        print_info "Installing in editable (development) mode..."
        print_info "Changes to source files will take effect immediately."
    else
        print_info "Installing in standard (release) mode..."
    fi

    install_cmd+=("$project_root")

    # Show command
    echo ""
    print_info "Running: ${install_cmd[*]}"
    echo ""

    # Execute installation
    if "${install_cmd[@]}"; then
        echo ""
        print_success "PyEngine installed successfully!"
        echo ""

        # Show installation info
        show_install_info
        return 0
    else
        echo ""
        print_error "Installation failed."
        return 1
    fi
}

# ----------------------------
# Uninstall pyengine
# ----------------------------
uninstall_pyengine() {
    print_header "Uninstalling PyEngine"

    if ! check_python; then
        return 1
    fi

    if ! check_pip; then
        return 1
    fi

    # Check if pyengine is installed
    if ! $PIP_CMD show pyengine &>/dev/null; then
        print_warn "pyengine is not installed."
        return 0
    fi

    echo ""
    print_info "Running: $PIP_CMD uninstall pyengine -y"
    echo ""

    if $PIP_CMD uninstall pyengine -y; then
        echo ""
        print_success "PyEngine uninstalled successfully!"
        return 0
    else
        echo ""
        print_error "Uninstallation failed."
        return 1
    fi
}

# ----------------------------
# Check installation status
# ----------------------------
check_installation() {
    print_header "PyEngine Installation Status"

    if ! check_python; then
        return 1
    fi

    if ! check_pip; then
        return 1
    fi

    echo ""

    # Check if pyengine is installed
    local pip_show_output
    pip_show_output=$($PIP_CMD show pyengine 2>/dev/null)

    if [[ -z "$pip_show_output" ]]; then
        print_warn "pyengine is NOT installed."
        echo ""
        print_info "To install, run: $0 --editable"
        return 1
    fi

    print_success "pyengine is installed."
    echo ""
    echo "$pip_show_output"
    echo ""

    # Check if it's an editable installation
    local location
    location=$(echo "$pip_show_output" | grep "^Location:" | cut -d: -f2 | xargs)
    local editable_location
    editable_location=$(echo "$pip_show_output" | grep "^Editable project location:" | cut -d: -f2- | xargs)

    if [[ -n "$editable_location" ]]; then
        print_info "Installation type: Editable (development mode)"
        print_info "Source location: $editable_location"
    else
        print_info "Installation type: Standard (release mode)"
        print_info "Installed at: $location"
    fi

    echo ""

    # Test import
    print_info "Testing import..."
    if $PYTHON_CMD -c "import pyengine; print(f'  pyengine imported successfully')" 2>/dev/null; then
        print_success "Import test passed!"
    else
        print_error "Import test failed!"
        return 1
    fi

    return 0
}

# ----------------------------
# Show installation info
# ----------------------------
show_install_info() {
    if ! $PIP_CMD show pyengine &>/dev/null; then
        return 1
    fi

    print_info "Installation details:"
    $PIP_CMD show pyengine | grep -E "^(Name|Version|Location|Editable)"

    echo ""
    print_info "You can now use pyengine in Python:"
    echo "  >>> import pyengine"
    echo "  >>> from pyengine.visualization import InferenceDrawer"
    echo "  >>> from pyengine.inference import YoloPosePipelineV2"
}

# ----------------------------
# Upgrade pyengine
# ----------------------------
upgrade_pyengine() {
    local project_root="$1"
    local editable="$2"

    print_header "Upgrading PyEngine"

    if ! check_python; then
        return 1
    fi

    if ! check_pip; then
        return 1
    fi

    # Check if pyengine is already installed
    if $PIP_CMD show pyengine &>/dev/null; then
        print_info "pyengine is already installed. Upgrading..."

        # For editable installs, just reinstall
        # For standard installs, use --upgrade
        if [[ "$editable" == "true" ]]; then
            install_pyengine "$project_root" "true"
        else
            echo ""
            print_info "Running: $PIP_CMD install --upgrade $project_root"
            echo ""

            if $PIP_CMD install --upgrade "$project_root"; then
                echo ""
                print_success "PyEngine upgraded successfully!"
                show_install_info
                return 0
            else
                echo ""
                print_error "Upgrade failed."
                return 1
            fi
        fi
    else
        print_info "pyengine is not installed. Installing..."
        install_pyengine "$project_root" "$editable"
    fi
}

# ----------------------------
# Help
# ----------------------------
show_usage() {
    cat << EOF
${GREEN}PyEngine Installer${RESET}

${BLUE}Usage:${RESET}
  $0                     Install pyengine in editable mode (default)
  $0 --editable, -e      Install in editable/development mode
  $0 --release, -r       Install in standard/release mode
  $0 --upgrade, -u       Upgrade existing installation
  $0 --uninstall         Uninstall pyengine
  $0 --check, -c         Check installation status
  $0 --help, -h          Show this help message

${BLUE}Installation Modes:${RESET}
  ${GREEN}Editable mode${RESET} (recommended for development):
    - Source files are linked, not copied
    - Changes to .py files take effect immediately
    - No need to reinstall after code changes
    - Use: $0 --editable

  ${GREEN}Release mode${RESET} (for production):
    - Package is copied to site-packages
    - More stable, isolated from source changes
    - Requires reinstall to update
    - Use: $0 --release

${BLUE}Examples:${RESET}
  $0                     # Install for development
  $0 --editable          # Same as above
  $0 --release           # Install for production
  $0 --check             # Check if installed
  $0 --upgrade           # Update to latest
  $0 --uninstall         # Remove pyengine

${BLUE}Requirements:${RESET}
  - Python >= 3.8
  - pip

${BLUE}Dependencies${RESET} (installed automatically):
  - numpy
  - opencv-python
  - Pillow
  - paho-mqtt
  - scipy
  - filterpy
  - numba

EOF
}

# ----------------------------
# Main
# ----------------------------
main() {
    local project_root
    project_root=$(get_project_root)

    case "${1:-}" in
        -h|--help)
            show_usage
            exit 0
            ;;
        -e|--editable)
            install_pyengine "$project_root" "true"
            exit $?
            ;;
        -r|--release)
            install_pyengine "$project_root" "false"
            exit $?
            ;;
        -u|--upgrade)
            # Default to editable mode for upgrade
            upgrade_pyengine "$project_root" "true"
            exit $?
            ;;
        --uninstall)
            uninstall_pyengine
            exit $?
            ;;
        -c|--check)
            check_installation
            exit $?
            ;;
        "")
            # Default: editable mode
            install_pyengine "$project_root" "true"
            exit $?
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_usage
            exit 1
            ;;
    esac
}

main "$@"
