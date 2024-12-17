#!/usr/bin/env bash

# This shell script will call the Python synchronization script.
# It will check whether paramiko is installed, and if not, install it via pip3.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/sync_remote.py"

# Check if the Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found at $PYTHON_SCRIPT"
    exit 1
fi

# Check if paramiko is installed
python3 -c "import paramiko" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Paramiko not found. Attempting to install..."
    pip3 install paramiko
    if [ $? -ne 0 ]; then
        echo "Failed to install paramiko. Please install it manually and try again."
        exit 1
    fi
fi

# If paramiko is successfully installed, run the Python script
python3 "$PYTHON_SCRIPT"
