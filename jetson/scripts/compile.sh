#!/bin/bash

# Check arguments, if no argument is provided, display an error
if [ $# -ne 1 ]; then
  echo "Usage: $0 [lib|jetson_adapter|jetson_infer|demo_4channels|demo_yolopose]"
  exit 1
fi

# Select the appropriate CMake file based on the argument and rename it to CMakeLists.txt
if [ "$1" == "lib" ]; then
  echo "Building in library mode..."
  cp cmakes/make.lib.txt CMakeLists.txt || { echo "Failed to copy make.lib.txt"; exit 1; }
elif [ "$1" == "jetson_adapter" ]; then
  echo "Building in adapter mode..."
  cp cmakes/make.jetson_adapter.txt CMakeLists.txt || { echo "Failed to copy make.jetson_adapter.txt"; exit 1; }
elif [ "$1" == "jetson_infer" ]; then
  echo "Building in application mode..."
  cp cmakes/make.jetson_infer.txt CMakeLists.txt || { echo "Failed to copy make.jetson_infer.txt"; exit 1; }
elif [ "$1" == "demo_4channels" ]; then
  echo "Building in test mode (4 channels)..."
  cp cmakes/make.demo_4channels.txt CMakeLists.txt || { echo "Failed to copy make.demo_4channels.txt"; exit 1; }
else
  echo "Invalid argument: $1"
  exit 1
fi

# If the build directory does not exist, create it. If it exists, clear its contents.
if [ ! -d build ]; then
  mkdir build
fi

# Navigate to the build directory
cd ./build || { echo "Failed to navigate to build directory"; exit 1; }
cmake .. || { echo "CMake configuration failed"; exit 1; }
make || { echo "Project build failed"; exit 1; }

# Copy the generated executable to the upper directory
if [ "$1" == "jetson_infer" ]; then
  EXE_FILE="jetson_infer"
  cp $EXE_FILE ../ || { echo "Failed to copy jetson_infer"; exit 1; }
elif [ "$1" == "jetson_adapter" ]; then
  EXE_FILE="jetson_adapter"
  cp $EXE_FILE ../ || { echo "Failed to copy jetson_adapter"; exit 1; }
elif [ "$1" == "demo_4channels" ]; then
  EXE_FILE="demo_4channels"
  cp $EXE_FILE ../ || { echo "Failed to copy demo_4channels"; exit 1; }
elif [ "$1" == "demo_yolopose" ]; then
  EXE_FILE="demo_yolopose"
  cp $EXE_FILE ../ || { echo "Failed to copy demo_yolopose"; exit 1; }
fi

echo "Compiling has successfully finished."
