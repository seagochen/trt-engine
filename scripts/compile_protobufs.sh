#!/bin/bash

# 当前脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
PROTOBUF_DIR="$PROJECT_ROOT/protobufs"

# 定义 Protobuf 文件及其输出路径
PROTO_FILES=("video_frame.proto" "inference_result.proto")
PYTHON_OUT_DIR="$PROJECT_ROOT/displayer/protobufs"
CPP_OUT_DIR="$PROJECT_ROOT/jetson/extend/protobufs"

# 检查 Protobuf 编译器是否存在
if ! command -v protoc &> /dev/null; then
  echo "Error: protoc is not installed or not in PATH"
  exit 1
fi

# 检查并创建目标目录
mkdir -p "$PYTHON_OUT_DIR" || { echo "Error: Unable to create Python output directory"; exit 1; }
mkdir -p "$CPP_OUT_DIR" || { echo "Error: Unable to create C++ output directory"; exit 1; }

# 编译 .proto 文件
for PROTO_FILE in "${PROTO_FILES[@]}"; do
  PROTO_PATH="$PROTOBUF_DIR/$PROTO_FILE"
  
  if [ ! -f "$PROTO_PATH" ]; then
    echo "Error: $PROTO_FILE not found in $PROTOBUF_DIR"
    exit 1
  fi
  
  echo "Compiling $PROTO_FILE..."
  
  # 编译 Python
  protoc --proto_path="$PROTOBUF_DIR" --python_out="$PYTHON_OUT_DIR" "$PROTO_PATH" || {
    echo "Error: Failed to compile $PROTO_FILE for Python"
    exit 1
  }
  
  # 编译 C++
  protoc --proto_path="$PROTOBUF_DIR" --cpp_out="$CPP_OUT_DIR" "$PROTO_PATH" || {
    echo "Error: Failed to compile $PROTO_FILE for C++"
    exit 1
  }
done

# 清理旧的 .pyc 文件
echo "Cleaning up old Python compiled files..."
find "$PYTHON_OUT_DIR" -name "*.pyc" -delete
find "$PYTHON_OUT_DIR" -name "__pycache__" -type d -exec rm -rf {} +

echo "Protobuf files compiled successfully!"
echo "Python files are in: $PYTHON_OUT_DIR"
echo "C++ files are in: $CPP_OUT_DIR"
