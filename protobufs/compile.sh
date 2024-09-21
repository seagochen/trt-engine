#!/bin/bash

# Go to the folder
cd ./protobufs || { echo "protobufs ディレクトリへの移動に失敗しました"; exit 1; }

# 定义 protobuf 文件和对应的输出目录
PROTO_FILES=("video_frame.proto" "inference_result.proto")
PYTHON_OUT_DIR="./pys"
CPP_OUT_DIR="./cpp"

# 创建输出目录（如果不存在）
mkdir -p "$PYTHON_OUT_DIR"
mkdir -p "$CPP_OUT_DIR"

# 编译 protobuf 文件到 Python 和 C++ 文件
for PROTO_FILE in "${PROTO_FILES[@]}"; do
  echo "Processing $PROTO_FILE..."
  protoc --python_out="$PYTHON_OUT_DIR" "$PROTO_FILE"
  protoc --cpp_out="$CPP_OUT_DIR" "$PROTO_FILE"
done

echo "Protobuf files have been compiled and organized."

# コンパイルしたファイルは video_adp と　jetson_infer　へ移動します
cd ..

# まず、フォルダを作る
PYTHON_PROTOBUF_FOLDERS=("video_adp/protobufs")
CPP_PROTOBUF_FOLDERS=("jetson_infer/extend/protobufs" "displayer/protobufs")

for FOLDER in "${PYTHON_PROTOBUF_FOLDERS[@]}"; do
  mkdir -p "$FOLDER"
done

for FOLDER in "${CPP_PROTOBUF_FOLDERS[@]}"; do
  mkdir -p "$FOLDER"
done

# それぞれのフォルダにコピーする
for FOLDER in "${PYTHON_PROTOBUF_FOLDERS[@]}"; do
  cp ./protobufs/pys/* "$FOLDER"
done

for FOLDER in "${CPP_PROTOBUF_FOLDERS[@]}"; do
  cp ./protobufs/cpp/* "$FOLDER"
done

echo "Protobuf files have been copied to video_adp and jetson_infer."