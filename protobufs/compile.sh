#!/bin/bash

# 現在のディレクトリを取得
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
echo "Current directory: $DIR"

# 定義する .proto ファイル
PROTO_FILES=("video_frame.proto" "inference_result.proto")
PROTO_PATH="$DIR"  # Protobuf ファイルのパス

# 確認: .proto ファイルの場所を探す
for PROTO_FILE in "${PROTO_FILES[@]}"; do
  PROTO_FULL_PATH=$(find "$DIR" -name "$PROTO_FILE" 2>/dev/null)
  
  if [ -z "$PROTO_FULL_PATH" ]; then
    echo "Error: $PROTO_FILE not found in $DIR or subdirectories"
    exit 1
  else
    echo "Found $PROTO_FILE at $PROTO_FULL_PATH"
  fi
done

# 出力ディレクトリを作成
PYTHON_OUT_DIR="./pys"
CPP_OUT_DIR="./cpp"

# ディレクトリが存在しない場合は作成
mkdir -p "$PYTHON_OUT_DIR" || { echo "Error: Unable to create Python output directory"; exit 1; }
mkdir -p "$CPP_OUT_DIR" || { echo "Error: Unable to create C++ output directory"; exit 1; }

# PythonとC++用にコンパイル
for PROTO_FILE in "${PROTO_FILES[@]}"; do
  PROTO_FULL_PATH=$(find "$DIR" -name "$PROTO_FILE" 2>/dev/null)
  
  if [ -f "$PROTO_FULL_PATH" ]; then
    echo "Processing $PROTO_FILE..."
    protoc --proto_path="$PROTO_PATH" --python_out="$PYTHON_OUT_DIR" "$PROTO_FULL_PATH" || { echo "Error: Failed to compile $PROTO_FILE for Python"; exit 1; }
    protoc --proto_path="$PROTO_PATH" --cpp_out="$CPP_OUT_DIR" "$PROTO_FULL_PATH" || { echo "Error: Failed to compile $PROTO_FILE for C++"; exit 1; }
  else
    echo "Error: $PROTO_FILE not found during processing"
    exit 1
  fi
done

# Protobufファイルをコピーした後、PythonとC++用に整理
echo "Protobuf files have been compiled and organized."
cd "$DIR/.." || { echo "Error: Unable to change directory to parent"; exit 1; }

# 目標のフォルダを作成
# PYTHON_PROTOBUF_FOLDERS=("video_adp/protobufs")
CPP_PROTOBUF_FOLDERS=("jetson/extend/protobufs" "displayer/protobufs")

# for FOLDER in "${PYTHON_PROTOBUF_FOLDERS[@]}"; do
#   mkdir -p "$FOLDER" || { echo "Error: Unable to create folder $FOLDER"; exit 1; }
# done

for FOLDER in "${CPP_PROTOBUF_FOLDERS[@]}"; do
  mkdir -p "$FOLDER" || { echo "Error: Unable to create folder $FOLDER"; exit 1; }
done

# それぞれのフォルダにコピーする
# if compgen -G "$PYTHON_OUT_DIR/*" > /dev/null; then
#   for FOLDER in "${PYTHON_PROTOBUF_FOLDERS[@]}"; do
#     cp "$PYTHON_OUT_DIR"/* "$FOLDER" || { echo "Error: Failed to copy Python protobuf files to $FOLDER"; exit 1; }
#   done
# else
#   echo "Warning: No Python protobuf files found to copy."
# fi

if compgen -G "$CPP_OUT_DIR/*" > /dev/null; then
  for FOLDER in "${CPP_PROTOBUF_FOLDERS[@]}"; do
    cp "$CPP_OUT_DIR"/* "$FOLDER" || { echo "Error: Failed to copy C++ protobuf files to $FOLDER"; exit 1; }
  done
else
  echo "Warning: No C++ protobuf files found to copy."
fi

# 削除生成されたファイルフォルダ
rm -rf "$PYTHON_OUT_DIR" "$CPP_OUT_DIR" || { echo "Error: Failed to delete generated output directories"; exit 1; }

echo "Protobuf files have been copied to video_adp and jetson, and output directories have been deleted."
