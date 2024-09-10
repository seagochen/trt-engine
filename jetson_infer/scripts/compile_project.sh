#!/bin/bash

# もし build ディレクトリが存在しなければ作成し、存在する場合は内容をクリア
if [ ! -d cmake-build-debug ]; then
  mkdir cmake-build-debug
fi

EXE_FILE="jetson_infer"

# cmake-build-debug ディレクトリに移動
cd ./cmake-build-debug || { echo "cmake-build-debug ディレクトリへの移動に失敗しました"; exit 1; }
cmake .. || { echo "CMake の設定に失敗しました"; exit 1; }
make || { echo "プロジェクトのビルドに失敗しました"; exit 1; }

# 生成された jetson_infer を一つ上のディレクトリにコピー
cp $EXE_FILE ../ || { echo "jetson_infer のコピーに失敗しました"; exit 1; }

echo "ビルドとコピーが正常に完了しました。"
