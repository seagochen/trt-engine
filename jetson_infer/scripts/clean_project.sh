#!/bin/bash

# ビルドディレクトリを削除
if [ -d build ]; then
  rm -rf build
  echo "build ディレクトリを削除しました。"
else
  echo "build ディレクトリは存在しません。"
fi

# cumatrix ディレクトリのクリーンアップ
cd cumatrix || { echo "cumatrix ディレクトリへの移動に失敗しました"; exit 1; }
make clean || { echo "cumatrix のクリーンアップに失敗しました"; exit 1; }
echo "cumatrix のクリーンアップが完了しました。"

# 上位ディレクトリに戻る
cd ..

# cnn_toolkits ディレクトリのクリーンアップ
cd cnn_toolkits || { echo "cnn_toolkits ディレクトリへの移動に失敗しました"; exit 1; }
make clean || { echo "cnn_toolkits のクリーンアップに失敗しました"; exit 1; }
echo "cnn_toolkits のクリーンアップが完了しました。"

# 上位ディレクトリに戻る
cd ..

# 不要な生成物を削除 (例: コピーされた jetson_infer)
if [ -f jetson_infer ]; then
  rm jetson_infer
  echo "jetson_infer を削除しました。"
else
  echo "jetson_infer は存在しません。"
fi

echo "クリーンアップが完了しました。"
