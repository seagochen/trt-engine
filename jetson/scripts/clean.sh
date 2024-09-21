#!/bin/bash

# ビルドディレクトリを削除
if [ -d build ]; then
  rm -rf build
  echo "build ディレクトリを削除しました。"
else
  echo "build ディレクトリは存在しません。"
fi

# 不要な生成物を削除 (例: コピーされた jetson_infer)
if [ -f jetson_infer ]; then
  rm jetson_infer
  echo "jetson_infer を削除しました。"
fi

if [ -f demo_4channels ]; then
  rm demo_4channels
  echo "demo_4channels を削除しました。"
fi

if [ -f demo_yolopose ]; then
  rm demo_yolopose
  echo "demo_yolopose を削除しました。"
fi

echo "クリーンアップが完了しました。"
