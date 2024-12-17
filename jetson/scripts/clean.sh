#!/bin/bash
set -euo pipefail

# ビルドディレクトリを削除
if [ -d build ]; then
  rm -rf build
  echo "build ディレクトリを削除しました。"
else
  echo "build ディレクトリは存在しません。"
fi

# 不要な生成物を削除
# 存在する場合のみメッセージを表示するために、チェック後に削除
for file in jetson_infer jetson_adapter demo_4channels demo_yolopose demo_autoscaling; do
  if [ -f "$file" ]; then
    rm -f "$file"
    echo "$file を削除しました。"
  fi
done

# 不要なCMakeLists.txtを削除
if [ -f CMakeLists.txt ]; then
  rm -f CMakeLists.txt
  echo "CMakeLists.txtを削除しました。"
fi

echo "クリーンアップが完了しました。"
