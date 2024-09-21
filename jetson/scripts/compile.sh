#!/bin/bash

# 引数が lib か app かを確認
if [ "$1" != "lib" ] && [ "$1" != "app" ] && [ "$1" != "demo_4channels" ] && [ "$1" != "demo_yolopose" ]; then
  echo "使い方: $0 [lib|app|demo_4channels|demo_yolopose]"
  exit 1
fi

# 引数に応じて適切なCMakeファイルを選択し、CMakeLists.txtにリネーム
if [ "$1" == "lib" ]; then
  echo "ライブラリモードでビルドしています..."
  cp res/config_cmake/make.lib.txt CMakeLists.txt || { echo "make.lib.txt のコピーに失敗しました"; exit 1; }
elif [ "$1" == "app" ]; then
  echo "アプリケーションモードでビルドしています..."
  cp res/config_cmake/make.app.txt CMakeLists.txt || { echo "make.app.txt のコピーに失敗しました"; exit 1; }
elif [ "$1" == "demo_4channels" ]; then
  echo "テストモードでビルドしています..."
  cp res/config_cmake/make.demo_4channels.txt CMakeLists.txt || { echo "make.test.txt のコピーに失敗しました"; exit 1; }
elif [ "$1" == "demo_yolopose" ]; then
  echo "テストモードでビルドしています..."
  cp res/config_cmake/make.demo_yolopose.txt CMakeLists.txt || { echo "make.test.txt のコピーに失敗しました"; exit 1; }
fi

# もし build ディレクトリが存在しなければ作成し、存在する場合は内容をクリア
if [ ! -d build ]; then
  mkdir build
fi

# build ディレクトリに移動
cd ./build || { echo "build ディレクトリへの移動に失敗しました"; exit 1; }
cmake .. || { echo "CMake の設定に失敗しました"; exit 1; }
make || { echo "プロジェクトのビルドに失敗しました"; exit 1; }

# 生成された jetson_infer を上のディレクトリにコピー
if [ "$1" == "app" ]; then
  EXE_FILE="jetson_infer"
  cp $EXE_FILE ../ || { echo "jetson_infer のコピーに失敗しました"; exit 1; }
elif [ "$1" == "demo_4channels" ]; then
  EXE_FILE="demo_4channels"
  cp $EXE_FILE ../ || { echo "demo_4channels のコピーに失敗しました"; exit 1; }
elif [ "$1" == "demo_yolopose" ]; then
  EXE_FILE="demo_yolopose"
  cp $EXE_FILE ../ || { echo "demo_yolopose のコピーに失敗しました"; exit 1; }
fi

echo "$1 のビルドとコピーが正常に完了しました。"
