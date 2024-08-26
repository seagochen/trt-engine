#!/bin/bash

# 使用方法を表示する関数
usage() {
    echo "Usage: $0 [directory_path]"
    exit 1
}

# 引数が正しく指定されているか確認
if [ "$#" -ne 1 ]; then
    usage
fi

# 指定されたディレクトリパスを変数に代入
DIRECTORY=$1

# ディレクトリが存在するか確認
if [ ! -d "$DIRECTORY" ]; then
    echo "Error: The directory '$DIRECTORY' does not exist."
    exit 1
fi

# 指定されたディレクトリ内のファイルとフォルダのサイズを計測し、サイズ順に表示
du -sh "$DIRECTORY"/.[!.]* "$DIRECTORY"/* | sort -h

