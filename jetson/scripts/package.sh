#!/bin/bash

# 打包前のクリーンアップスクリプトを実行
bash scripts/clean_project.sh

# クリーンアップが成功したか確認
if [ $? -ne 0 ]; then
    echo "クリーンアップスクリプトの実行に失敗しました。"
    exit 1
fi

# 打包する対象のディレクトリとファイルを指定
EXCLUDE_DIRS="--exclude=models --exclude=res --exclude=tarballs --exclude=tensorrt --exclude=cmake-build-debug"
TAR_FILE="jetson_infer.tar.gz"

# tarコマンドを使ってアーカイブを作成
tar -czvf $TAR_FILE $EXCLUDE_DIRS ./*

# パッケージが正常に作成されたか確認
if [ $? -eq 0 ]; then
    echo "プロジェクトのパッケージ化が成功しました: $TAR_FILE"
else
    echo "プロジェクトのパッケージ化に失敗しました。"
    exit 1
fi
