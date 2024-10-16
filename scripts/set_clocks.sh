#!/bin/bash

# 检查是否以 root 用户执行
if [ "$EUID" -ne 0 ]; then 
  echo "このスクリプトを実行するには root 権限が必要です。"
  exit 1
fi

# If /root/.jetsonclocks_conf.txt is not present, create it
if [ ! -f /root/.jetsonclocks_conf.txt ]; then
    echo "Creating /root/.jetsonclocks_conf.txt..."
    sudo  /usr/bin/jetson_clocks --store

    if [ $? -ne 0 ]; then
        echo "エラーが発生しました。"
        exit 1
    else
        echo "設定ファイルを作成しました。"
    fi
fi

# 检查当前的 nvpmodel 模式
MODE=$(sudo nvpmodel -q | grep -oP '(?<=Power Mode: ).*')

if [[ "$MODE" == "MAXN" ]]; then
    echo "全性能モードです。クロックを最大に設定します..."
    sudo jetson_clocks  # 最大频率
elif [[ "$MODE" == "MODE_10W" ]]; then
    echo "省電力モードです。クロックをリセットします..."
    sudo jetson_clocks --restore  # 恢复默认频率
else
    echo "不明なモードです。手動で確認してください。"
    exit 1
fi

echo "クロックの設定が完了しました。"
