#!/bin/bash

# Function to check if required tools are installed
check_requirements() {
    if ! command -v jq &> /dev/null; then
        echo "jq is not installed. Attempting to install..."
        if command -v apt-get &> /dev/null; then
            sudo apt-get update && sudo apt-get install -y jq
            if [ $? -ne 0 ]; then
                echo "Error: Failed to install jq. Please install it manually using: sudo apt-get install jq"
                exit 1
            fi
        else
            echo "Error: Unable to automatically install jq. Please install it manually."
            exit 1
        fi
    fi

    if ! command -v trtexec &> /dev/null; then
        echo "Error: trtexec is not installed or not in PATH"
        exit 1
    fi
}

# 0. 检查依赖
check_requirements

# 1. 检查参数
if [ -z "$1" ]; then
  echo "Usage: $0 <path-to-config.json>"
  exit 1
fi
CONFIG="$1"

# 2. 查找 Python 解释器
if command -v python3 &> /dev/null; then
  PY=python3
elif command -v python &> /dev/null; then
  PY=python
else
  echo "Error: 找不到 python3 或 python" >&2
  exit 1
fi

# 3. 调用 Python 脚本
#    注意：这里假设 trt_builder.py 和 launcher.sh 在同一目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
$PY "$SCRIPT_DIR/build_engine.py" "$CONFIG"
RET=$?

# 4. 返回状态
if [ $RET -eq 0 ]; then
  echo "🎉 All done!"
else
  echo "❌ build_engine.py 运行失败 (exit code $RET)" >&2
fi
exit $RET
