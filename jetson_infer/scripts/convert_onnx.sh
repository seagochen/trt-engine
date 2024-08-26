#!/bin/bash

# 检查是否有任何参数传递
if [ $# -eq 0 ]; then
    echo "No arguments provided."
    echo "Usage: sh ./convert_model.sh <input_model.onnx> <output_model.engine> [precision]"
    echo "Precision options: fp32, fp16, int8, best"
    exit 1
fi

# 获取输入参数
ONNX_MODEL=$1
OUTPUT_MODEL=$2
PRECISION=$3

# 检查trtexec路径
if [ -f ./tensorrt/bin/trtexec ]; then
    TRTEXEC=./tensorrt/bin/trtexec
else
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi

# 检查精度参数
if [ -z "$PRECISION" ]; then
    PRECISION="fp16"  # 默认使用fp16
fi

# 将tensorrt/lib临时添加到LD_LIBRARY_PATH
if [ -d "$(pwd)/tensorrt/lib" ]; then
    export LD_LIBRARY_PATH=$(pwd)/tensorrt/lib:$LD_LIBRARY_PATH
fi

# 检查输入参数是否正确
if [ -z "$ONNX_MODEL" ] || [ -z "$OUTPUT_MODEL" ]; then
    echo "Usage: sh ./convert_model.sh <input_model.onnx> <output_model.engine> [precision]"
    echo "Precision options: fp32, fp16, int8, best"
    exit 1
fi

# 删除目标模型
if [ -f $OUTPUT_MODEL ]; then
    rm $OUTPUT_MODEL
fi

# 如果目标模型不存在，则将源模型转换为目标模型
if [ ! -f $OUTPUT_MODEL ]; then
    echo "The target model is not found. Converting the source model to the target model."
    case $PRECISION in
        fp32)
            $TRTEXEC --onnx=$ONNX_MODEL --saveEngine=$OUTPUT_MODEL
            ;;
        fp16)
            $TRTEXEC --onnx=$ONNX_MODEL --saveEngine=$OUTPUT_MODEL --fp16
            ;;
        int8)
            $TRTEXEC --onnx=$ONNX_MODEL --saveEngine=$OUTPUT_MODEL --int8
            ;;
        best)
            $TRTEXEC --onnx=$ONNX_MODEL --saveEngine=$OUTPUT_MODEL --best
            ;;
        *)
            echo "Invalid precision option. Use fp32, fp16, int8, or best."
            exit 1
            ;;
    esac

    if [ $? -ne 0 ]; then
        echo "Failed to convert the source model to the target model."
        exit 1
    fi
fi
