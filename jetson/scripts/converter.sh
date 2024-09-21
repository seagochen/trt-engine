#!/bin/bash

# 检查是否有任何参数传递
if [ $# -lt 4 ]; then
    echo "No sufficient arguments provided."
    echo "Usage: sh ./convert_model.sh <input_model.onnx> <output_model.engine> <tensor_name> <dynamic> [precision]"
    echo "Precision options: fp32, fp16, int8, best"
    echo "Dynamic options: dynamic, static"
    exit 1
fi

# 获取输入参数
ONNX_MODEL=$1
OUTPUT_MODEL=$2
TENSOR_NAME=$3
DYNAMIC=$4
PRECISION=$5

# 检查trtexec路径
if [ -f /opt/tensorrt/bin/trtexec ]; then
    TRTEXEC=/opt/tensorrt/bin/trtexec
else
    TRTEXEC=/usr/src/tensorrt/bin/trtexec
fi

# 检查精度参数
if [ -z "$PRECISION" ]; then
    PRECISION="fp16"  # 默认使用fp16
fi

# 检查输入参数是否正确
if [ -z "$ONNX_MODEL" ] || [ -z "$OUTPUT_MODEL" ] || [ -z "$TENSOR_NAME" ] || [ -z "$DYNAMIC" ]; then
    echo "Usage: sh ./convert_model.sh <input_model.onnx> <output_model.engine> <tensor_name> <dynamic> [precision]"
    echo "Precision options: fp32, fp16, int8, best"
    echo "Dynamic options: dynamic, static"
    exit 1
fi

# 删除目标模型
if [ -f $OUTPUT_MODEL ]; then
    rm $OUTPUT_MODEL
fi

# 设置动态和静态输入的形状
if [ "$DYNAMIC" == "dynamic" ]; then
    MIN_SHAPES="$TENSOR_NAME:1x3x640x640"
    OPT_SHAPES="$TENSOR_NAME:2x3x640x640"
    MAX_SHAPES="$TENSOR_NAME:4x3x640x640"
fi

# 准备精度相关的参数
PRECISION_FLAG=""
case $PRECISION in
    fp32)
        PRECISION_FLAG=""
        ;;
    fp16)
        PRECISION_FLAG="--fp16"
        ;;
    int8)
        PRECISION_FLAG="--int8"
        ;;
    best)
        PRECISION_FLAG="--best"
        ;;
    *)
        echo "Invalid precision option. Use fp32, fp16, int8, or best."
        exit 1
        ;;
esac

# 如果目标模型不存在，则将源模型转换为目标模型
if [ ! -f $OUTPUT_MODEL ]; then
    echo "The target model is not found. Converting the source model to the target model."
    
    if [ "$DYNAMIC" == "dynamic" ]; then
        $TRTEXEC --onnx=$ONNX_MODEL --saveEngine=$OUTPUT_MODEL $PRECISION_FLAG \
                 --minShapes=$MIN_SHAPES --optShapes=$OPT_SHAPES --maxShapes=$MAX_SHAPES
    else
        $TRTEXEC --onnx=$ONNX_MODEL --saveEngine=$OUTPUT_MODEL $PRECISION_FLAG
    fi

    if [ $? -ne 0 ]; then
        echo "Failed to convert the source model to the target model."
        exit 1
    fi
fi
