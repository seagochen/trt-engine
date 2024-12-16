#!/bin/bash

# Exit on error, undefined variable usage, and pipeline errors
set -euo pipefail

print_usage() {
    echo "Usage:"
    echo "  $0 <input_model.onnx> <output_model.engine> <input_tensor_name> <dynamic|static> [precision] <shape>"
    echo
    echo "Parameters:"
    echo "  input_model.onnx      Path to the input ONNX model."
    echo "  output_model.engine   Desired path of the output TensorRT engine file."
    echo "  input_tensor_name     The name of the input tensor in the ONNX model."
    echo "  dynamic|static        Specify whether to use dynamic shapes or static shapes."
    echo "  precision (optional)  The precision mode: fp32, fp16, int8, or best. Default: fp16"
    echo "  shape                 The input height and width in format HeightxWidth."
    echo
    echo "For static mode, one shape (HeightxWidth) is required."
    echo "For dynamic mode, also one shape (HeightxWidth) is required. The batch dimension will vary: min=1, opt=2, max=4."
    echo
    echo "Examples:"
    echo "  $0 model.onnx model.engine input_tensor dynamic fp16 224x224"
    echo "  $0 model.onnx model.engine input_tensor static 224x224"
}

# Check for minimum arguments
if [ $# -lt 4 ]; then
    echo "Error: Insufficient arguments."
    print_usage
    exit 1
fi

ONNX_MODEL="$1"
OUTPUT_MODEL="$2"
TENSOR_NAME="$3"
MODE="$4"

if [ "$MODE" != "dynamic" ] && [ "$MODE" != "static" ]; then
    echo "Error: The fourth argument must be either 'dynamic' or 'static'."
    print_usage
    exit 1
fi

# Shift past the first four arguments
shift 4

# Precision default
PRECISION="fp16"

# Check if the next argument is a known precision
if [ $# -gt 0 ]; then
    case "$1" in
        fp32|fp16|int8|best)
            PRECISION="$1"
            shift
            ;;
        *)
            # If it's not a known precision, we assume the default fp16 and treat this arg as a shape
            ;;
    esac
fi

# For both dynamic and static modes, we now expect one shape argument
if [ $# -lt 1 ]; then
    echo "Error: Please provide a shape argument (e.g., 224x224)."
    print_usage
    exit 1
fi

SHAPE="$1"
if ! [[ "$SHAPE" =~ ^[0-9]+x[0-9]+$ ]]; then
    echo "Error: Invalid shape format '$SHAPE'. Expected HeightxWidth (e.g., 224x224)."
    exit 1
fi

IFS='x' read -r H W <<< "$SHAPE"

# Determine path to trtexec
if [ -f "/opt/tensorrt/bin/trtexec" ]; then
    TRTEXEC="/opt/tensorrt/bin/trtexec"
else
    TRTEXEC="/usr/src/tensorrt/bin/trtexec"
fi

# Determine precision flag
case "$PRECISION" in
    fp32) PRECISION_FLAG="" ;;
    fp16) PRECISION_FLAG="--fp16" ;;
    int8) PRECISION_FLAG="--int8" ;;
    best) PRECISION_FLAG="--best" ;;
esac

# Remove existing engine file if exists
if [ -f "$OUTPUT_MODEL" ]; then
    rm "$OUTPUT_MODEL"
fi

echo "Converting the ONNX model to TensorRT engine..."

if [ "$MODE" = "dynamic" ]; then
    # Dynamic mode: fix height and width, vary batch dimension
    MIN_SHAPES="${TENSOR_NAME}:1x3x${H}x${W}"
    OPT_SHAPES="${TENSOR_NAME}:2x3x${H}x${W}"
    MAX_SHAPES="${TENSOR_NAME}:4x3x${H}x${W}"

    "$TRTEXEC" --onnx="$ONNX_MODEL" --saveEngine="$OUTPUT_MODEL" $PRECISION_FLAG \
                --minShapes="$MIN_SHAPES" --optShapes="$OPT_SHAPES" --maxShapes="$MAX_SHAPES"
else
    # Static mode
    SHAPE_ARG="--shapes=${TENSOR_NAME}:1x3x${H}x${W}"

    "$TRTEXEC" --onnx="$ONNX_MODEL" --saveEngine="$OUTPUT_MODEL" $PRECISION_FLAG \
                $SHAPE_ARG
fi

if [ $? -ne 0 ]; then
    echo "Failed to convert the source model to the target engine."
    exit 1
fi

echo "Conversion successful. The engine file is saved at: $OUTPUT_MODEL"
