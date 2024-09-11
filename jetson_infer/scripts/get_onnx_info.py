import onnx
import sys  # 添加sys模块

def print_onnx_io_info(onnx_model_path):
    # 加载ONNX模型
    model = onnx.load(onnx_model_path)

    # 获取模型的图结构
    graph = model.graph

    # 输出输入tensor的信息
    print("Inputs:")
    for input_tensor in graph.input:
        name = input_tensor.name
        shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
        print(f"Name: {name}, Shape: {shape}")

    # 输出输出tensor的信息
    print("\nOutputs:")
    for output_tensor in graph.output:
        name = output_tensor.name
        shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
        print(f"Name: {name}, Shape: {shape}")

# 检查是否提供了ONNX模型路径
if len(sys.argv) < 2:
    print("Usage: python get_onnx_info.py <onnx_model_path>")
    sys.exit(1)

# 调用函数并传入ONNX模型的路径
onnx_model_path = sys.argv[1]
print_onnx_io_info(onnx_model_path)
