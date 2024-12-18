import onnx
import sys

def extract_onnx_model_info(model_path):
    """
    Extracts input and output names and dimensions from an ONNX model.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        dict: A dictionary containing inputs and outputs with their names and dimensions.
    """
    # Load the ONNX model
    model = onnx.load(model_path)
    
    # Check the model to ensure it is valid
    onnx.checker.check_model(model)
    
    # Get the model's graph
    graph = model.graph

    # Extract inputs
    inputs = [
        {"name": input.name, 
         "dims": [dim.dim_value if dim.HasField('dim_value') else "dynamic" for dim in input.type.tensor_type.shape.dim]}
        for input in graph.input
    ]

    # Extract outputs
    outputs = [
        {"name": output.name, 
         "dims": [dim.dim_value if dim.HasField('dim_value') else "dynamic" for dim in output.type.tensor_type.shape.dim]}
        for output in graph.output
    ]

    return {"inputs": inputs, "outputs": outputs}

if __name__ == "__main__":
    # Check if the model path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_onnx_model>")
        sys.exit(1)

    # Get the model path from the command-line arguments
    model_path = sys.argv[1]

    # Extract and display model information
    model_info = extract_onnx_model_info(model_path)
    
    print("Inputs:")
    for input in model_info["inputs"]:
        print(f"  Name: {input['name']}, Dims: {input['dims']}")

    print("\nOutputs:")
    for output in model_info["outputs"]:
        print(f"  Name: {output['name']}, Dims: {output['dims']}")
