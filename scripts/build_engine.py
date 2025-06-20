import json
import subprocess
import os
import sys # 导入sys模块用于处理命令行参数

def build_tensorrt_engine(config_file):
    """
    Reads a configuration file, constructs a trtexec command,
    and builds a TensorRT engine.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{config_file}'. Please check the file's syntax.")
        return

    # Extract paths
    onnx_path = config.get("onnx_path")
    target_path = config.get("target_path")

    if not onnx_path or not target_path:
        print("Error: 'onnx_path' and 'target_path' must be specified in the config.")
        return

    # Extract trt_config
    trt_config = config.get("trt_config", {})

    # Start building the command
    command = ["trtexec"]
    command.append(f"--onnx={onnx_path}")
    command.append(f"--saveEngine={target_path}")

    # Add precision
    precision = trt_config.get("precision", "fp32")
    if precision == "fp16":
        command.append("--fp16")
    elif precision == "int8":
        command.append("--int8")
        # For INT8, you'd typically need calibration, which is more complex.
        # This example just adds --int8, but real-world usage would need --calib or other int8 flags.
        print("Warning: INT8 precision selected. Full INT8 calibration setup is not included in this basic script.")
    # No explicit flag needed for fp32 as it's often the default or handled implicitly.

    # Add dynamic shapes
    min_shapes = trt_config.get("min_shapes")
    opt_shapes = trt_config.get("opt_shapes")
    max_shapes = trt_config.get("max_shapes")

    if min_shapes and opt_shapes and max_shapes:
        command.append(f"--minShapes={min_shapes}")
        command.append(f"--optShapes={opt_shapes}")
        command.append(f"--maxShapes={max_shapes}")
    else:
        print("Warning: Dynamic shapes (min_shapes, opt_shapes, max_shapes) are recommended for flexible engines.")

    # Add verbose flag
    verbose = trt_config.get("verbose", False)
    if verbose:
        command.append("--verbose")

    # --- Execute the command ---
    print("\n" + "="*50)
    print("TensorRT Engine Building Initiated")
    print("="*50)
    print(f"Configuration file: {config_file}")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output Engine: {target_path}")
    print(f"Precision: {precision}")
    print(f"Generated Command: {' '.join(command)}")
    print("="*50 + "\n")

    try:
        # Use subprocess.run for a cleaner way to execute external commands
        # check=True will raise an exception if the command returns a non-zero exit code
        # text=True decodes stdout/stderr as text
        process = subprocess.run(command, check=True, text=True, capture_output=True)
        print("STDOUT:\n", process.stdout)
        if process.stderr:
            print("STDERR:\n", process.stderr)
        print("\n" + "="*50)
        print("TensorRT Engine Built Successfully!")
        print("="*50)
    except FileNotFoundError:
        print(f"Error: 'trtexec' command not found. Please ensure TensorRT is installed and configured in your PATH.")
    except subprocess.CalledProcessError as e:
        print(f"Error: trtexec command failed with exit code {e.returncode}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Check for command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python build_engine.py <config_file.json>")
        print("Example: python build_engine.py config1.json")
        # Optionally, create a dummy config if no argument is provided, but it's better to enforce usage
        # config_file_name = "config.json"
        # if not os.path.exists(config_file_name):
        #     print(f"Creating a dummy '{config_file_name}' for demonstration...")
        #     with open(config_file_name, 'w') as f:
        #         json.dump(config_example_content, f, indent=4)
        #     print("Dummy file created. Please ensure ONNX model paths are correct before running.")
        # build_tensorrt_engine(config_file_name)
        sys.exit(1) # Exit if no config file is provided

    config_file_name = sys.argv[1] # Get the config file name from the first argument

    # Example dummy config content (only created if no config file exists for initial setup)
    # You should create your actual config files manually or as part of your project setup.
    config_example_content = {
        "onnx_path": "/opt/models/efficientnet_b0_feat_logits.onnx",
        "target_path": "/opt/models/efficientnet_b0_feat_logits.engine",
        "model_information": {
            "name": "EfficientNet B0 Feature and Logits",
            "description": "EfficientNet B0 model for feature extraction and logits output",
            "version": "1.0.0",
            "author": "Orlando Chen",
            "date": "2023-10-01",
            "input_tensors": [
                {"name": "input", "dimensions": [1, 3, 224, 224]}
            ],
            "output_tensors": [
                {"name": "logits", "dimensions": [1, 2]},
                {"name": "feat", "dimensions": [1, 256]}
            ]
        },
        "trt_config": {
            "precision": "fp16", # You can change this to "fp32" or "int8"
            "min_shapes": "input:1x3x224x224",
            "opt_shapes": "input:8x3x224x224",
            "max_shapes": "input:32x3x224x224",
            "verbose": True
        }
    }

    # If the specified config file doesn't exist, provide a hint to create one.
    if not os.path.exists(config_file_name):
        print(f"Error: Configuration file '{config_file_name}' not found.")
        print("You can create a new config file based on this example:")
        print(json.dumps(config_example_content, indent=4))
        sys.exit(1)

    # Run the engine building process with the specified config file
    build_tensorrt_engine(config_file_name)