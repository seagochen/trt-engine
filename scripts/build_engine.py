import json
import subprocess
import os
import sys
import time

# ANSI 颜色代码
# WARNING_COLOR = '\033[93m' # Yellow
WARNING_COLOR = '\033[33m' # Less bright yellow
RESET_COLOR = '\033[0m'    # Reset to default color

def build_tensorrt_engine(config_file):
    """
    Reads a configuration file, constructs a trtexec command,
    and builds a TensorRT engine. It attempts with NVDLA first if enabled,
    and falls back to GPU-only if NVDLA compilation fails.
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"{WARNING_COLOR}Error: Configuration file '{config_file}' not found.{RESET_COLOR}")
        return
    except json.JSONDecodeError:
        print(f"{WARNING_COLOR}Error: Could not decode JSON from '{config_file}'. Please check the file's syntax.{RESET_COLOR}")
        return

    # Extract paths
    onnx_path = config.get("onnx_path")
    target_path = config.get("target_path")

    if not onnx_path or not target_path:
        print(f"{WARNING_COLOR}Error: 'onnx_path' and 'target_path' must be specified in the config.{RESET_COLOR}")
        return

    # Extract trt_config
    trt_config = config.get("trt_config", {})

    # Build base command parts
    base_command = ["trtexec"]
    base_command.append(f"--onnx={onnx_path}")
    base_command.append(f"--saveEngine={target_path}")

    # Add precision
    precision = trt_config.get("precision", "fp32")
    if precision == "fp16":
        base_command.append("--fp16")
    elif precision == "int8":
        base_command.append("--int8")
        # For INT8, you'd typically need calibration, which is more complex.
        # This example just adds --int8, but real-world usage would need --calib or other int8 flags.
        print(f"{WARNING_COLOR}Warning: INT8 precision selected. Full INT8 calibration setup is not included in this basic script.{RESET_COLOR}")

    # Add dynamic shapes
    min_shapes = trt_config.get("min_shapes")
    opt_shapes = trt_config.get("opt_shapes")
    max_shapes = trt_config.get("max_shapes")

    if min_shapes and opt_shapes and max_shapes:
        base_command.append(f"--minShapes={min_shapes}")
        base_command.append(f"--optShapes={opt_shapes}")
        base_command.append(f"--maxShapes={max_shapes}")
    else:
        print(f"{WARNING_COLOR}Warning: Dynamic shapes (min_shapes, opt_shapes, max_shapes) are recommended for flexible engines.{RESET_COLOR}")

    # Add verbose flag
    verbose = trt_config.get("verbose", False)
    if verbose:
        base_command.append("--verbose")

    # --- NVDLA Attempt Logic ---
    use_nvdla = trt_config.get("use_nvdla", False)
    nvdla_core_id = trt_config.get("nvdla_core_id") # Optional: allow specifying which core

    attempt_with_nvdla = False
    if use_nvdla:
        command_with_nvdla = list(base_command) # Create a copy of the base command
        if nvdla_core_id is not None:
            command_with_nvdla.append(f"--useDLACore={nvdla_core_id}")
        else:
            command_with_nvdla.append("--useDLACore=0") # Default to core 0 if not specified
        command_with_nvdla.append("--allowGPUFallback")
        attempt_with_nvdla = True

        print("\n" + "="*50)
        print("TensorRT Engine Building Initiated (Attempting with NVDLA)")
        print("="*50)
        print(f"Configuration file: {config_file}")
        print(f"Input ONNX: {onnx_path}")
        print(f"Output Engine: {target_path}")
        print(f"Precision: {precision}")
        print(f"NVDLA Enabled: Yes (Core {nvdla_core_id if nvdla_core_id is not None else 0}) with GPU fallback.")
        print(f"Generated Command: {' '.join(command_with_nvdla)}")
        print("="*50 + "\n")

        try:
            # Execute command with NVDLA
            process = subprocess.run(command_with_nvdla, check=True, text=True, capture_output=True)
            print("STDOUT:\n", process.stdout)
            if process.stderr:
                print("STDERR:\n", process.stderr)
            print("\n" + "="*50)
            print("TensorRT Engine Built Successfully with NVDLA!")
            print("="*50)
            return # Successfully built with NVDLA, exit function

        except subprocess.CalledProcessError as e:
            if "Cannot create DLA engine" in e.stderr or "DLA not available" in e.stderr:
                print(f"\n{WARNING_COLOR}⚠️ WARNING: NVDLA compilation failed. Reason: {e.stderr.strip().splitlines()[-1]}{RESET_COLOR}")
                print(f"{WARNING_COLOR}⚠️ Attempting to build engine without NVDLA acceleration...{RESET_COLOR}")
                time.sleep(1) # Small delay for readability
            else:
                # Other non-DLA related errors, re-raise or handle as critical
                print(f"{WARNING_COLOR}Error: trtexec command failed with unexpected exit code {e.returncode}{RESET_COLOR}")
                print("STDOUT:\n", e.stdout)
                print("STDERR:\n", e.stderr)
                return
        except FileNotFoundError:
            print(f"{WARNING_COLOR}Error: 'trtexec' command not found. Please ensure TensorRT is installed and configured in your PATH.{RESET_COLOR}")
            return
        except Exception as e:
            print(f"{WARNING_COLOR}An unexpected error occurred during NVDLA compilation attempt: {e}{RESET_COLOR}")
            return


    # --- GPU-only Fallback / Default Compilation ---
    # This block will run if:
    # 1. 'use_nvdla' was False in the config.
    # 2. 'use_nvdla' was True, but the NVDLA compilation attempt failed.
    
    print("\n" + "="*50)
    print("TensorRT Engine Building Initiated (GPU Only)")
    print("="*50)
    print(f"Configuration file: {config_file}")
    print(f"Input ONNX: {onnx_path}")
    print(f"Output Engine: {target_path}")
    print(f"Precision: {precision}")
    print("NVDLA Enabled: No (or NVDLA compilation failed, falling back to GPU).")
    print(f"Generated Command: {' '.join(base_command)}") # Note: This is base_command without NVDLA flags
    print("="*50 + "\n")

    try:
        process = subprocess.run(base_command, check=True, text=True, capture_output=True)
        print("STDOUT:\n", process.stdout)
        if process.stderr:
            print("STDERR:\n", process.stderr)
        print("\n" + "="*50)
        print("TensorRT Engine Built Successfully (GPU Only)!")
        print("="*50)
    except FileNotFoundError:
        print(f"{WARNING_COLOR}Error: 'trtexec' command not found. Please ensure TensorRT is installed and configured in your PATH.{RESET_COLOR}")
    except subprocess.CalledProcessError as e:
        print(f"{WARNING_COLOR}Error: trtexec command failed with exit code {e.returncode}{RESET_COLOR}")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
    except Exception as e:
        print(f"{WARNING_COLOR}An unexpected error occurred during GPU-only compilation: {e}{RESET_COLOR}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python build_engine.py <config_file.json>")
        print("Example: python build_engine.py config1.json")
        sys.exit(1)

    config_file_name = sys.argv[1]

    # No change to example config content, it's just for demonstration
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
            "precision": "fp16",
            "min_shapes": "input:1x3x224x224",
            "opt_shapes": "input:8x3x224x224",
            "max_shapes": "input:32x3x224x224",
            "use_nvdla": True,
            "nvdla_core_id": 0,
            "verbose": True
        }
    }

    if not os.path.exists(config_file_name):
        print(f"{WARNING_COLOR}Error: Configuration file '{config_file_name}' not found.{RESET_COLOR}")
        print("You can create a new config file based on this example:")
        print(json.dumps(config_example_content, indent=4))
        sys.exit(1)

    build_tensorrt_engine(config_file_name)