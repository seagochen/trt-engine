import json
import subprocess
import os
import sys
import time

# ANSI 颜色代码
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

    def run_trtexec_command(command, attempt_name=""):
        """Helper function to run trtexec and print output in real-time."""
        print(f"\n{'='*50}")
        print(f"TensorRT Engine Building Initiated ({attempt_name})")
        print(f"{'='*50}")
        print(f"Configuration file: {config_file}")
        print(f"Input ONNX: {onnx_path}")
        print(f"Output Engine: {target_path}")
        print(f"Precision: {precision}")
        print(f"Generated Command: {' '.join(command)}")
        print(f"{'='*50}\n")

        process = None
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            
            # Read output in real-time
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip()) # Print stdout lines as they come

            # Read remaining stderr (if any, after stdout is done)
            stderr_output = process.stderr.read()
            if stderr_output:
                print(f"{WARNING_COLOR}STDERR:\n{stderr_output.strip()}{RESET_COLOR}")

            process.wait() # Wait for the process to truly finish

            if process.returncode != 0:
                # If there's an error and it's NVDLA related, we'll handle it outside
                # Otherwise, it's a general trtexec error
                return False, stderr_output # Return False and stderr for error handling
            else:
                return True, "" # Return True for success
        except FileNotFoundError:
            print(f"{WARNING_COLOR}Error: 'trtexec' command not found. Please ensure TensorRT is installed and configured in your PATH.{RESET_COLOR}")
            return False, ""
        except Exception as e:
            print(f"{WARNING_COLOR}An unexpected error occurred: {e}{RESET_COLOR}")
            return False, ""
        finally:
            if process and process.poll() is None: # If process is still running, terminate it
                process.terminate()

    # --- NVDLA Attempt Logic ---
    use_nvdla = trt_config.get("use_nvdla", False)
    nvdla_core_id = trt_config.get("nvdla_core_id")

    if use_nvdla:
        command_with_nvdla = list(base_command)
        if nvdla_core_id is not None:
            command_with_nvdla.append(f"--useDLACore={nvdla_core_id}")
        else:
            command_with_nvdla.append("--useDLACore=0")
        command_with_nvdla.append("--allowGPUFallback")

        print(f"NVDLA Enabled: Yes (Core {nvdla_core_id if nvdla_core_id is not None else 0}) with GPU fallback.")
        
        success, stderr = run_trtexec_command(command_with_nvdla, "Attempting with NVDLA")

        if success:
            print("\n" + "="*50)
            print("TensorRT Engine Built Successfully with NVDLA!")
            print("="*50)
            return
        elif "Cannot create DLA engine" in stderr or "DLA not available" in stderr:
            print(f"\n{WARNING_COLOR}⚠️ WARNING: NVDLA compilation failed. Reason: {stderr.strip().splitlines()[-1]}{RESET_COLOR}")
            print(f"{WARNING_COLOR}⚠️ Attempting to build engine without NVDLA acceleration...{RESET_COLOR}")
            time.sleep(1)
        else:
            print(f"{WARNING_COLOR}Error: trtexec command failed during NVDLA attempt.{RESET_COLOR}")
            print(f"STDERR:\n{stderr}")
            return

    # --- GPU-only Fallback / Default Compilation ---
    print("NVDLA Enabled: No (or NVDLA compilation failed, falling back to GPU).")
    success, stderr = run_trtexec_command(base_command, "GPU Only")

    if success:
        print("\n" + "="*50)
        print("TensorRT Engine Built Successfully (GPU Only)!")
        print("="*50)
    else:
        print(f"{WARNING_COLOR}Error: trtexec command failed during GPU-only compilation.{RESET_COLOR}")
        print(f"STDERR:\n{stderr}")

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