import os
import sys

def import_rawframe(pb2_dir: str):
    pb2_dir = os.path.abspath(pb2_dir)
    if pb2_dir not in sys.path:
        sys.path.insert(0, pb2_dir)
    import raw_frames_pb2
    return raw_frames_pb2.RawFrame

def import_inference_result(pb2_dir: str):
    pb2_dir = os.path.abspath(pb2_dir)
    if pb2_dir not in sys.path:
        sys.path.insert(0, pb2_dir)
    import inference_result_pb2
    return inference_result_pb2.InferenceResult