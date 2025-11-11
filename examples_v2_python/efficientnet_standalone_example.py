#!/usr/bin/env python3
"""
EfficientNet Standalone Example (V2 API)

This example demonstrates how to use the standalone EfficientNet pipeline
for image classification and feature extraction.

Requirements:
- TensorRT engine file for EfficientNet
- Input image(s)
- Built libtrtengine_v2.so library
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyengine.inference.c_pipeline import EfficientNetPipelineV2


def main():
    if len(sys.argv) < 4:
        print("Usage: python efficientnet_standalone_example.py <library.so> <engine.trt> <image.jpg>")
        print("\nExample:")
        print("  python efficientnet_standalone_example.py \\")
        print("    build/libtrtengine_v2.so \\")
        print("    efficientnet_b0.engine \\")
        print("    test_image.jpg")
        sys.exit(1)

    library_path = sys.argv[1]
    engine_path = sys.argv[2]
    image_path = sys.argv[3]

    print("=" * 70)
    print("EfficientNet Standalone Example (V2 API)")
    print("=" * 70)
    print(f"Library:  {library_path}")
    print(f"Engine:   {engine_path}")
    print(f"Image:    {image_path}")
    print("=" * 70)

    # Load image
    print("\n[1/3] Loading image...")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ERROR: Failed to load image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"  Image shape: {image_rgb.shape}")

    # Create pipeline
    print("\n[2/3] Creating EfficientNet pipeline...")
    pipeline = EfficientNetPipelineV2(
        library_path=library_path,
        engine_path=engine_path,
        input_width=224,
        input_height=224,
        max_batch_size=1,
        num_classes=2,
        feature_size=512,
        mean=[0.485, 0.456, 0.406],
        stddev=[0.229, 0.224, 0.225]
    )

    # Initialize pipeline
    print("  Creating pipeline context...")
    pipeline.create()
    print("  Pipeline created successfully!")

    # Run inference
    print("\n[3/3] Running inference...")
    results = pipeline.infer([image_rgb])

    if not results:
        print("  No results returned")
        pipeline.close()
        sys.exit(0)

    # Display results
    print("\nResults:")
    for result in results:
        img_idx = result['image_idx']
        class_id = result['class_id']
        confidence = result['confidence']
        logits = result['logits']
        features = result['features']

        print(f"\n  Image {img_idx}:")
        print(f"    Predicted Class: {class_id}")
        print(f"    Confidence: {confidence:.4f}")
        print(f"    Logits: {logits}")
        print(f"    Feature vector shape: {features.shape}")
        print(f"    Feature L2 norm: {np.linalg.norm(features):.4f}")
        print(f"    Feature preview (first 10): {features[:10]}")

    # Cleanup
    print("\nCleaning up...")
    pipeline.close()
    print("Done!")


if __name__ == "__main__":
    main()
