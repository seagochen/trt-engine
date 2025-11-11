#!/usr/bin/env python3
"""
YOLOv8-Pose Standalone Example (V2 API)

This example demonstrates how to use the standalone YOLOv8-Pose pipeline
without coupling to EfficientNet or other models.

Requirements:
- TensorRT engine file for YOLOv8-Pose
- Input image(s)
- Built libtrtengine_v2.so library
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyengine.inference.c_pipeline import YoloPosePipelineV2


def draw_pose_on_image(image: np.ndarray, detections: list) -> np.ndarray:
    """
    Draw bounding boxes and keypoints on image

    Args:
        image: Input image (RGB)
        detections: List of pose detections

    Returns:
        Image with visualizations (RGB)
    """
    result_img = image.copy()

    # COCO skeleton connections
    skeleton = [
        (0, 1), (0, 2),   # nose to eyes
        (1, 3), (2, 4),   # eyes to ears
        (0, 5), (0, 6),   # nose to shoulders
        (5, 7), (7, 9),   # left arm
        (6, 8), (8, 10),  # right arm
        (5, 11), (6, 12), # shoulders to hips
        (11, 13), (13, 15), # left leg
        (12, 14), (14, 16)  # right leg
    ]

    for det in detections:
        # Draw bounding box
        bbox = det['bbox']
        cv2.rectangle(result_img,
                     (bbox[0], bbox[1]),
                     (bbox[2], bbox[3]),
                     (0, 255, 0), 2)

        # Draw confidence
        conf_text = f"{det['conf']:.2f}"
        cv2.putText(result_img, conf_text,
                   (bbox[0], bbox[1] - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   (0, 255, 0), 2)

        # Draw keypoints
        keypoints = det['keypoints']
        for kpt in keypoints:
            if kpt['conf'] > 0.5:
                x, y = int(kpt['x']), int(kpt['y'])
                cv2.circle(result_img, (x, y), 3, (255, 0, 0), -1)

        # Draw skeleton
        for conn in skeleton:
            kpt1 = keypoints[conn[0]]
            kpt2 = keypoints[conn[1]]
            if kpt1['conf'] > 0.5 and kpt2['conf'] > 0.5:
                x1, y1 = int(kpt1['x']), int(kpt1['y'])
                x2, y2 = int(kpt2['x']), int(kpt2['y'])
                cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return result_img


def main():
    if len(sys.argv) < 4:
        print("Usage: python yolopose_standalone_example.py <library.so> <engine.trt> <image.jpg>")
        print("\nExample:")
        print("  python yolopose_standalone_example.py \\")
        print("    build/libtrtengine_v2.so \\")
        print("    yolov8n-pose.engine \\")
        print("    test_image.jpg")
        sys.exit(1)

    library_path = sys.argv[1]
    engine_path = sys.argv[2]
    image_path = sys.argv[3]

    print("=" * 70)
    print("YOLOv8-Pose Standalone Example (V2 API)")
    print("=" * 70)
    print(f"Library:  {library_path}")
    print(f"Engine:   {engine_path}")
    print(f"Image:    {image_path}")
    print("=" * 70)

    # Load image
    print("\n[1/4] Loading image...")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ERROR: Failed to load image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"  Image shape: {image_rgb.shape}")

    # Create pipeline
    print("\n[2/4] Creating YOLOv8-Pose pipeline...")
    pipeline = YoloPosePipelineV2(
        library_path=library_path,
        engine_path=engine_path,
        input_width=640,
        input_height=640,
        max_batch_size=1,
        conf_threshold=0.25,
        iou_threshold=0.45
    )

    # Initialize pipeline
    print("  Creating pipeline context...")
    pipeline.create()
    print("  Pipeline created successfully!")

    # Run inference
    print("\n[3/4] Running inference...")
    results = pipeline.infer([image_rgb])

    if not results:
        print("  No results returned")
        pipeline.close()
        sys.exit(0)

    # Display results
    print("\n[4/4] Results:")
    for result in results:
        img_idx = result['image_idx']
        detections = result['detections']
        print(f"\n  Image {img_idx}: {len(detections)} person(s) detected")

        for i, det in enumerate(detections):
            bbox = det['bbox']
            conf = det['conf']
            print(f"    Person {i}:")
            print(f"      BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")
            print(f"      Confidence: {conf:.4f}")

            # Count visible keypoints
            visible_kpts = sum(1 for kpt in det['keypoints'] if kpt['conf'] > 0.5)
            print(f"      Visible keypoints: {visible_kpts}/17")

    # Visualize
    print("\n[5/5] Saving visualization...")
    if results and results[0]['detections']:
        vis_image_rgb = draw_pose_on_image(image_rgb, results[0]['detections'])
        vis_image_bgr = cv2.cvtColor(vis_image_rgb, cv2.COLOR_RGB2BGR)
        output_path = "output_yolopose_standalone.jpg"
        cv2.imwrite(output_path, vis_image_bgr)
        print(f"  Saved visualization to: {output_path}")
    else:
        print("  No detections to visualize")

    # Cleanup
    print("\nCleaning up...")
    pipeline.close()
    print("Done!")


if __name__ == "__main__":
    main()
