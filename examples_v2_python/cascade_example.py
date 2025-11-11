#!/usr/bin/env python3
"""
Cascade Pipeline Example (V2 API)

This example demonstrates how to use both YOLOv8-Pose and EfficientNet
pipelines together in a cascade:
1. YOLOv8-Pose detects people and keypoints
2. Crop detected regions
3. EfficientNet classifies/extracts features from crops

Requirements:
- TensorRT engine files for both models
- Input image
- Built libtrtengine_v2.so library
"""
import sys
import cv2
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from pyengine.inference.c_pipeline import YoloPosePipelineV2, EfficientNetPipelineV2


def crop_bbox_with_margin(image: np.ndarray, bbox: list, margin: float = 0.1) -> np.ndarray:
    """
    Crop bounding box with margin

    Args:
        image: Input image (RGB)
        bbox: [lx, ly, rx, ry]
        margin: Margin ratio (default 10%)

    Returns:
        Cropped image (RGB)
    """
    h, w = image.shape[:2]
    lx, ly, rx, ry = bbox

    # Add margin
    bbox_w = rx - lx
    bbox_h = ry - ly
    margin_x = int(bbox_w * margin)
    margin_y = int(bbox_h * margin)

    # Expand with margin
    lx = max(0, lx - margin_x)
    ly = max(0, ly - margin_y)
    rx = min(w, rx + margin_x)
    ry = min(h, ry + margin_y)

    return image[ly:ry, lx:rx]


def draw_results(image: np.ndarray, detections: list, eff_results: dict) -> np.ndarray:
    """
    Draw detection results on image

    Args:
        image: Input image (RGB)
        detections: YOLOv8-Pose detections
        eff_results: EfficientNet results (indexed by detection)

    Returns:
        Visualization image (RGB)
    """
    result_img = image.copy()

    for idx, det in enumerate(detections):
        bbox = det['bbox']
        conf = det['conf']

        # Draw bounding box
        color = (0, 255, 0) if idx in eff_results else (255, 0, 0)
        cv2.rectangle(result_img,
                     (bbox[0], bbox[1]),
                     (bbox[2], bbox[3]),
                     color, 2)

        # Draw detection info
        info_text = f"Person {idx} ({conf:.2f})"
        cv2.putText(result_img, info_text,
                   (bbox[0], bbox[1] - 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                   color, 2)

        # Draw classification result if available
        if idx in eff_results:
            eff_res = eff_results[idx]
            class_text = f"Class: {eff_res['class_id']} ({eff_res['confidence']:.2f})"
            cv2.putText(result_img, class_text,
                       (bbox[0], bbox[1] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                       color, 1)

        # Draw keypoints
        keypoints = det['keypoints']
        for kpt in keypoints:
            if kpt['conf'] > 0.5:
                x, y = int(kpt['x']), int(kpt['y'])
                cv2.circle(result_img, (x, y), 2, (255, 0, 0), -1)

    return result_img


def main():
    if len(sys.argv) < 5:
        print("Usage: python cascade_example.py <library.so> <yolo.trt> <eff.trt> <image.jpg>")
        print("\nExample:")
        print("  python cascade_example.py \\")
        print("    build/libtrtengine_v2.so \\")
        print("    yolov8n-pose.engine \\")
        print("    efficientnet_b0.engine \\")
        print("    test_image.jpg")
        sys.exit(1)

    library_path = sys.argv[1]
    yolo_engine_path = sys.argv[2]
    eff_engine_path = sys.argv[3]
    image_path = sys.argv[4]

    print("=" * 70)
    print("Cascade Pipeline Example (V2 API)")
    print("=" * 70)
    print(f"Library:            {library_path}")
    print(f"YOLOv8-Pose Engine: {yolo_engine_path}")
    print(f"EfficientNet Engine: {eff_engine_path}")
    print(f"Image:              {image_path}")
    print("=" * 70)

    # Load image
    print("\n[1/5] Loading image...")
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"ERROR: Failed to load image: {image_path}")
        sys.exit(1)

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    print(f"  Image shape: {image_rgb.shape}")

    # Create YOLOv8-Pose pipeline
    print("\n[2/5] Creating YOLOv8-Pose pipeline...")
    yolo_pipeline = YoloPosePipelineV2(
        library_path=library_path,
        engine_path=yolo_engine_path,
        input_width=640,
        input_height=640,
        max_batch_size=1,
        conf_threshold=0.25,
        iou_threshold=0.45
    )
    yolo_pipeline.create()
    print("  YOLOv8-Pose pipeline ready")

    # Create EfficientNet pipeline
    print("\n[3/5] Creating EfficientNet pipeline...")
    eff_pipeline = EfficientNetPipelineV2(
        library_path=library_path,
        engine_path=eff_engine_path,
        input_width=224,
        input_height=224,
        max_batch_size=1,
        num_classes=2,
        feature_size=512
    )
    eff_pipeline.create()
    print("  EfficientNet pipeline ready")

    # Step 1: YOLOv8-Pose detection
    print("\n[4/5] Running cascade inference...")
    print("  Step 1: YOLOv8-Pose detection...")
    yolo_results = yolo_pipeline.infer([image_rgb])

    if not yolo_results or not yolo_results[0]['detections']:
        print("    No persons detected!")
        yolo_pipeline.close()
        eff_pipeline.close()
        sys.exit(0)

    detections = yolo_results[0]['detections']
    print(f"    Detected {len(detections)} person(s)")

    # Step 2: Crop and classify
    print("  Step 2: Cropping and classifying detected regions...")
    eff_results = {}

    for idx, det in enumerate(detections):
        bbox = det['bbox']
        print(f"    Processing person {idx}...")

        # Crop region
        try:
            cropped = crop_bbox_with_margin(image_rgb, bbox, margin=0.1)
            if cropped.size == 0:
                print(f"      Warning: Empty crop, skipping")
                continue

            # Run EfficientNet
            results = eff_pipeline.infer([cropped])
            if results:
                eff_results[idx] = results[0]
                print(f"      Class: {results[0]['class_id']}, "
                      f"Conf: {results[0]['confidence']:.4f}")
        except Exception as e:
            print(f"      Error processing crop: {e}")
            continue

    # Display results
    print("\n[5/5] Results Summary:")
    print(f"  Total persons detected: {len(detections)}")
    print(f"  Successfully classified: {len(eff_results)}")

    for idx, det in enumerate(detections):
        bbox = det['bbox']
        conf = det['conf']
        visible_kpts = sum(1 for kpt in det['keypoints'] if kpt['conf'] > 0.5)

        print(f"\n  Person {idx}:")
        print(f"    Detection confidence: {conf:.4f}")
        print(f"    Visible keypoints: {visible_kpts}/17")
        print(f"    BBox: [{bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]}]")

        if idx in eff_results:
            eff_res = eff_results[idx]
            print(f"    Classification:")
            print(f"      Class ID: {eff_res['class_id']}")
            print(f"      Confidence: {eff_res['confidence']:.4f}")
            print(f"      Feature L2 norm: {np.linalg.norm(eff_res['features']):.4f}")

    # Visualize
    print("\nSaving visualization...")
    vis_image = draw_results(image_rgb, detections, eff_results)
    vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
    output_path = "output_cascade.jpg"
    cv2.imwrite(output_path, vis_image_bgr)
    print(f"  Saved to: {output_path}")

    # Cleanup
    print("\nCleaning up...")
    yolo_pipeline.close()
    eff_pipeline.close()
    print("Done!")


if __name__ == "__main__":
    main()
