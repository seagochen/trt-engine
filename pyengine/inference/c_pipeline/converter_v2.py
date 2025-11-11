"""
V2 Architecture Data Converters

This module provides converters to transform C API output (dictionaries)
to Python unified data structures (Skeleton, ClassificationResult, etc.)

Key differences from V1:
- YoloPose and EfficientNet are decoupled
- Separate converters for each model
- No features in Skeleton (features only in ClassificationResult)
"""
from typing import List, Dict
import numpy as np

from pyengine.inference.unified_structs.inference_results import (
    Rect, Point, Skeleton, ClassificationResult
)


def yolopose_to_skeletons(raw_results: List[Dict]) -> List[List[Skeleton]]:
    """
    将 YoloPose V2 API 的原始输出转换为 Skeleton 对象列表

    这个函数用于独立的 YoloPose 推理，不包含特征向量。

    Args:
        raw_results: YoloPosePipelineV2.infer() 返回的原始结果
                    格式:
                    [
                        {
                            "image_idx": int,
                            "detections": [
                                {
                                    "bbox": [lx, ly, rx, ry],
                                    "cls": int,
                                    "conf": float,
                                    "keypoints": [
                                        {"x": float, "y": float, "conf": float},
                                        ...  # 17 keypoints
                                    ]
                                },
                                ...
                            ]
                        },
                        ...
                    ]

    Returns:
        List[List[Skeleton]]: 嵌套列表，外层对应每张图像，内层是该图像的 Skeleton 对象
    """
    all_skeletons_per_image: List[List[Skeleton]] = []

    for image_result in raw_results:
        current_image_skeletons: List[Skeleton] = []
        detections = image_result['detections']

        for det in detections:
            # 提取边界框
            bbox = det['bbox']
            rect = Rect(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3])
            )

            # 提取关键点
            points_list: List[Point] = []
            keypoints = det['keypoints']
            for kp in keypoints:
                points_list.append(
                    Point(
                        x=kp['x'],
                        y=kp['y'],
                        confidence=kp['conf']
                    )
                )

            # 创建 Skeleton 对象（不包含 features）
            skeleton = Skeleton(
                rect=rect,
                classification=det['cls'],
                confidence=det['conf'],
                track_id=0,  # 默认 0，tracking 需要单独处理
                points=points_list
            )
            current_image_skeletons.append(skeleton)

        all_skeletons_per_image.append(current_image_skeletons)

    return all_skeletons_per_image


def efficientnet_to_classifications(raw_results: List[Dict]) -> List[ClassificationResult]:
    """
    将 EfficientNet V2 API 的原始输出转换为 ClassificationResult 对象列表

    这个函数用于独立的 EfficientNet 推理。

    Args:
        raw_results: EfficientNetPipelineV2.infer() 返回的原始结果
                    格式:
                    [
                        {
                            "image_idx": int,
                            "class_id": int,
                            "confidence": float,
                            "logits": np.ndarray,     # shape: (num_classes,)
                            "features": np.ndarray    # shape: (feature_size,)
                        },
                        ...
                    ]

    Returns:
        List[ClassificationResult]: ClassificationResult 对象列表
    """
    classifications: List[ClassificationResult] = []

    for result in raw_results:
        # 转换 numpy 数组为列表
        logits = result['logits'].tolist() if isinstance(result['logits'], np.ndarray) else result['logits']
        features = result['features'].tolist() if isinstance(result['features'], np.ndarray) else result['features']

        classification = ClassificationResult(
            class_id=result['class_id'],
            confidence=result['confidence'],
            logits=logits,
            features=features
        )
        classifications.append(classification)

    return classifications


def cascade_results_to_unified(
    yolopose_results: List[Dict],
    efficientnet_results_per_detection: Dict[int, Dict]
) -> List[List[Dict]]:
    """
    将级联推理的结果合并为统一格式

    这个函数用于级联场景：先用 YoloPose 检测，然后对每个检测区域用 EfficientNet 分类。

    Args:
        yolopose_results: YoloPose 的原始输出
        efficientnet_results_per_detection: EfficientNet 结果字典
                                           key: (image_idx, detection_idx)
                                           value: EfficientNet 结果字典

    Returns:
        List[List[Dict]]: 每张图像的检测结果列表
                         每个检测包含: Skeleton + ClassificationResult
    """
    all_results_per_image: List[List[Dict]] = []

    for image_result in yolopose_results:
        image_idx = image_result['image_idx']
        current_image_results: List[Dict] = []

        # 转换 YoloPose 检测为 Skeleton
        skeletons = yolopose_to_skeletons([image_result])[0]

        for det_idx, skeleton in enumerate(skeletons):
            result_dict = {
                'skeleton': skeleton,
                'classification': None
            }

            # 查找对应的 EfficientNet 结果
            key = (image_idx, det_idx)
            if key in efficientnet_results_per_detection:
                eff_result = efficientnet_results_per_detection[key]
                classification = efficientnet_to_classifications([eff_result])[0]
                result_dict['classification'] = classification

            current_image_results.append(result_dict)

        all_results_per_image.append(current_image_results)

    return all_results_per_image


# 向后兼容的别名（用于从 V1 迁移）
def pipeline_v2_to_skeletons(raw_results: List[Dict]) -> List[List[Skeleton]]:
    """
    向后兼容的别名，映射到 yolopose_to_skeletons

    注意: 这个函数假设输入不包含 features（V2 架构）
    如果你的 V1 代码依赖 features，需要修改为使用级联模式
    """
    return yolopose_to_skeletons(raw_results)
