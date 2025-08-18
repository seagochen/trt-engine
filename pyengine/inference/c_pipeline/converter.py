# pyengine/data_conversion/pipeline_converter.py

from typing import List, Dict

# 导入你的数据结构
from pyengine.inference.unified_structs.inference_results import Rect, Point, Skeleton


def pipeline_v1_to_skeletons(raw_pipeline_results: List[Dict]) -> List[List[Skeleton]]:
    """
    将 C++ pipeline 的原始输出转换为 Skeleton 数据类对象的列表。

    Args:
        raw_pipeline_results: PosePipeline.process_batched_images 返回的原始结果列表。
                              格式为：
                              [
                                  {"image_idx": int,
                                   "detections": [{"bbox": [lx, ly, rx, ry],
                                                   "classification": int,
                                                   "confidence": float,
                                                   "keypoints": [{"x": float, "y": float, "conf": float}, ...],
                                                   "features": np.ndarray}, ...] # 注意 features 此时是 np.ndarray
                                  }, ...
                              ]

    Returns:
        List[List[Skeleton]]: 一个嵌套列表，外层列表对应每个图像，内层列表是该图像的 Skeleton 对象。
    """
    all_skeletons_per_image: List[List[Skeleton]] = []

    for image_result_dict in raw_pipeline_results:
        current_image_skeletons: List[Skeleton] = [] # 当前图像的 Skeleton 列表
        raw_detections_for_image = image_result_dict['detections'] # 获取当前图像的检测结果列表

        for det in raw_detections_for_image: # 遍历每个检测结果
            # 实例化 Rect
            bbox_raw = det['bbox'] # 获取边界框

            # C++ 返回的 bbox 是 List[int]，Rect 期望 float，所以进行转换
            rect = Rect(x1=float(bbox_raw[0]), y1=float(bbox_raw[1]), x2=float(bbox_raw[2]), y2=float(bbox_raw[3])) # 转换为 Rect 对象

            # 实例化 Point 列表
            points_list: List[Point] = []
            keypoints_raw = det['keypoints']

            for kp_dict in keypoints_raw: # 将每个关键点字典转换为 Point 对象
                points_list.append(Point(x=kp_dict['x'], y=kp_dict['y'], confidence=kp_dict['conf']))

            # 获取 features
            # C++ 返回的 features 是 np.ndarray，ObjectDetection 期望 List[float]
            features = det['features'].tolist() if det['features'] is not None else [] # 转换为 List[float]，如果 features 为空则使用空列表

            # 实例化 Skeleton (继承 ObjectDetection)
            skeleton = Skeleton(
                rect=rect,
                classification=det['classification'],
                confidence=det['confidence'],
                features=features,
                points=points_list
            )
            current_image_skeletons.append(skeleton)

        all_skeletons_per_image.append(current_image_skeletons)

    return all_skeletons_per_image
