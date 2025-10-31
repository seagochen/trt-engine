from typing import List

from pyengine.inference.unified_structs.inference_results import Skeleton, Rect


def iou(a: Rect, b: Rect) -> float:
    ax1, ay1, ax2, ay2 = a.x1, a.y1, a.x2, a.y2
    bx1, by1, bx2, by2 = b.x1, b.y1, b.x2, b.y2
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def center_distance_normalized(a: Rect, b: Rect) -> float:
    """
    计算两个矩形中心点的归一化距离。
    归一化方式：除以两个矩形平均尺寸的对角线长度。
    返回值越小表示中心越接近。
    """
    # 计算中心点
    ax_center = (a.x1 + a.x2) / 2.0
    ay_center = (a.y1 + a.y2) / 2.0
    bx_center = (b.x1 + b.x2) / 2.0
    by_center = (b.y1 + b.y2) / 2.0

    # 欧几里得距离
    dist = ((ax_center - bx_center) ** 2 + (ay_center - by_center) ** 2) ** 0.5

    # 归一化：使用两个框的平均对角线长度
    a_diag = ((a.x2 - a.x1) ** 2 + (a.y2 - a.y1) ** 2) ** 0.5
    b_diag = ((b.x2 - b.x1) ** 2 + (b.y2 - b.y1) ** 2) ** 0.5
    avg_diag = (a_diag + b_diag) / 2.0

    if avg_diag <= 0.0:
        return float('inf')

    return dist / avg_diag


def nms_skeletons(
    sks: List[Skeleton],
    iou_threshold: float = 0.5,
    class_aware: bool = True,
    center_dist_threshold: float = 0.5,
    debug: bool = False
) -> List[Skeleton]:
    """
    Hard-NMS：同类间按置信度降序抑制。

    增强版 NMS，除了 IoU，还考虑中心距离。这对于处理 tile 边界的检测框特别有用。

    Args:
        sks: 骨架列表
        iou_threshold: IoU 阈值，超过此值认为重叠
        class_aware: 是否只在同类间做 NMS
        center_dist_threshold: 归一化中心距离阈值。当 IoU=0 但中心距离小于此值时，
                              也认为是同一个物体（用于 tile 边界情况）
        debug: 是否输出调试信息
    """
    if not sks:
        return []
    order = sorted(range(len(sks)), key=lambda i: sks[i].confidence, reverse=True)
    picked: List[int] = []

    def same_class(i: int, j: int) -> bool:
        return (sks[i].classification == sks[j].classification) if class_aware else True

    if debug:
        from pyengine.utils.logger import logger
        logger.debug("NMS", f"Processing {len(sks)} detections")
        for idx, sk in enumerate(sks):
            logger.debug("NMS", f"  [{idx}] bbox=({sk.rect.x1:.0f}, {sk.rect.y1:.0f}, {sk.rect.x2:.0f}, {sk.rect.y2:.0f}), conf={sk.confidence:.3f}, cls={sk.classification}")

    for i in order:
        keep = True
        for j in picked:
            if not same_class(i, j):
                continue

            # 计算 IoU
            overlap = iou(sks[i].rect, sks[j].rect)

            # 如果 IoU 超过阈值，直接抑制
            if overlap > iou_threshold:
                if debug:
                    from pyengine.utils.logger import logger
                    logger.debug("NMS", f"  Suppress [{i}] by [{j}]: IoU={overlap:.4f} > {iou_threshold}")
                keep = False
                break

            # 检查中心距离（特别适用于 tile 边界情况，即使有一些小的 IoU）
            # 当 IoU 未达到抑制阈值时，检查中心距离作为补充判定
            if overlap <= iou_threshold:
                center_dist = center_distance_normalized(sks[i].rect, sks[j].rect)
                if debug:
                    from pyengine.utils.logger import logger
                    logger.debug("NMS", f"  Compare [{i}] vs [{j}]: IoU={overlap:.4f}, CenterDist={center_dist:.4f}")
                if center_dist < center_dist_threshold:
                    if debug:
                        from pyengine.utils.logger import logger
                        logger.debug("NMS", f"  Suppress [{i}] by [{j}]: CenterDist={center_dist:.4f} < {center_dist_threshold}")
                    keep = False
                    break

        if keep:
            picked.append(i)

    if debug:
        from pyengine.utils.logger import logger
        logger.debug("NMS", f"Kept {len(picked)} detections: {picked}")

    return [sks[i] for i in picked]
