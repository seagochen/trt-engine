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


def nms_skeletons(
    sks: List[Skeleton],
    iou_threshold: float = 0.5,
    class_aware: bool = True
) -> List[Skeleton]:
    """Hard-NMS：同类间按置信度降序抑制。"""
    if not sks:
        return []
    order = sorted(range(len(sks)), key=lambda i: sks[i].confidence, reverse=True)
    picked: List[int] = []

    def same_class(i: int, j: int) -> bool:
        return (sks[i].classification == sks[j].classification) if class_aware else True

    for i in order:
        keep = True
        for j in picked:
            if not same_class(i, j):
                continue
            if iou(sks[i].rect, sks[j].rect) > iou_threshold:
                keep = False
                break
        if keep:
            picked.append(i)
    return [sks[i] for i in picked]
