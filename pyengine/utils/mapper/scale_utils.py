from pyengine.inference.unified_structs.inference_results import Rect


def scale_point(src_width: int, src_height: int, dst_width: int, dst_height: int, point: tuple) -> tuple:
    """将单个归一化的点 (x, y) 缩放到像素坐标"""

    scale_x = src_width / dst_width
    scale_y = src_height / dst_height
    return int(point[0] * scale_x), int(point[1] * scale_y)


def scale_rect(src_width: int, src_height: int, dst_width: int, dst_height: int, rect: Rect) -> tuple:
    """将归一化的矩形 Rect 缩放到像素坐标"""

    scale_x = src_width / dst_width
    scale_y = src_height / dst_height
    return (
        int(rect.x1 * scale_x),
        int(rect.y1 * scale_y),
        int(rect.x2 * scale_x),
        int(rect.y2 * scale_y)
    )
