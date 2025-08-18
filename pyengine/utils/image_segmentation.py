import os
from typing import Iterable, List, Tuple, Union, Optional, Dict
import cv2
import numpy as np

Box = Tuple[int, int, int, int]

def crop_regions(
    img_or_path: Union[str, np.ndarray],
    boxes: Iterable[Box],
    save_dir: Optional[str] = None,
    clip: bool = True,
    pad: int = 0,
    min_size: int = 1,
    return_info: bool = True,
) -> Union[List[np.ndarray], Tuple[List[np.ndarray], List[Dict]]]:
    """
    批量从图像中裁剪小图（BGR），框格式为 (x1, y1, x2, y2)——左上到右下。

    参数
    ----
    img_or_path : str | np.ndarray
        图片路径或已读入的 BGR 图像 (cv2.imread 返回的格式)。
    boxes : Iterable[Tuple[int,int,int,int]]
        一组矩形框 (x1, y1, x2, y2)。允许给出越界或 x2<x1 / y2<y1，会自动纠正。
        注意：Python 切片是左闭右开，所以这里遵循常见像素区间，最终会做 clip/swap。
    save_dir : str | None
        若提供，将把裁剪图保存到该目录，文件名自动生成。
    clip : bool
        是否将框裁剪到图像边界内（推荐 True）。
    pad : int
        在四周额外留白像素（先 pad 后再 clip 到图像边界），可为 0。
    min_size : int
        过滤过小的框（宽或高小于该值的将跳过）。
    return_info : bool
        是否同时返回每个裁剪的元信息（坐标、保存路径等）。

    返回
    ----
    crops : List[np.ndarray]
        裁剪得到的小图（BGR）。
    infos : List[dict] (可选)
        与每个小图对应的信息，如 {"index", "box_in", "box_used", "shape", "save_path"}。

    备注
    ----
    - OpenCV 图像索引顺序为 [y, x]，因此裁剪时用 img[y1:y2, x1:x2]。
    - 坐标单位为像素，假设原点在左上角，x 向右、y 向下。
    """
    # 读图
    if isinstance(img_or_path, str):
        img = cv2.imread(img_or_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"无法读取图片：{img_or_path}")
        base_name = os.path.splitext(os.path.basename(img_or_path))[0]
    elif isinstance(img_or_path, np.ndarray):
        img = img_or_path
        base_name = "image"
    else:
        raise TypeError("img_or_path 需为路径 str 或 numpy.ndarray")

    h, w = img.shape[:2]

    # 保存目录
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    crops: List[np.ndarray] = []
    infos: List[Dict] = []

    for i, b in enumerate(boxes):
        if len(b) != 4:
            # 跳过非法框
            continue

        x1, y1, x2, y2 = b

        # 1) 纠正反向框
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # 2) pad
        if pad > 0:
            x1 -= pad; y1 -= pad; x2 += pad; y2 += pad

        # 3) clip 到图像边界
        if clip:
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))

        # 4) 过滤过小框
        if (x2 - x1) < min_size or (y2 - y1) < min_size:
            continue

        # 5) 裁剪（注意顺序：img[y1:y2, x1:x2]）
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        crops.append(crop)

        # 可选保存
        save_path = None
        if save_dir is not None:
            save_path = os.path.join(save_dir, f"{base_name}_crop_{i:03d}.png")
            cv2.imwrite(save_path, crop)

        if return_info:
            infos.append({
                "index": i,
                "box_in": tuple(b),
                "box_used": (int(x1), int(y1), int(x2), int(y2)),
                "shape": crop.shape,  # (h, w, c)
                "save_path": save_path,
            })

    return (crops, infos) if return_info else crops


# —— 示例用法 ——
if __name__ == "__main__":
    # 例子：从 test.jpg 中裁两块区域
    boxes = [
        (50, 40, 220, 180),   # 左上(50,40)，右下(220,180)
        (300, 100, 500, 320), # 第二个框
    ]
    crops, infos = crop_regions("test.jpg", boxes, save_dir="crops", pad=4)
    print(f"得到 {len(crops)} 个小图")
    for info in infos:
        print(info)
