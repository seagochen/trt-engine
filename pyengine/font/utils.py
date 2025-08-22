# text_painter.py
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Tuple

BGRColor = Tuple[int, int, int]
Point = Tuple[int, int]

# ========== 低层工具(颜色/测量) ==========

def calculate_average_color(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """计算 bbox(x,y,w,h) 区域的平均 BGR 颜色。区域越接近真实文本背景，自动对比色越准确。"""
    x, y, w, h = bbox
    H, W = image.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)
    if x2 <= x1 or y2 <= y1:
        cy, cx = H // 2, W // 2
        if image.ndim == 2 or image.shape[2] == 1:
            return np.array([image[cy, cx]] * 3)
        return image[cy, cx]
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.array([128, 128, 128])
    if crop.ndim == 2 or crop.shape[2] == 1:
        v = float(np.mean(crop))
        return np.array([v, v, v])
    return np.average(np.average(crop, axis=0), axis=0)

def decide_text_color(avg_bgr: np.ndarray) -> BGRColor:
    """根据平均背景亮度返回黑/白(BGR)。"""
    # 亮度估计(BGR)：Y = 0.114*B + 0.587*G + 0.299*R
    lum = 0.114 * avg_bgr[0] + 0.587 * avg_bgr[1] + 0.299 * avg_bgr[2]
    return (0, 0, 0) if lum > 127 else (255, 255, 255)

def auto_text_color(frame: np.ndarray, top_left: Point, text_size: Tuple[int, int]) -> BGRColor:
    """对给定文本矩形自动决定前景色。text_size=(w,h)。"""
    x, y = top_left
    w, h = text_size
    avg = calculate_average_color(frame, (x, y, w, h))
    return decide_text_color(avg)

def ensure_font(font_path: Optional[str], font_size: int) -> Optional[ImageFont.FreeTypeFont]:
    if not font_path:
        return None
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found: {font_path}")
    return ImageFont.truetype(font_path, font_size)

# ========== 文本尺寸测量 ==========

def measure_text(text: str,
                 font_path: Optional[str] = None,
                 font_size: int = 20,
                 font_scale: float = 0.7,
                 thickness: int = 1) -> Tuple[int, int, int]:
    """
    返回 (width, height, baseline)。提供 font_path 则用 Pillow 测量，否则用 OpenCV。
    """
    if font_path:
        font = ensure_font(font_path, font_size)
        # 用 Pillow 的 textbbox 估算
        tmp = Image.new('RGB', (1, 1))
        draw = ImageDraw.Draw(tmp)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top
        baseline = 0  # Pillow 无基线概念，调用方无需关心
        return int(w), int(h), baseline
    else:
        (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        return int(w), int(h), int(baseline)

# ========== 背景绘制 ==========

def draw_bg_rect_cv2(frame: np.ndarray,
                     rect_lt: Point,
                     rect_rb: Point,
                     color: BGRColor,
                     alpha: float):
    if alpha >= 1.0:
        cv2.rectangle(frame, rect_lt, rect_rb, color, -1)
        return
    overlay = frame.copy()
    cv2.rectangle(overlay, rect_lt, rect_rb, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def draw_bg_rect_pil(pil_img: Image.Image,
                     rect: Tuple[int, int, int, int],
                     color_bgr: BGRColor,
                     alpha: float) -> Image.Image:
    # RGBA 叠加
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    r, g, b = color_bgr[2], color_bgr[1], color_bgr[0]
    draw.rectangle(rect, fill=(r, g, b, int(255 * alpha)))
    return Image.alpha_composite(pil_img.convert('RGBA'), overlay).convert('RGB')