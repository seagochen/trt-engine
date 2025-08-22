from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw

from pyengine.font import utils
from pyengine.font.utils import Point, BGRColor


# ========== 核心绘制 API ==========

def draw_text(frame: np.ndarray,
              text: str,
              top_left: Point,
              *,
              font_path: Optional[str] = None,
              font_size: int = 20,
              font_scale: float = 0.7,
              thickness: int = 1,
              color: Optional[BGRColor] = None,
              bg_color: Optional[BGRColor] = None,
              bg_alpha: float = 0.0,
              bg_padding: int = 4) -> np.ndarray:
    """
    在图像上绘制文本(支持 UTF-8 / 中文)，并可选带背景。
    - 提供 font_path ==> 使用 Pillow(支持 UTF-8)
    - 否则使用 OpenCV(ASCII/西文字体)
    坐标统一为「左上角」。
    """
    x, y = top_left

    # 1) 预先测量尺寸
    w, h, baseline = utils.measure_text(text, font_path, font_size, font_scale, thickness)
    text_rect = (x, y, w, h)  # 供自动取色与背景使用

    # 2) 自动前景色
    final_color: BGRColor = color if color is not None else utils.auto_text_color(frame, top_left, (w, h))

    if font_path:
        # --- Pillow 路径 ---
        font = utils.ensure_font(font_path, font_size)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # 背景
        if bg_color is not None:
            rect = (x - bg_padding, y - bg_padding, x + w + bg_padding, y + h + bg_padding)
            pil = utils.draw_bg_rect_pil(pil, rect, bg_color, bg_alpha)

        draw = ImageDraw.Draw(pil)
        r, g, b = final_color[2], final_color[1], final_color[0]
        draw.text((x, y), text, font=font, fill=(r, g, b))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        # --- OpenCV 路径 ---
        font_face = cv2.FONT_HERSHEY_SIMPLEX

        # 背景
        if bg_color is not None:
            lt = (x - bg_padding, y - bg_padding)
            rb = (x + w + bg_padding, y + h + baseline + bg_padding)
            utils.draw_bg_rect_cv2(frame, lt, rb, bg_color, bg_alpha)

        # OpenCV 的 org 是**左下角**，需要把 top_left 转换为基线坐标
        org = (x, y + h)
        cv2.putText(frame, text, org, font_face, font_scale, final_color, thickness, cv2.LINE_AA)
        return frame

def draw_text_with_outline(frame: np.ndarray,
                           text: str,
                           top_left: Point,
                           *,
                           font_path: Optional[str] = None,
                           font_size: int = 20,
                           font_scale: float = 0.7,
                           thickness: int = 1,
                           color: Optional[BGRColor] = None,
                           outline_color: Optional[BGRColor] = None,
                           outline_thickness: int = 2,
                           bg_color: Optional[BGRColor] = None,
                           bg_alpha: float = 0.0,
                           bg_padding: int = 4) -> np.ndarray:
    """
    绘制带描边的文本。
    - 默认行为等同“白字+黑描边”：若前景自动算出为白，则描边黑；若前景为黑，则描边白。
    - 支持 Pillow 或 OpenCV 两种后端(根据是否提供 font_path)。
    """
    x, y = top_left
    w, h, baseline = utils.measure_text(text, font_path, font_size, font_scale, thickness)
    auto_color = utils.auto_text_color(frame, top_left, (w, h))
    fg: BGRColor = color if color is not None else auto_color
    default_outline = (0, 0, 0) if fg == (255, 255, 255) else (255, 255, 255)
    ol: BGRColor = outline_color if outline_color is not None else default_outline

    # 背景(可选)
    if bg_color is not None:
        if font_path:
            pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            rect = (x - bg_padding, y - bg_padding, x + w + bg_padding, y + h + bg_padding)
            pil = utils.draw_bg_rect_pil(pil, rect, bg_color, bg_alpha)
            frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        else:
            lt = (x - bg_padding, y - bg_padding)
            rb = (x + w + bg_padding, y + h + baseline + bg_padding)
            utils.draw_bg_rect_cv2(frame, lt, rb, bg_color, bg_alpha)

    if font_path:
        # Pillow 版描边：先画描边(多方向偏移)，再画正文
        font = utils.ensure_font(font_path, font_size)
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        fr, fg_, fb = fg[2], fg[1], fg[0]
        or_, og, ob = ol[2], ol[1], ol[0]

        # 简单 8 邻域描边
        for dx in range(-outline_thickness, outline_thickness + 1):
            for dy in range(-outline_thickness, outline_thickness + 1):
                if dx == 0 and dy == 0:
                    continue
                draw.text((x + dx, y + dy), text, font=font, fill=(or_, og, ob))
        draw.text((x, y), text, font=font, fill=(fr, fg_, fb))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        # OpenCV 版描边：先粗笔描边，再细笔正文(你原始 put_text 思路的通用化)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        org = (x, y + h)
        cv2.putText(frame, text, org, font_face, font_scale, ol, max(thickness + outline_thickness, thickness), cv2.LINE_AA)
        cv2.putText(frame, text, org, font_face, font_scale, fg, thickness, cv2.LINE_AA)
        return frame