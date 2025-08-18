from typing import Tuple


def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string (e.g. "#FF00AA") to a BGR tuple.
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return b, g, r  # Return in BGR order for OpenCV


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    """
    Convert a hex color string (e.g. "#FF00AA) to a RGB tuple
    """
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return r, g, b


def bgr_to_hex(bgr_color: Tuple[int, int, int]) -> str:
    """
    Convert a BGR tuple to a hex color string.
    
    Args:
        bgr_color (Tuple[int, int, int]): A tuple of BGR values (0-255).
        
    Returns:
        str: Hex color string in the format "#RRGGBB".
    """
    b, g, r = bgr_color
    return "#{:02X}{:02X}{:02X}".format(r, g, b)  # RGB hex format


def rgb_to_hex(rgb_color: Tuple[int, int, int]) -> str:
    """
    Convert an RGB tuple to a hex color string.

    Args:
        rgb_color (Tuple[int, int, int]): RGB tuple (0-255 each).

    Returns:
        str: Hex color string in the format "#RRGGBB".
    """
    r, g, b = rgb_color
    return "#{:02X}{:02X}{:02X}".format(r, g, b)
