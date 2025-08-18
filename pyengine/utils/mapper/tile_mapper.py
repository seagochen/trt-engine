"""
tile_mapper.py

Utilities to project coordinates between a cropped tile (subregion) 
and the full image.
"""

from typing import List, Sequence, Tuple

Point = Tuple[float, float]
Size = Tuple[int, int]
LTRB = Tuple[int, int, int, int]  # (left, top, right, bottom)


def _validate_full_size(full_size: Size) -> None:
    W, H = full_size
    if not (isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0):
        raise ValueError(f"full_size must be positive ints (W,H), got {full_size}")


def _validate_ltrb(crop_ltrb: LTRB) -> None:
    l, t, r, b = crop_ltrb
    if not (isinstance(l, int) and isinstance(t, int) and isinstance(r, int) and isinstance(b, int)):
        raise ValueError("crop_ltrb must be ints (left, top, right, bottom)")
    if not (l <= r and t <= b):
        raise ValueError(f"Invalid crop_ltrb (left<=right, top<=bottom required), got {crop_ltrb}")


def _validate_point(p: Point) -> None:
    x, y = p
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError(f"point must be numeric (x,y), got {p}")


def project_point_to_full(
    full_size: Size,
    crop_ltrb: LTRB,
    point_in_crop: Point,
    *, clamp_to_full: bool = False,
) -> Point:
    """
    Map a point from a crop's local coordinates to full-image coordinates.

    Args:
        full_size: (W, H) of the full image.
        crop_ltrb: (left, top, right, bottom) of the crop in full-image pixels.
                   Only 'left' and 'top' matter for translation.
        point_in_crop: (x, y) in the crop's local pixel coordinates,
                       with (0,0) at crop's top-left.
        clamp_to_full: if True, clamp the projected point into [0..W-1], [0..H-1].

    Returns:
        (X, Y) in full-image coordinates.
    """
    _validate_full_size(full_size)
    _validate_ltrb(crop_ltrb)
    _validate_point(point_in_crop)

    full_w, full_h = full_size
    left, top, right, bottom = crop_ltrb
    x_local, y_local = point_in_crop

    # Basic translation: global = crop origin + local
    X = left + x_local
    Y = top + y_local

    if clamp_to_full:
        X = min(max(0.0, X), full_w - 1)
        Y = min(max(0.0, Y), full_h - 1)

    return (float(X), float(Y))


def project_points_to_full(
    full_size: Size,
    crop_ltrb: LTRB,
    points_in_crop: Sequence[Point],
    *, clamp_to_full: bool = False,
) -> List[Point]:
    """
    Vectorized version for a list/sequence of points.
    """
    return [
        project_point_to_full(full_size, crop_ltrb, p, clamp_to_full=clamp_to_full)
        for p in points_in_crop
    ]


# (Optional) inverse helpers ---------------------------------------------------

def project_point_to_crop(
    full_size: Size,
    crop_ltrb: LTRB,
    point_in_full: Point,
) -> Point:
    """
    Map a point from full-image coordinates into the crop's local coordinates.
    """
    _validate_full_size(full_size)
    _validate_ltrb(crop_ltrb)
    _validate_point(point_in_full)

    left, top, _, _ = crop_ltrb
    X, Y = point_in_full
    return (float(X - left), float(Y - top))


def project_points_to_crop(
    full_size: Size,
    crop_ltrb: LTRB,
    points_in_full: Sequence[Point],
) -> List[Point]:
    return [project_point_to_crop(full_size, crop_ltrb, p) for p in points_in_full]
