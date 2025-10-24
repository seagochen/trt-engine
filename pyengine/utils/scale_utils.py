"""
Coordinate transformation utilities for multi-scale image processing.

This module provides functions to:
1. Transform coordinates between crop and full image space
2. Scale coordinates between different resolution spaces (e.g., model input vs original)
3. Handle domain objects (Rect, Point, Skeleton) transformations
"""

from typing import List, Tuple, Union, Sequence

from pyengine.inference.unified_structs.auxiliary_structs import ExpandedSkeleton
from pyengine.inference.unified_structs.inference_results import Rect, Skeleton, Point

# Type aliases
Coordinate = Tuple[float, float]  # Generic (x, y) coordinate
Size = Tuple[int, int]  # (width, height)
LTRB = Tuple[int, int, int, int]  # (left, top, right, bottom)


# =============================================================================
# Validation utilities
# =============================================================================

def _validate_size(size: Size, name: str = "size") -> None:
    """Validate that size is a tuple of positive integers."""
    W, H = size
    if not (isinstance(W, int) and isinstance(H, int) and W > 0 and H > 0):
        raise ValueError(f"{name} must be positive ints (W, H), got {size}")


def _validate_ltrb(ltrb: LTRB) -> None:
    """Validate that LTRB is properly formatted."""
    l, t, r, b = ltrb
    if not all(isinstance(v, int) for v in (l, t, r, b)):
        raise ValueError(f"LTRB must be ints (left, top, right, bottom), got {ltrb}")
    if not (l <= r and t <= b):
        raise ValueError(f"Invalid LTRB (left<=right, top<=bottom required), got {ltrb}")


def _validate_coordinate(coord: Coordinate) -> None:
    """Validate that coordinate is a tuple of numeric values."""
    x, y = coord
    if not (isinstance(x, (int, float)) and isinstance(y, (int, float))):
        raise ValueError(f"Coordinate must be numeric (x, y), got {coord}")


# =============================================================================
# Core transformation: Crop <-> Full image
# =============================================================================

def crop_to_full(
        crop_ltrb: LTRB,
        coord_in_crop: Coordinate,
        full_size: Size,
        *,
        clamp: bool = False,
) -> Coordinate:
    """
    Transform a coordinate from crop-local space to full-image space.

    Args:
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        coord_in_crop: (x, y) in crop's local coordinates, origin at crop's top-left
        full_size: (width, height) of the full image
        clamp: If True, clamp result to [0, width-1] × [0, height-1]

    Returns:
        (x, y) in full-image coordinates
    """
    _validate_ltrb(crop_ltrb)
    _validate_coordinate(coord_in_crop)
    _validate_size(full_size, "full_size")

    left, top, _, _ = crop_ltrb
    x_local, y_local = coord_in_crop
    full_w, full_h = full_size

    # Translation: full = crop_origin + local
    x_full = left + x_local
    y_full = top + y_local

    if clamp:
        x_full = max(0.0, min(x_full, full_w - 1))
        y_full = max(0.0, min(y_full, full_h - 1))

    return (float(x_full), float(y_full))


def full_to_crop(
        crop_ltrb: LTRB,
        coord_in_full: Coordinate,
        full_size: Size,
) -> Coordinate:
    """
    Transform a coordinate from full-image space to crop-local space.

    Args:
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        coord_in_full: (x, y) in full-image coordinates
        full_size: (width, height) of the full image

    Returns:
        (x, y) in crop's local coordinates
    """
    _validate_ltrb(crop_ltrb)
    _validate_coordinate(coord_in_full)
    _validate_size(full_size, "full_size")

    left, top, _, _ = crop_ltrb
    x_full, y_full = coord_in_full

    return (float(x_full - left), float(y_full - top))


def batch_crop_to_full(
        crop_ltrb: LTRB,
        coords_in_crop: Sequence[Coordinate],
        full_size: Size,
        *,
        clamp: bool = False,
) -> List[Coordinate]:
    """Batch version of crop_to_full."""
    return [
        crop_to_full(crop_ltrb, coord, full_size, clamp=clamp)
        for coord in coords_in_crop
    ]


def batch_full_to_crop(
        crop_ltrb: LTRB,
        coords_in_full: Sequence[Coordinate],
        full_size: Size,
) -> List[Coordinate]:
    """Batch version of full_to_crop."""
    return [
        full_to_crop(crop_ltrb, coord, full_size)
        for coord in coords_in_full
    ]


# =============================================================================
# Core transformation: Scale between different resolutions
# =============================================================================

def scale_coordinate(
        coord: Coordinate,
        src_size: Size,
        dst_size: Size,
) -> Coordinate:
    """
    Scale a coordinate from source resolution to destination resolution.

    Args:
        coord: (x, y) in source resolution
        src_size: (width, height) of source resolution
        dst_size: (width, height) of destination resolution

    Returns:
        (x, y) in destination resolution
    """
    _validate_coordinate(coord)
    _validate_size(src_size, "src_size")
    _validate_size(dst_size, "dst_size")

    src_w, src_h = src_size
    dst_w, dst_h = dst_size

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    x, y = coord
    return (x * scale_x, y * scale_y)


def batch_scale_coordinates(
        coords: Sequence[Coordinate],
        src_size: Size,
        dst_size: Size,
) -> List[Coordinate]:
    """Batch version of scale_coordinate."""
    return [
        scale_coordinate(coord, src_size, dst_size)
        for coord in coords
    ]


# =============================================================================
# Domain object transformations: Point
# =============================================================================

def scale_point(
        point: Point,
        src_size: Size,
        dst_size: Size,
) -> Point:
    """
    Scale a Point object from source resolution to destination resolution.

    Args:
        point: Point object with x, y, confidence
        src_size: (width, height) of source resolution
        dst_size: (width, height) of destination resolution

    Returns:
        New Point object in destination resolution
    """
    _validate_size(src_size, "src_size")
    _validate_size(dst_size, "dst_size")

    src_w, src_h = src_size
    dst_w, dst_h = dst_size

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    return Point(
        int(point.x * scale_x),
        int(point.y * scale_y),
        point.confidence
    )


def batch_scale_points(
        points: List[Point],
        src_size: Size,
        dst_size: Size,
) -> List[Point]:
    """Batch version of scale_point."""
    return [scale_point(p, src_size, dst_size) for p in points]


# =============================================================================
# Domain object transformations: Rect
# =============================================================================

def scale_rect(
        rect: Rect,
        src_size: Size,
        dst_size: Size,
) -> Rect:
    """
    Scale a Rect object from source resolution to destination resolution.

    Args:
        rect: Rect object with x1, y1, x2, y2
        src_size: (width, height) of source resolution
        dst_size: (width, height) of destination resolution

    Returns:
        New Rect object in destination resolution
    """
    _validate_size(src_size, "src_size")
    _validate_size(dst_size, "dst_size")

    src_w, src_h = src_size
    dst_w, dst_h = dst_size

    scale_x = dst_w / src_w
    scale_y = dst_h / src_h

    return Rect(
        int(rect.x1 * scale_x),
        int(rect.y1 * scale_y),
        int(rect.x2 * scale_x),
        int(rect.y2 * scale_y)
    )


# =============================================================================
# Combined transformations: Scale + Crop-to-Full projection
# =============================================================================

def transform_coordinate_to_full(
        coord: Coordinate,
        crop_ltrb: LTRB,
        full_size: Size,
        *,
        src_size: Size | None = None,
        clamp: bool = False,
) -> Coordinate:
    """
    Transform a coordinate to full-image space, with optional scaling.

    This is a convenience function that combines:
    1. Optional scaling from src_size to crop size
    2. Projection from crop to full image

    Args:
        coord: (x, y) in source resolution (or crop resolution if src_size is None)
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        full_size: (width, height) of the full image
        src_size: If provided, scale from this size to crop size first
        clamp: If True, clamp result to full image bounds

    Returns:
        (x, y) in full-image coordinates
    """
    x, y = coord

    # Step 1: Scale to crop size if needed
    if src_size is not None:
        l, t, r, b = crop_ltrb
        crop_size = (r - l, b - t)
        x, y = scale_coordinate((x, y), src_size, crop_size)

    # Step 2: Project to full image
    return crop_to_full(crop_ltrb, (x, y), full_size, clamp=clamp)


def transform_point_to_full(
        point: Point,
        crop_ltrb: LTRB,
        full_size: Size,
        *,
        src_size: Size | None = None,
        clamp: bool = False,
) -> Point:
    """
    Transform a Point object to full-image space, with optional scaling.

    Args:
        point: Point object in source resolution
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        full_size: (width, height) of the full image
        src_size: If provided (e.g., model input size), scale from this size to crop size first
        clamp: If True, clamp result to full image bounds

    Returns:
        New Point object in full-image coordinates
    """
    # Step 1: Scale to crop size if needed
    if src_size is not None:
        l, t, r, b = crop_ltrb
        crop_size = (r - l, b - t)
        point = scale_point(point, src_size, crop_size)

    # Step 2: Project to full image
    x_full, y_full = crop_to_full(
        crop_ltrb,
        (point.x, point.y),
        full_size,
        clamp=clamp
    )

    return Point(x_full, y_full, point.confidence)


def batch_transform_points_to_full(
        points: List[Point],
        crop_ltrb: LTRB,
        full_size: Size,
        *,
        src_size: Size | None = None,
        clamp: bool = False,
) -> List[Point]:
    """Batch version of transform_point_to_full."""
    return [
        transform_point_to_full(p, crop_ltrb, full_size, src_size=src_size, clamp=clamp)
        for p in points
    ]


def transform_rect_to_full(
        rect: Rect,
        crop_ltrb: LTRB,
        full_size: Size,
        *,
        src_size: Size | None = None,
        clamp: bool = False,
) -> Rect:
    """
    Transform a Rect object to full-image space, with optional scaling.

    Args:
        rect: Rect object in source resolution
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        full_size: (width, height) of the full image
        src_size: If provided, scale from this size to crop size first
        clamp: If True, clamp result to full image bounds

    Returns:
        New Rect object in full-image coordinates
    """
    # Step 1: Scale to crop size if needed
    if src_size is not None:
        l, t, r, b = crop_ltrb
        crop_size = (r - l, b - t)
        rect = scale_rect(rect, src_size, crop_size)

    # Step 2: Project corners to full image
    x1_full, y1_full = crop_to_full(crop_ltrb, (rect.x1, rect.y1), full_size, clamp=clamp)
    x2_full, y2_full = crop_to_full(crop_ltrb, (rect.x2, rect.y2), full_size, clamp=clamp)

    return Rect(x1_full, y1_full, x2_full, y2_full)


# =============================================================================
# Domain object transformations: Skeleton
# =============================================================================

def transform_skeleton_to_full(
        skeleton: Union[Skeleton, ExpandedSkeleton],
        crop_ltrb: LTRB,
        full_size: Size,
        *,
        src_size: Size | None = None,
        clamp: bool = False,
        in_place: bool = False,
) -> Union[Skeleton, ExpandedSkeleton]:
    """
    Transform a Skeleton object to full-image space, with optional scaling.

    Args:
        skeleton: Skeleton or ExpandedSkeleton object
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        full_size: (width, height) of the full image
        src_size: If provided (e.g., model input size), scale from this size to crop size first
        clamp: If True, clamp result to full image bounds
        in_place: If True, modify the skeleton in place; otherwise create a new one

    Returns:
        Skeleton or ExpandedSkeleton in full-image coordinates
    """
    # Transform points and rect
    points = batch_transform_points_to_full(
        skeleton.points,
        crop_ltrb,
        full_size,
        src_size=src_size,
        clamp=clamp
    )

    rect = transform_rect_to_full(
        skeleton.rect,
        crop_ltrb,
        full_size,
        src_size=src_size,
        clamp=clamp
    )

    if in_place:
        skeleton.points = points
        skeleton.rect = rect

        # Handle ExpandedSkeleton's direction_origin
        if isinstance(skeleton, ExpandedSkeleton):
            skeleton.direction_origin = transform_coordinate_to_full(
                skeleton.direction_origin,
                crop_ltrb,
                full_size,
                src_size=src_size,
                clamp=clamp
            )

        return skeleton
    else:
        # Create new skeleton
        if isinstance(skeleton, ExpandedSkeleton):
            direction_origin = transform_coordinate_to_full(
                skeleton.direction_origin,
                crop_ltrb,
                full_size,
                src_size=src_size,
                clamp=clamp
            )

            return ExpandedSkeleton(
                rect=rect,
                classification=skeleton.classification,
                confidence=skeleton.confidence,
                track_id=skeleton.track_id,
                features=(skeleton.features.copy() if skeleton.features else []),
                points=points,
                posture_type=skeleton.posture_type,
                cx=skeleton.cx,
                cy=skeleton.cy,
                direction_type=skeleton.direction_type,
                direction_angle=skeleton.direction_angle,
                direction_modulus=skeleton.direction_modulus,
                direction_vector=skeleton.direction_vector,
                direction_origin=direction_origin
            )
        elif isinstance(skeleton, Skeleton):
            return Skeleton(
                rect=rect,
                classification=skeleton.classification,
                confidence=skeleton.confidence,
                track_id=skeleton.track_id,
                features=(skeleton.features.copy() if skeleton.features else []),
                points=points
            )
        else:
            raise TypeError(f"Unsupported skeleton type: {type(skeleton)}")


def batch_transform_skeletons_to_full(
        skeletons: List[Union[Skeleton, ExpandedSkeleton]],
        crop_ltrb: LTRB,
        full_size: Size,
        *,
        src_size: Size | None = None,
        clamp: bool = False,
        in_place: bool = False,
) -> List[Union[Skeleton, ExpandedSkeleton]]:
    """
    Batch version of transform_skeleton_to_full.

    Args:
        skeletons: List of Skeleton or ExpandedSkeleton objects
        crop_ltrb: (left, top, right, bottom) of the crop in full image
        full_size: (width, height) of the full image
        src_size: If provided (e.g., model input size like (640, 640)),
                  scale from this size to crop size first
        clamp: If True, clamp results to full image bounds
        in_place: If True, modify skeletons in place; otherwise create new ones

    Returns:
        List of transformed skeletons in full-image coordinates
    """
    return [
        transform_skeleton_to_full(
            sk,
            crop_ltrb,
            full_size,
            src_size=src_size,
            clamp=clamp,
            in_place=in_place
        )
        for sk in skeletons
    ]