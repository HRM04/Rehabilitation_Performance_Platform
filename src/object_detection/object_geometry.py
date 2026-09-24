import cv2
import numpy as np


def calculate_object_geometry(mask):
    """
    Estimate 2D geometry from a binary object mask.

    Returns:
        center: (x, y), in pixels
        orientation_deg: angle of the major axis relative
                         to the image's horizontal axis
        major_axis_px: length of the longer rectangle side
        minor_axis_px: length of the shorter rectangle side
        aspect_ratio: major_axis_px / minor_axis_px
        oriented_box: four corner points of the rotated box

    The orientation is an undirected image-plane axis:
    0 degrees is horizontal, and 90 degrees is vertical.
    It does not describe full 3D object orientation.
    """

    if mask is None or mask.ndim != 2:
        return None

    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) == 0:
        return None

    (center_x, center_y), (width, height), angle = (
        cv2.minAreaRect(contour)
    )

    major_axis_px = max(width, height)
    minor_axis_px = min(width, height)

    if minor_axis_px <= 0:
        return None

    # OpenCV reports the angle of one rectangle side.
    # Adjust it so the angle always describes the longer side.
    if width >= height:
        major_axis_angle = angle
    else:
        major_axis_angle = angle + 90

    # Treat opposite directions along the same axis as equivalent.
    orientation_deg = major_axis_angle % 180

    oriented_box = cv2.boxPoints(
        ((center_x, center_y), (width, height), angle)
    )

    aspect_ratio = major_axis_px / minor_axis_px

    orientation_reliable = aspect_ratio >= 1.2

    return {
        "center": (float(center_x), float(center_y)),
        "orientation_deg": float(orientation_deg),
        "major_axis_px": float(major_axis_px),
        "minor_axis_px": float(minor_axis_px),
        "aspect_ratio": float(aspect_ratio),
        "orientation_reliable": orientation_reliable,
        "oriented_box": np.rint(oriented_box).astype(np.int32),
    }