"""Shoulders-only posture status, for partial posture calibrations (hips not visible).

Uses only the two shoulder landmarks against the calibration reference: upward rise of either
shoulder, tilt of the shoulder line, and sideways shift of the shoulder midpoint. The thresholds are the
same provisional values the full-calibration overlay uses. They are not validated, and this is an
image-plane descriptor, not a compensation classification.
"""
import math

ELEVATION_MAX = 0.03
SHIFT_MAX = 0.03
TILT_MAX_DEG = 10.0
MIN_VIS = 0.5
OK_COLOR = (0, 200, 0)
WARN_COLOR = (0, 165, 255)
NA_COLOR = (0, 255, 255)


def _pts(landmarks):
    if not landmarks:
        return None
    left, right = landmarks.get("left_shoulder"), landmarks.get("right_shoulder")
    if not left or not right:
        return None
    if left.get("visibility", 0) < MIN_VIS or right.get("visibility", 0) < MIN_VIS:
        return None
    return left, right


def shoulder_tilt_change_deg(baseline, current):
    b, c = _pts(baseline), _pts(current)
    if b is None or c is None:
        return None
    tilt = lambda p: math.degrees(math.atan2(p[1]["y"] - p[0]["y"], p[1]["x"] - p[0]["x"]))
    return tilt(c) - tilt(b)


def shoulder_midpoint_shift_norm(baseline, current):
    b, c = _pts(baseline), _pts(current)
    if b is None or c is None:
        return None
    mid = lambda p: (p[0]["x"] + p[1]["x"]) / 2.0
    return mid(c) - mid(b)


def shoulder_rise_norm(baseline, current):
    """Largest upward movement of either shoulder (positive = up in the image)."""
    b, c = _pts(baseline), _pts(current)
    if b is None or c is None:
        return None
    return max(b[0]["y"] - c[0]["y"], b[1]["y"] - c[1]["y"])


def shoulder_only_status(baseline, current):
    if shoulder_rise_norm(baseline, current) is None:
        return "Posture (shoulders only): shoulders not visible", NA_COLOR
    tilt = abs(shoulder_tilt_change_deg(baseline, current))
    shift = abs(shoulder_midpoint_shift_norm(baseline, current))
    rise = shoulder_rise_norm(baseline, current)
    if tilt > TILT_MAX_DEG or shift > SHIFT_MAX:
        return "Posture (shoulders only): tilt or shift detected", WARN_COLOR
    if rise > ELEVATION_MAX:
        return "Posture (shoulders only): shoulder elevation detected", WARN_COLOR
    return "Posture (shoulders only): within provisional range", OK_COLOR
