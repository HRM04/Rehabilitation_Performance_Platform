"""Handbook E23: posture calibration computation (written 2 Nov 2026).

Frames have the shape the pipeline uses: {"landmarks": {name: {"x", "y", "visibility"}}}.
Jitter alternates +/- around a fixed point, so the median absolute deviation equals the jitter size.
"""
import json
import math

import pytest

from src.assessment.posture_calibration import (
    POSTURE_LANDMARKS, REQUIRED_LANDMARKS, calibrate_reference_posture,
    calibration_subset, landmark_visible, usable_frames)
from src.assessment.posture_calibration_session import prepare_posture_calibration
from src.assessment.posture_stability import assess_posture_stability


def pt(x, y, v=0.9):
    return {"x": x, "y": y, "visibility": v}


def frame(i=0, hips=True, shoulder_jitter=0.0, hip_jitter=0.0, vis=0.9):
    s = shoulder_jitter if i % 2 == 0 else -shoulder_jitter
    h = hip_jitter if i % 2 == 0 else -hip_jitter
    lm = {"left_shoulder": pt(0.40 + s, 0.50 + s, vis), "right_shoulder": pt(0.60 + s, 0.50 + s, vis)}
    if hips:
        lm["left_hip"] = pt(0.42 + h, 0.90 + h, vis)
        lm["right_hip"] = pt(0.58 + h, 0.90 + h, vis)
    return {"landmarks": lm}


def frames(n, **kw):
    return [frame(i, **kw) for i in range(n)]


# ---- visibility and frame selection
def test_E23_landmark_visible_rules():
    lm = {"a": pt(0.5, 0.5, 0.9)}
    assert landmark_visible(lm, "a")
    assert landmark_visible({"a": pt(0.5, 0.5, 0.5)}, "a")            # threshold is inclusive
    assert not landmark_visible({"a": pt(0.5, 0.5, 0.49)}, "a")
    assert not landmark_visible(lm, "missing")
    assert not landmark_visible({"a": "text"}, "a")
    assert not landmark_visible({"a": pt(float("nan"), 0.5)}, "a")
    assert not landmark_visible({"a": pt(0.5, float("inf"))}, "a")
    assert not landmark_visible({"a": pt(0.5, 0.5, float("nan"))}, "a")
    assert not landmark_visible({"a": {"x": "abc", "y": 0.5, "visibility": 0.9}}, "a")
    assert not landmark_visible({"a": {"x": 0.5, "y": 0.5}}, "a")        # no visibility counts as 0


def test_E23_usable_frames_filters_bad_input():
    good = frame(0)
    no_right = {"landmarks": {"left_shoulder": pt(0.4, 0.5)}}
    low = frame(1, vis=0.3)
    mixed = [good, "junk", None, {"landmarks": None}, {}, no_right, low]
    assert usable_frames(mixed) == [good["landmarks"]]
    assert usable_frames(None) == []
    assert usable_frames([]) == []


def test_E23_calibration_subset_modes():
    assert calibration_subset(frames(9, hips=False)) == (None, None, None)
    sub, names, mode = calibration_subset(frames(10, hips=False))
    assert mode == "PARTIAL" and names == REQUIRED_LANDMARKS and len(sub) == 10
    sub, names, mode = calibration_subset(frames(10))
    assert mode == "FULL" and names == POSTURE_LANDMARKS and len(sub) == 10
    mix = frames(9) + frames(3, hips=False)                              # only 9 frames with hips
    sub, names, mode = calibration_subset(mix)
    assert mode == "PARTIAL" and len(sub) == 12


# ---- stability
def test_E23_stability_statuses():
    ok = assess_posture_stability(frames(12, shoulder_jitter=0.001, hip_jitter=0.001))
    assert ok["status"] == "STABLE" and ok["stable"] is True and ok["calibration_mode"] == "FULL"
    bad = assess_posture_stability(frames(12, shoulder_jitter=0.05, hip_jitter=0.05))
    assert bad["status"] == "UNSTABLE" and bad["stable"] is False
    few = assess_posture_stability(frames(9))
    assert few["status"] == "INSUFFICIENT_DATA" and few["valid_frames"] == 9 and few["stable"] is None


def test_E23_stability_threshold_boundary():
    assert assess_posture_stability(frames(12, shoulder_jitter=0.0149, hip_jitter=0.0149))["status"] == "STABLE"
    assert assess_posture_stability(frames(12, shoulder_jitter=0.0151, hip_jitter=0.0151))["status"] == "UNSTABLE"
    # a looser limit accepts the same movement
    assert assess_posture_stability(frames(12, shoulder_jitter=0.0151, hip_jitter=0.0151), maximum_mad_norm=0.02)["status"] == "STABLE"


def test_E23_unstable_hips_fall_back_to_shoulders():
    r = assess_posture_stability(frames(12, shoulder_jitter=0.001, hip_jitter=0.05))
    assert r["status"] == "STABLE" and r["calibration_mode"] == "PARTIAL"


def test_E23_unstable_shoulders_fail_even_with_steady_hips():
    r = assess_posture_stability(frames(12, shoulder_jitter=0.05, hip_jitter=0.001))
    assert r["status"] == "UNSTABLE"


def test_E23_invalid_parameters_raise():
    with pytest.raises(ValueError):
        assess_posture_stability(frames(12), min_valid_frames=0)
    with pytest.raises(ValueError):
        assess_posture_stability(frames(12), maximum_mad_norm=-0.1)
    with pytest.raises(ValueError):
        calibrate_reference_posture(frames(12), min_valid_frames=0)


# ---- session preparation
def test_E23_prepare_no_recording():
    r = prepare_posture_calibration(None)
    assert r["status"] == "UNAVAILABLE" and r["reference_posture"] is None


def test_E23_prepare_insufficient_and_unstable_give_no_reference():
    r = prepare_posture_calibration(frames(6))
    assert r["status"] == "INSUFFICIENT_DATA" and r["reference_posture"] is None
    assert r["valid_frames"] == 6 and r["required_valid_frames"] == 10
    r = prepare_posture_calibration(frames(12, shoulder_jitter=0.05, hip_jitter=0.05))
    assert r["status"] == "UNSTABLE" and r["reference_posture"] is None


def test_E23_prepare_stable_gives_reference():
    r = prepare_posture_calibration(frames(12, shoulder_jitter=0.001, hip_jitter=0.001))
    assert r["status"] not in ("UNAVAILABLE", "INSUFFICIENT_DATA", "UNSTABLE")
    assert r["reference_posture"] is not None


def test_E23_prepare_partial_reference_has_no_hips():
    r = prepare_posture_calibration(frames(12, shoulder_jitter=0.001, hip_jitter=0.05))
    assert r["reference_posture"] is not None
    def keys(o):
        if isinstance(o, dict):
            for k, v in o.items():
                yield str(k)
                yield from keys(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                yield from keys(v)
    assert not [k for k in keys(r["reference_posture"]) if k in ("left_hip", "right_hip")]
