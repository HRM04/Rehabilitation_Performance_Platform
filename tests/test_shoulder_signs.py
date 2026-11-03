from src.assessment.posture_metrics import calculate_shoulder_elevation, calculate_shoulder_vertical_displacement
from src.assessment.shoulder_only_posture import shoulder_only_status


def lm(left_y=0.40, right_y=0.40, dx=0.0, hip_y=0.80, vis=1.0):
    pts = [("left_shoulder", 0.40 + dx, left_y), ("right_shoulder", 0.60 + dx, right_y),
           ("left_hip", 0.42, hip_y), ("right_hip", 0.58, hip_y)]
    return {n: {"x": x, "y": y, "visibility": vis} for n, x, y in pts}


def test_raised_shoulder_gives_positive_elevation():
    r = calculate_shoulder_elevation(lm(0.40, 0.40), lm(0.36, 0.36))
    assert r["left_shoulder_elevation_norm"] > 0 and r["right_shoulder_elevation_norm"] > 0


def test_lowered_shoulder_gives_negative_elevation():
    r = calculate_shoulder_elevation(lm(0.40, 0.40), lm(0.44, 0.44))
    assert r["left_shoulder_elevation_norm"] < 0


def test_vertical_displacement_up_is_positive():
    r = calculate_shoulder_vertical_displacement(lm(0.40, 0.40), lm(0.36, 0.36))
    assert r["left_vertical_displacement_norm"] > 0


def test_shoulder_only_within_range():
    assert "within provisional range" in shoulder_only_status(lm(), lm())[0]


def test_shoulder_only_elevation():
    assert "elevation" in shoulder_only_status(lm(), lm(0.35, 0.35))[0]


def test_shoulder_only_tilt():
    assert "tilt or shift" in shoulder_only_status(lm(), lm(0.35, 0.45))[0]


def test_shoulder_only_shift():
    assert "tilt or shift" in shoulder_only_status(lm(), lm(dx=0.05))[0]


def test_shoulder_only_not_visible():
    assert "not visible" in shoulder_only_status(lm(), lm(vis=0.1))[0]
    assert "not visible" in shoulder_only_status(None, lm())[0]
