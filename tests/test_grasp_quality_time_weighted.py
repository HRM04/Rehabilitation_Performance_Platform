from src.assessment.grasp_quality import assess_grasp_quality


def _obs(t, detected, kind=None):
    return {"timestamp_ms": t, "grasp_detected": detected, "detected_grasp_type": kind or ""}


def test_time_weighting_differs_from_counts():
    obs = [_obs(0, True, "cylindrical"), _obs(400, False), _obs(500, False), _obs(600, False)]
    r = assess_grasp_quality(obs)
    assert r["grasp_maintenance_percent"] == 25.0
    assert r["time_weighted"]["grasp_maintenance_percent"] == 57.1


def test_gap_is_capped():
    obs = [_obs(0, True, "cylindrical"), _obs(100, True, "cylindrical"), _obs(5000, True, "cylindrical")]
    r = assess_grasp_quality(obs)
    assert r["time_weighted"]["observed_time_ms"] == 1100


def test_empty_input():
    r = assess_grasp_quality([])
    assert r["time_weighted"]["observed_time_ms"] == 0
    assert r["time_weighted"]["coverage_percent"] is None
    assert r["quality_score"] is None
