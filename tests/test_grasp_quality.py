import pytest

from src.assessment.grasp_quality import assess_grasp_quality
from src.assessment.repetition_assessment import (
    create_repetition_assessment,
)


def obs(t, detected, grasp="cylindrical", reliable=True):
    return {
        "timestamp_ms": t,
        "grasp_detected": detected,
        "detected_grasp_type": grasp,
        "association_reliable": reliable,
    }


def test_stable_grasp():
    data = [obs(t, True) for t in range(0, 1000, 100)]
    result = assess_grasp_quality(data)

    assert result["status"] == "DESCRIPTIVE_ASSESSMENT"
    assert result["quality_score"] is None
    assert result["grasp_maintenance_percent"] == 100.0
    assert result["regrasp_count"] == 0


def test_regrasp_and_interruption():
    data = [
        obs(0, True),
        obs(100, True),
        obs(200, False),
        obs(300, False),
        obs(400, False),
        obs(500, True),
        obs(600, True),
    ]

    result = assess_grasp_quality(data)

    assert result["status"] == "DESCRIPTIVE_ASSESSMENT"
    assert result["regrasp_count"] == 1
    assert result["interruption_count"] == 1
    assert result["quality_score"] is None


def test_short_release_not_sustained_interruption():
    data = [
        obs(0, True),
        obs(100, False),
        obs(150, True),
        obs(200, True),
    ]

    result = assess_grasp_quality(data)

    assert result["regrasp_count"] == 1
    assert result["interruption_count"] == 0


def test_missing_data_prevents_score():
    data = [
        obs(0, True),
        obs(100, None),
        obs(200, None),
        obs(300, None),
        obs(400, True),
    ]

    result = assess_grasp_quality(data)

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["quality_score"] is None


def test_grasp_type_consistency():
    data = [
        obs(0, True, "cylindrical"),
        obs(100, True, "cylindrical"),
        obs(200, True, "tripod"),
        obs(300, True, "cylindrical"),
    ]

    result = assess_grasp_quality(data)

    assert result["dominant_grasp_type"] == "cylindrical"
    assert result["grasp_type_consistency_percent"] == 75.0


def test_unreliable_association():
    data = [
        obs(0, True),
        obs(100, True, reliable=False),
        obs(200, True),
        obs(300, True),
    ]

    result = assess_grasp_quality(data)

    assert result["coverage_percent"] == 75.0


def test_no_observations():
    result = assess_grasp_quality([])

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["quality_score"] is None


def test_repetition_assessment_integration():
    quality = assess_grasp_quality([
        obs(0, True),
        obs(100, True),
        obs(200, True),
    ])

    repetition = {
        "repetition_number": 1,
        "start_timestamp_ms": 0,
        "end_timestamp_ms": 200,
        "duration_ms": 200,
    }

    assessment = create_repetition_assessment(
        repetition,
        grasp_assessment=quality,
    )

    assert assessment["grasp_assessment"] == quality
    assert assessment["grasp_assessment"]["quality_score"] is None
