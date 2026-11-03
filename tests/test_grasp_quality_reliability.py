from src.assessment.grasp_quality import assess_grasp_quality


def observation(t, detected, grasp="cylindrical", reliable=True):
    return {
        "timestamp_ms": t,
        "grasp_detected": detected,
        "detected_grasp_type": grasp,
        "association_reliable": reliable,
    }


def test_ambiguous_grasp_cannot_receive_quality_score():
    rows = [
        observation(t, True, "AMBIGUOUS")
        for t in range(0, 1000, 100)
    ]
    result = assess_grasp_quality(rows)

    assert result["status"] == "PARTIAL_ASSESSMENT"
    assert result["quality_score"] is None
    assert result["grasp_maintenance_percent"] == 100.0
    assert result["grasp_type_consistency_percent"] is None


def test_confident_grasp_produces_descriptors_not_clinical_score():
    rows = [
        observation(t, True)
        for t in range(0, 1000, 100)
    ]
    result = assess_grasp_quality(rows)

    assert result["status"] == "DESCRIPTIVE_ASSESSMENT"
    assert result["grasp_type_consistency_percent"] == 100.0
    assert result["quality_score"] is None


def test_unreliable_object_association_blocks_presence():
    rows = [
        observation(t, True, reliable=False)
        for t in range(0, 1000, 100)
    ]
    result = assess_grasp_quality(
        rows, require_object_association=True
    )

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["coverage_percent"] == 0.0
    assert result["quality_score"] is None


def test_missing_observation_does_not_create_regrasp():
    rows = [
        observation(0, True),
        observation(100, None),
        observation(200, False),
        observation(300, True),
        observation(400, True),
    ]
    result = assess_grasp_quality(rows)

    assert result["regrasp_count"] == 0


def test_observed_release_and_regrasp():
    rows = [
        observation(0, True),
        observation(100, False),
        observation(200, False),
        observation(400, True),
        observation(500, True),
    ]
    result = assess_grasp_quality(rows)

    assert result["regrasp_count"] == 1
    assert result["interruption_count"] == 1


def test_missing_observations():
    result = assess_grasp_quality([])

    assert result["status"] == "INSUFFICIENT_DATA"
    assert result["quality_score"] is None
