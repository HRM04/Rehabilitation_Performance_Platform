"""
Session-level analysis of trajectory rehabilitation game results.

Reuses the project's existing statistical helpers.
Does not modify game outcomes or infer clinical fatigue.
"""

import math
import statistics

from src.assessment.consistency_analysis import (
    coefficient_of_variation,
)
from src.assessment.performance_deterioration import (
    percent_change,
)
from src.assessment.trajectory_movement_quality import (
    assess_trajectory_movement_quality,
)

from src.assessment.trajectory_reliability import (
    assess_trajectory_reliability,
)


METRICS = {
    "duration_ms": "duration_ms",
    "movement_efficiency": "movement_efficiency",
    "mean_trajectory_deviation_px": "mean_trajectory_deviation_px",
    "final_target_error_px": "final_target_error_px",
    "average_speed_px_s": "average_speed_px_s",
    "speed_variability_px_s": "speed_variability_px_s",
}


def valid_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def analyse_trajectory_session(session):
    """Assess completed repetitions and summarize the session."""

    attempts = session.get("attempts", [])
    assessed_attempts = []

    for attempt in attempts:
        quality = assess_trajectory_movement_quality(
            attempt.get("trajectory_samples") or []
        )

        # Supplemental engineering reliability assessment.
        # Preserve existing raw metrics and legacy data-quality logic.
        config = attempt.get("trajectory_config") or {}

        reference_path = (
            config.get("desired_path")
            or config.get("reference_path")
        )

        if reference_path is None:
            reference_path = []

        reliability = assess_trajectory_reliability(
            attempt.get("trajectory_samples") or [],
            reference_path,
        )

        if len(reference_path) < 2:
            reliability["eligible"] = False
            if "Reference path unavailable" not in reliability["reasons"]:
                reliability["reasons"].append(
                    "Reference path unavailable"
                )

        trajectory = attempt.get("metrics") or {}

        values = {
            "duration_ms": attempt.get("duration_ms"),
            "movement_efficiency": trajectory.get(
                "movement_efficiency"
            ),
            "mean_trajectory_deviation_px": trajectory.get(
                "mean_trajectory_deviation_px"
            ),
            "final_target_error_px": trajectory.get(
                "final_target_error_px"
            ),
            "average_speed_px_s": quality.get(
                "average_speed_px_s"
            ),
            "speed_variability_px_s": quality.get(
                "speed_variability_px_s"
            ),
        }

        warnings = quality.get("quality_warnings", [])

        assessed_attempts.append({
            "attempt_number": attempt.get("attempt_number"),
            "repetition_number": attempt.get("repetition_number"),
            "outcome": attempt.get("outcome"),
            "metrics": values,
            "movement_quality": quality,
            "movement_reliability": reliability,
            "data_quality": (
                "REVIEW" if warnings else
                "SUFFICIENT" if quality["assessment_available"]
                else "UNAVAILABLE"
            ),
        })

    completed = [
        attempt for attempt in assessed_attempts
        if attempt["outcome"] == "COMPLETED"
    ]

    consistency = {}
    performance_change = {}

    for metric in METRICS:
        values = [
            attempt["metrics"].get(metric)
            for attempt in completed
        ]

        valid_values = [
            value for value in values
            if valid_number(value)
        ]

        consistency[metric] = {
            "mean": (
                statistics.mean(valid_values)
                if valid_values else None
            ),
            "cv": coefficient_of_variation(valid_values),
            "sample_count": len(valid_values),
        }

        first = completed[0]["metrics"].get(metric) if completed else None
        last = completed[-1]["metrics"].get(metric) if completed else None

        performance_change[metric] = {
            "first": first,
            "last": last,
            "change_percent": (
                percent_change(first, last)
                if valid_number(first) and valid_number(last)
                else None
            ),
        }

    return {
        "session_id": session.get("session_id"),
        "patient_id": session.get("patient_id"),
        "object_id": session.get("object_id"),
        "trajectory": session.get("trajectory"),
        "target_repetitions": session.get("target_repetitions"),
        "total_attempts": len(attempts),
        "completed_repetitions": len(completed),
        "interrupted_attempts": sum(
            attempt.get("outcome") == "INTERRUPTED"
            for attempt in attempts
        ),
        "other_unsuccessful_attempts": sum(
            attempt.get("outcome") not in ("COMPLETED", "INTERRUPTED")
            for attempt in attempts
        ),
        "attempt_assessments": assessed_attempts,
        "consistency": consistency,
        "observed_performance_change": performance_change,
        "interpretation_note": (
            "Observed performance change is not a direct "
            "measurement of physiological fatigue."
        ),
    }
