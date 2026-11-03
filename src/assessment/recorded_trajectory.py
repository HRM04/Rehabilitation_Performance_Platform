"""Replay saved trajectory recordings using the live metric calculator."""

import json
from pathlib import Path

from src.assessment.trajectory_assessment import TrajectoryAssessment


def load_recorded_trajectory(session_id):
    path = (
        Path("data/trajectory_results")
        / f"{session_id}_trajectory_results.json"
    )

    if not path.exists():
        return None, None

    data = json.loads(path.read_text())

    if data.get("session_id") != session_id:
        raise ValueError("Trajectory session ID mismatch")

    attempts = data.get("attempts") or []

    if not attempts:
        return None, path

    config = data.get("trajectory_config") or {}

    start_point = config.get("start_point")
    target_center = config.get("target_center")
    target_radius = config.get("target_radius", 50)
    waypoints = config.get("waypoints")

    if start_point is None or target_center is None:
        raise ValueError(
            "Recorded trajectory geometry is incomplete"
        )

    class RecordedTrajectory(TrajectoryAssessment):
        def calculate_metrics_for_window(
            self,
            start_timestamp_ms,
            end_timestamp_ms,
            object_id=None,
        ):
            # Normalize object IDs only for recorded-session replay.
            # Preserve the original trajectory samples and geometry.
            if object_id is not None:
                matching_ids = {
                    sample.get("object_id")
                    for sample in self.trajectory_samples
                    if str(sample.get("object_id")) == str(object_id)
                }

                if len(matching_ids) == 1:
                    object_id = next(iter(matching_ids))

            return super().calculate_metrics_for_window(
                start_timestamp_ms=start_timestamp_ms,
                end_timestamp_ms=end_timestamp_ms,
                object_id=object_id,
            )


        def __init__(self):
            super().__init__(
                start_point=start_point,
                target_center=target_center,
                target_radius=target_radius,
                waypoints=waypoints,
            )

            self.attempts = attempts

            self.trajectory_samples = [
                sample
                for attempt in attempts
                for sample in attempt.get(
                    "trajectory_samples", []
                )
                if sample.get("timestamp_ms") is not None
                and sample.get("center") is not None
            ]

            self.complete_attempt_results = [
                {
                    "attempt_number": attempt.get(
                        "attempt_number"
                    ),
                    "outcome": attempt.get("outcome"),
                    "start_timestamp_ms": attempt.get(
                        "start_timestamp_ms"
                    ),
                    "end_timestamp_ms": attempt.get(
                        "end_timestamp_ms"
                    ),
                    "metrics": attempt.get("metrics"),
                }
                for attempt in attempts
            ]

    return RecordedTrajectory(), path
