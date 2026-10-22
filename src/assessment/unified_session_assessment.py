"""
Unified session assessment for the rehabilitation platform.

Combines existing trajectory and pose assessment outputs.
Does not alter original recordings or infer clinical diagnoses.
"""

from src.assessment.trajectory_session_analysis import (
    analyse_trajectory_session,
)
from src.assessment.trajectory_pose_adapter import (
    assess_pose_for_attempts,
)
from src.assessment.joint_quality_control import (
    assess_joint_quality,
)


def build_unified_session_assessment(session, pose_frames):
    trajectory = analyse_trajectory_session(session)
    pose_results = assess_pose_for_attempts(session, pose_frames)

    # Preserve attempt order rather than relying on attempt numbers
    # being unique or always present.
    combined_attempts = []

    for index, trajectory_attempt in enumerate(
        trajectory["attempt_assessments"]
    ):
        pose = pose_results[index] if index < len(pose_results) else {}

        pose_quality = pose.get("pose_quality", {})
        movement = pose.get("movement_quality")

        warnings = []

        trajectory_quality = trajectory_attempt.get(
            "movement_quality", {}
        )
        warnings.extend(
            trajectory_quality.get("quality_warnings", [])
        )

        if movement:
            for joint, coverage in pose_quality.get(
                "joint_angle_coverage", {}
            ).items():
                if coverage["coverage_percent"] < 80:
                    warnings.append(
                        f"{joint}: insufficient angle coverage "
                        f"({coverage['coverage_percent']}%)"
                    )

            if pose_quality.get("valid_trunk_frames", 0) == 0:
                warnings.append(
                    "Trunk displacement unavailable: "
                    "insufficient reliable landmarks"
                )
        else:
            warnings.append("Pose movement assessment unavailable")

        # Apply joint-quality screening without altering raw measurements.
        original_attempt = session["attempts"][index]
        start = original_attempt.get("start_timestamp_ms")
        end = original_attempt.get("end_timestamp_ms")

        matching_frames = (
            [
                frame for frame in pose_frames
                if start <= frame["timestamp_ms"] <= end
            ]
            if start is not None and end is not None and end >= start
            else []
        )

        joint_quality = {}

        screened_movement = (
            {
                **movement,
                "joints": {
                    name: dict(metrics)
                    for name, metrics in movement["joints"].items()
                },
            }
            if movement else None
        )

        for joint in (
            "left_elbow",
            "right_elbow",
            "left_shoulder",
            "right_shoulder",
        ):
            quality = assess_joint_quality(matching_frames, joint)
            joint_quality[joint] = quality

            if not quality["clinician_display_allowed"]:
                if screened_movement and joint in screened_movement["joints"]:
                    screened_movement["joints"][joint] = {
                        key: None
                        for key in screened_movement["joints"][joint]
                    }

                warnings.append(
                    f"{joint}: measurements withheld "
                    f"({'; '.join(quality['warnings'])})"
                )

        combined_attempts.append({
            **trajectory_attempt,
            "active_hand": "UNKNOWN",
            "hand_switch_detected": None,
            "pose_quality": pose_quality,
            "pose_movement_quality_raw": movement,
            "joint_quality_control": joint_quality,
            "pose_movement_quality_screened": screened_movement,
            "quality_warnings": warnings,
        })

    return {
        "schema_version": "0.1",
        "session_id": session.get("session_id"),
        "patient_id": session.get("patient_id"),
        "task": {
            "object_id": session.get("object_id"),
            "trajectory": session.get("trajectory"),
            "target_repetitions": session.get("target_repetitions"),
        },
        "session_summary": {
            "total_attempts": trajectory["total_attempts"],
            "completed_repetitions": trajectory["completed_repetitions"],
            "interrupted_attempts": trajectory["interrupted_attempts"],
            "other_unsuccessful_attempts": trajectory[
                "other_unsuccessful_attempts"
            ],
        },
        "attempts": combined_attempts,
        "consistency": trajectory["consistency"],
        "observed_performance_change": trajectory[
            "observed_performance_change"
        ],
        "clinical_interpretation": {
            "status": "PROTOTYPE_MEASUREMENTS",
            "notes": [
                "Measurements have not been clinically validated.",
                "Active hand was not recorded for this session.",
                "Raw joint measurements require quality review.",
                "Observed performance changes do not establish fatigue.",
                "Unavailable measurements must not be interpreted as zero.",
            ],
        },
    }
