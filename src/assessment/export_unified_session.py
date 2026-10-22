"""Export an existing session for the clinician dashboard."""

import json
import sys
from pathlib import Path

from src.assessment.unified_session_assessment import (
    build_unified_session_assessment,
)
from src.assessment.trajectory_pose_adapter import (
    load_pose_frames,
)

ROOT = Path(__file__).resolve().parents[2]


def export_session(session_id):
    trajectory_path = (
        ROOT / "data" / "trajectory_results"
        / f"{session_id}_trajectory_results.json"
    )
    pose_path = (
        ROOT / "data" / "raw" / "pose"
        / f"{session_id}_pose_landmarks.csv"
    )
    output_path = (
        ROOT / "data" / "processed" / "unified_sessions"
        / f"{session_id}_unified_assessment_v02.json"
    )

    if not trajectory_path.is_file():
        raise FileNotFoundError(trajectory_path)
    if not pose_path.is_file():
        raise FileNotFoundError(pose_path)
    if output_path.exists():
        raise FileExistsError(output_path)

    session = json.loads(
        trajectory_path.read_text(encoding="utf-8")
    )

    if str(session.get("session_id")) != str(session_id):
        raise ValueError("Trajectory session ID mismatch")

    pose_frames = load_pose_frames(pose_path)

    assessment = build_unified_session_assessment(
        session, pose_frames
    )

    required = {
        "session_id", "task", "session_summary",
        "attempts", "clinical_interpretation",
    }
    if not required.issubset(assessment):
        raise ValueError("Unified assessment is incomplete")

    if any(
        "joint_quality_control" not in attempt
        or "pose_movement_quality_screened" not in attempt
        for attempt in assessment["attempts"]
    ):
        raise ValueError(
            "Version 0.2 joint-quality fields are missing"
        )

    assessment["schema_version"] = "0.2"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("x", encoding="utf-8") as file:
        json.dump(assessment, file, indent=2, allow_nan=False)

    return output_path


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 -m "
            "src.assessment.export_unified_session SESSION_ID"
        )

    result = export_session(sys.argv[1])
    print(f"Assessment exported: {result}")
