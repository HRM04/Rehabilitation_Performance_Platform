from src.assessment.recorded_trajectory import load_recorded_trajectory

"""Process recorded rehabilitation sessions with available modalities."""

import csv
import json
from collections import OrderedDict
from pathlib import Path

from src.assessment.session_pipeline import process_session


def read_csv(path):
    with Path(path).open(newline="") as file:
        return list(csv.DictReader(file))


def numeric(value):
    if value is None or str(value).strip() == "":
        return None
    return float(value)


def load_pose_frames(path):
    """Convert landmark-per-row CSV into one dictionary per frame."""

    rows = read_csv(path)
    frames = OrderedDict()

    for row in rows:
        timestamp = int(float(row["timestamp_ms"]))
        frame_number = int(row["frame_number"])
        key = (frame_number, timestamp)

        if key not in frames:
            frames[key] = {
                "timestamp_ms": timestamp,
                "landmarks": {},
                "angles": {},
            }

        frame = frames[key]
        name = (row.get("landmark_name") or "").strip()

        x = numeric(row.get("x_norm"))
        y = numeric(row.get("y_norm"))

        if name and x is not None and y is not None:
            frame["landmarks"][name] = {
                "x": x,
                "y": y,
                "z": numeric(row.get("z_rel")),
                "visibility": numeric(row.get("visibility")),
            }

        for column in (
            "left_elbow_angle",
            "right_elbow_angle",
            "left_shoulder_angle",
            "right_shoulder_angle",
        ):
            value = numeric(row.get(column))
            if value is not None:
                frame["angles"][column.removesuffix("_angle")] = value

    return sorted(
        frames.values(),
        key=lambda frame: frame["timestamp_ms"],
    )


def build_grasp_observations(rows):
    observations = []

    for row in rows:
        object_id = str(row.get("object_id") or "").strip()
        handedness = str(row.get("handedness") or "").strip()

        if not object_id or not handedness:
            continue

        observations.append({
            "timestamp_ms": int(float(row["timestamp_ms"])),
            "object_id": object_id,
            "handedness": handedness,
            "grasp_detected": row.get("grasp_detected"),
            "detected_grasp_type": row.get("detected_grasp_type"),
            "association_reliable": (
                True
                if str(row.get("association_status") or "").strip().upper()
                == "ASSOCIATED"
                else False
                if str(row.get("association_status") or "").strip().upper()
                in {"NONE", "AMBIGUOUS"}
                else None
            ),
        })

    return observations


def main():
    files = sorted(
        Path("data/raw/interactions").glob("*_interaction_data.csv")
    )

    if not files:
        raise SystemExit("No interaction recordings found.")

    interaction_path = files[-1]
    session_id = interaction_path.name.replace(
        "_interaction_data.csv", ""
    )

    rows = read_csv(interaction_path)

    pose_path = Path(
        f"data/raw/pose/{session_id}_pose_landmarks.csv"
    )

    pose_frames = (
        load_pose_frames(pose_path)
        if pose_path.exists()
        else None
    )

    calibration_path = Path(
        f"data/processed/posture_calibration/"
        f"{session_id}_posture_calibration.json"
    )

    calibration = None
    reference_posture = None

    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())

        if calibration.get("status") == "READY":
            reference_posture = calibration.get("reference_posture")

    task_id = next(
        (row.get("task_id") for row in rows if row.get("task_id")),
        "unknown_task",
    )

    trajectory_assessment, trajectory_path = load_recorded_trajectory(session_id)

    result = process_session(
        session_id=session_id,
        task_id=task_id,
        interaction_rows=rows,
        pose_frames=pose_frames,
        trajectory_assessment=trajectory_assessment,
        reference_posture=reference_posture,
        grasp_observations=build_grasp_observations(rows),
    )

    
    # Preserve complete trajectory attempts independently of
    # interaction repetition assessments.
    if trajectory_assessment is not None:
        attempts = trajectory_assessment.complete_attempt_results

        result["trajectory_attempts"] = attempts

        result["trajectory_task_summary"] = {
            "total_attempts": len(attempts),
            "completed_attempts": sum(
                a.get("outcome") == "COMPLETED"
                for a in attempts
            ),
            "target_reached_attempts": sum(
                (a.get("metrics") or {}).get("target_reached") is True
                for a in attempts
            ),
        }

        repetitions = (
            result.get("session_assessment") or {}
        ).get("repetition_assessments", [])

        alignment = []

        for rep in repetitions:
            timing = rep.get("timing") or {}
            rep_start = timing.get("start_timestamp_ms")
            rep_end = timing.get("end_timestamp_ms")

            if rep_start is None or rep_end is None:
                continue

            for attempt in attempts:
                attempt_start = attempt.get("start_timestamp_ms")
                attempt_end = attempt.get("end_timestamp_ms")

                if attempt_start is None or attempt_end is None:
                    continue

                overlap_ms = max(
                    0,
                    min(rep_end, attempt_end)
                    - max(rep_start, attempt_start),
                )

                if overlap_ms == 0:
                    continue

                alignment.append({
                    "repetition_number": rep.get("repetition_number"),
                    "attempt_number": attempt.get("attempt_number"),
                    "overlap_ms": overlap_ms,
                    "repetition_start_ms": rep_start,
                    "repetition_end_ms": rep_end,
                    "attempt_start_ms": attempt_start,
                    "attempt_end_ms": attempt_end,
                    "end_offset_ms": attempt_end - rep_end,
                    "windows_identical": (
                        rep_start == attempt_start
                        and rep_end == attempt_end
                    ),
                    "interpretation": (
                        "DIFFERENT_TIME_WINDOWS"
                        if rep_start != attempt_start
                        or rep_end != attempt_end
                        else "ALIGNED"
                    ),
                })

        result["trajectory_alignment"] = alignment

    result["recording_provenance"] = {
        "interaction_csv": str(interaction_path),
        "pose_csv": str(pose_path) if pose_frames is not None else None,
        "pose_frame_count": len(pose_frames) if pose_frames else 0,
        "posture_calibration": (
            str(calibration_path) if calibration is not None else None
        ),
        "calibration_mode": (
            (calibration.get("reference_posture") or {})
            .get("calibration_mode")
            if calibration else None
        ),
        "grasp_association": "NOT_INDEPENDENTLY_VALIDATED",
        "trajectory_assessment_loaded": trajectory_assessment is not None,
        "trajectory_results_json": str(trajectory_path) if trajectory_path else None,
        "clinical_validation": False,
    }

    output_dir = Path("data/processed/session_assessments")
    output_dir.mkdir(parents=True, exist_ok=True)

    output = output_dir / f"{session_id}_assessment.json"

    with output.open("w") as file:
        json.dump(result, file, indent=2, default=str)

    assessments = result["session_assessment"].get(
        "repetition_assessments", []
    )

    print("Recording:", interaction_path)
    print("Pose frames:", len(pose_frames) if pose_frames else 0)
    print("Calibration:", (
        result["recording_provenance"]["calibration_mode"]
    ))
    print("Repetitions assessed:", len(assessments))

    for rep in assessments:
        movement = rep.get("movement_quality") or {}
        print(
            "Repetition:", rep.get("repetition_number"),
            "| Movement:", "AVAILABLE" if movement else "UNAVAILABLE",
            "| Grasp:", (rep.get("grasp_quality") or {}).get("status"),
            "| Compensation:", rep.get("compensation"),
        )

    print("Saved:", output)


if __name__ == "__main__":
    main()
