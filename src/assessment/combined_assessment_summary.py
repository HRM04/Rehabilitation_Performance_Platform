"""Combine trajectory results with recorded grasp observations."""

import csv
import json
import math
from collections import Counter
from pathlib import Path


TRAJECTORY_DIR = Path("data/trajectory_results")
INTERACTION_DIR = Path("data/raw/interactions")


def summarize_grasps(rows):
    """Summarize frame-level hand observations without clinical scoring."""
    hand_rows = [
        row for row in rows
        if (row.get("handedness") or "").strip()
    ]

    def summarize(group):
        grasp_types = Counter(
            row["detected_grasp_type"]
            for row in group
            if row.get("detected_grasp_type")
        )
        appropriateness = Counter(
            row["grasp_appropriateness"]
            for row in group
            if row.get("grasp_appropriateness")
        )
        events = Counter(
            row["interaction_event"]
            for row in group
            if row.get("interaction_event")
        )
        return {
            "hand_observations": len(group),
            "grasp_type_counts": dict(grasp_types),
            "appropriateness_counts": dict(appropriateness),
            "interaction_event_counts": dict(events),
        }

    by_hand = {}
    for hand in sorted({
        row["handedness"].strip() for row in hand_rows
    }):
        by_hand[hand] = summarize([
            row for row in hand_rows
            if row["handedness"].strip() == hand
        ])

    return {
        **summarize(hand_rows),
        "by_hand": by_hand,
    }



def valid_timestamp(value):
    """Return a finite, nonnegative timestamp or None."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def rows_in_attempt(rows, start, end):
    """Select observations within a valid attempt window."""
    start = valid_timestamp(start)
    end = valid_timestamp(end)

    if start is None or end is None or end < start:
        return None

    selected = []
    for row in rows:
        timestamp = valid_timestamp(row.get("timestamp_ms"))
        if timestamp is not None and start <= timestamp <= end:
            selected.append(row)

    return selected


def load_combined_session(session_id):
    """Load and summarize one recorded session."""
    trajectory_file = (
        TRAJECTORY_DIR / f"{session_id}_trajectory_results.json"
    )
    trajectory = json.loads(
        trajectory_file.read_text(encoding="utf-8")
    )

    matching = []
    for path in sorted(INTERACTION_DIR.glob("*_interaction_data.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        if first and first.get("session_id") == session_id:
            matching.append(path)

    if len(matching) != 1:
        raise ValueError(
            f"Expected one interaction CSV for {session_id}, "
            f"found {len(matching)}"
        )

    with matching[0].open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    attempts = []
    for attempt in trajectory.get("attempts", []):
        start = attempt.get("start_timestamp_ms")
        end = attempt.get("end_timestamp_ms")

        attempt_rows = rows_in_attempt(rows, start, end)

        if attempt_rows is None:
            grasp_summary = None
            alignment_status = "TIMESTAMPS_UNAVAILABLE"
        else:
            grasp_summary = summarize_grasps(attempt_rows)
            alignment_status = (
                "OBSERVATIONS_FOUND"
                if attempt_rows else "NO_OBSERVATIONS"
            )

        attempts.append({
            "attempt_number": attempt.get("attempt_number"),
            "repetition_number": attempt.get("repetition_number"),
            "outcome": attempt.get("outcome"),
            "start_timestamp_ms": start,
            "end_timestamp_ms": end,
            "movement_metrics": attempt.get("metrics", {}),
            "grasp_alignment_status": alignment_status,
            "grasp_summary": grasp_summary,
        })

    return {
        "session_id": session_id,
        "patient_id": trajectory.get("patient_id"),
        "object_id": trajectory.get("object_id"),
        "trajectory": trajectory.get("trajectory"),
        "interaction_file": matching[0].name,
        "session_grasp_summary": summarize_grasps(rows),
        "attempts": attempts,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python3 -m "
            "src.assessment.combined_assessment_summary SESSION_ID"
        )

    result = load_combined_session(sys.argv[1])
    print(json.dumps(result, indent=2))
