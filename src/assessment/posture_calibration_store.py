"""Persist session-specific posture calibration results."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_DIR = ROOT / "data" / "processed" / "posture_calibration"


def calibration_path(session_id):
    session_id = str(session_id)

    if (
        not session_id
        or session_id in (".", "..")
        or "/" in session_id
        or chr(92) in session_id
    ):
        raise ValueError("Invalid session ID")

    return CALIBRATION_DIR / f"{session_id}_posture_calibration.json"


def save_calibration(session_id, result):
    """Save calibration status and validated reference."""

    if not isinstance(result, dict):
        raise TypeError("Calibration result must be a dictionary")

    status = result.get("status", "UNAVAILABLE")

    reference = (
        result.get("reference_posture")
        if status == "READY"
        else None
    )

    if status == "READY" and not isinstance(reference, dict):
        raise ValueError("READY calibration requires a reference")

    record = {
        "session_id": str(session_id),
        "status": status,
        "reference_posture": reference,
        "calibration_details": {
            key: value
            for key, value in result.items()
            if key != "reference_posture"
        },
    }

    path = calibration_path(session_id)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("x", encoding="utf-8") as file:
        json.dump(record, file, indent=2, allow_nan=False)

    return path


def load_calibration(session_id):
    """Load a saved calibration, or return None if absent."""

    path = calibration_path(session_id)

    if not path.exists():
        return None

    record = json.loads(path.read_text(encoding="utf-8"))

    if str(record.get("session_id")) != str(session_id):
        raise ValueError("Calibration session ID mismatch")

    return record
