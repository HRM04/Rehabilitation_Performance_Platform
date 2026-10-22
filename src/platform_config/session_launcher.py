"""
Interactive launcher for the unified rehabilitation platform.
"""

import json
import os
import subprocess
import sys

from dataclasses import asdict
from pathlib import Path

from src.platform_config.object_library import list_objects
from src.platform_config.session_config import (
    SessionConfig,
    TRAJECTORY_PRESETS,
)


def choose_option(title, options):
    print(f"\n{title}")
    print("-" * 50)

    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    while True:
        choice = input("\nSelect a number: ").strip()

        if choice.isdigit():
            index = int(choice) - 1

            if 0 <= index < len(options):
                return index

        print("Invalid selection. Please try again.")


def main():
    print("\n" + "=" * 60)
    print("UNIFIED REHABILITATION PLATFORM")
    print("=" * 60)

    objects = list_objects()
    object_ids = list(objects)

    object_labels = [
        objects[object_id]["display_name"]
        for object_id in object_ids
    ]

    object_index = choose_option(
        "SELECT REHABILITATION OBJECT",
        object_labels,
    )

    object_id = object_ids[object_index]

    trajectory_index = choose_option(
        "SELECT MOVEMENT TRAJECTORY",
        TRAJECTORY_PRESETS,
    )

    trajectory = TRAJECTORY_PRESETS[trajectory_index]

    patient_id = input(
        "\nPatient ID [demo_patient]: "
    ).strip() or "demo_patient"

    repetitions_input = input(
        "Target repetitions [3]: "
    ).strip()

    try:
        target_repetitions = (
            int(repetitions_input)
            if repetitions_input
            else 3
        )
    except ValueError:
        print("Invalid repetition count.")
        return

    config = SessionConfig(
        object_id=object_id,
        trajectory=trajectory,
        patient_id=patient_id,
        target_repetitions=target_repetitions,
    )

    config.validate()

    root = Path(__file__).resolve().parents[2]

    config_dir = root / "data" / "session_configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    config_path = (
        config_dir / f"{config.session_id}_session.json"
    )

    config_path.write_text(
        json.dumps(asdict(config), indent=2)
    )

    print("\n" + "=" * 60)
    print("SESSION READY")
    print("=" * 60)

    for key, value in config.get_task_details().items():
        print(f"{key}: {value}")

    print("\nCamera controls:")
    print("1–5: Change trajectory between attempts")
    print("Q: Stop session")

    confirmation = input(
        "\nStart camera session? [y/N]: "
    ).strip().lower()

    if confirmation != "y":
        print("Session configuration saved.")
        return

    env = os.environ.copy()
    env["REHAB_SESSION_CONFIG"] = str(config_path)

    print("\nStarting rehabilitation session...")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.capture.camera_capture",
        ],
        cwd=root,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"Camera exited with code {result.returncode}. "
            "Automatic assessment export skipped."
        )
        return

    print("\nGenerating clinician assessment...")

    try:
        from src.assessment.export_unified_session import export_session

        output_path = export_session(config.session_id)
    except Exception as error:
        print(f"Assessment export unavailable: {error}")
        print("Original session recordings remain unchanged.")
    else:
        print(f"Assessment ready: {output_path}")


if __name__ == "__main__":
    main()
