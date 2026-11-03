"""
Interactive launcher for the unified rehabilitation platform.
"""

import json
import os
import subprocess
import shutil
import csv
import re
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



def archive_volunteer_session(root, config, config_path, started_at):
    """Copy original session files into a volunteer-specific archive."""
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "_", config.patient_id)
    if not safe_id or safe_id in (".", ".."):
        raise ValueError("Invalid participant identifier.")

    destination = (
        root / "data" / "volunteer_testing"
        / safe_id
        / f"session_{config.volunteer_session}"
        / f"run_{config.session_id}"
    )

    sources = {
        "video": root / "data/raw/video",
        "landmarks": root / "data/raw/landmarks",
        "interactions": root / "data/raw/interactions",
        "objects": root / "data/raw/objects",
        "pose": root / "data/raw/pose",
        "trajectory_results": root / "data/trajectory_results",
    }

    copied = []

    for category, directory in sources.items():
        if not directory.exists():
            continue

        for source in directory.iterdir():
            if not source.is_file():
                continue

            # Session-ID files match exactly; camera files use
            # a separate timestamp generated after launch.
            matches_session = source.name.startswith(config.session_id + "_")
            matches_recording = (
                category in ("video", "landmarks", "interactions")
                and source.stat().st_mtime >= started_at
            )

            if not (matches_session or matches_recording):
                continue

            # Interaction CSVs contain the actual session ID.
            if category == "interactions":
                try:
                    with source.open(newline="") as f:
                        reader = csv.DictReader(f)
                        first = next(reader, {})
                    if first.get("session_id") != config.session_id:
                        continue
                except (OSError, UnicodeError, csv.Error):
                    continue

            target = destination / category / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(str(target.relative_to(destination)))

    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, destination / "session_config.json")

    manifest = {
        "participant_id": config.patient_id,
        "volunteer_session": config.volunteer_session,
        "session_id": config.session_id,
        "files": copied,
    }

    (destination / "archive_manifest.json").write_text(
        json.dumps(manifest, indent=2)
    )

    print(f"Volunteer archive: {destination}")
    print(f"Archived {len(copied)} data files.")

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

    purpose_index = choose_option(
        "SESSION PURPOSE",
        ["Regular assessment", "Volunteer testing"],
    )
    session_purpose = "testing" if purpose_index == 1 else "regular"

    volunteer_session = 1
    if session_purpose == "testing":
        while True:
            value = input("Volunteer session number [1]: ").strip() or "1"
            if value.isdigit() and int(value) >= 1:
                volunteer_session = int(value)
                break
            print("Please enter a positive session number.")

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

    print("\nOPTIONAL ASSESSMENT FEATURES")
    print("-" * 50)

    trajectory_enabled = (
        input("Enable trajectory game? [Y/n]: ")
        .strip().lower() not in ("n", "no")
    )

    orientation_enabled = (
        input("Enable experimental orientation assessment? [y/N]: ")
        .strip().lower() in ("y", "yes")
    )

    segmentation_enabled = (
        input("Enable experimental segmentation? [y/N]: ")
        .strip().lower() in ("y", "yes")
    )

    if orientation_enabled:
        segmentation_enabled = True
        print(
            "Segmentation automatically enabled "
            "for orientation assessment."
        )

    config = SessionConfig(
        object_id=object_id,
        trajectory=trajectory,
        patient_id=patient_id,
        session_purpose=session_purpose,
        volunteer_session=volunteer_session,
        target_repetitions=target_repetitions,
        trajectory_enabled=trajectory_enabled,
        orientation_enabled=orientation_enabled,
        segmentation_enabled=segmentation_enabled,
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

    started_at = __import__("time").time()

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

    if config.session_purpose == "testing":
        try:
            archive_volunteer_session(
                root, config, config_path, started_at
            )
        except Exception as error:
            print(f"Volunteer archive failed: {error}")
            print("Original recordings remain unchanged.")

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
