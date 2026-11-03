"""Launch the rehabilitation showcase components."""

import subprocess
import sys

OPTIONS = {
    "1": (
        "Camera / Rehabilitation Task",
        ["-m", "src.capture.camera_capture"],
    ),
    "2": (
        "Clinician Session Dashboard",
        ["-m", "streamlit", "run",
         "src/dashboard/clinician_session.py"],
    ),
    "3": (
        "Clinician Longitudinal Dashboard",
        ["-m", "streamlit", "run",
         "src/dashboard/clinician_longitudinal.py"],
    ),
}

def main():
    print("\\nREHABILITATION SHOWCASE")
    print("=" * 35)

    for key, (label, _) in OPTIONS.items():
        print(f"{key}. {label}")

    print("4. Open both dashboards")
    choice = input("\\nChoose: ").strip()

    if choice == "4":
        for key in ("2", "3"):
            label, args = OPTIONS[key]
            print(f"Starting {label}...")
            subprocess.Popen(
                [sys.executable, *args],
                cwd=".",
            )
        return

    if choice not in OPTIONS:
        print("Invalid selection.")
        return

    label, args = OPTIONS[choice]
    print(f"Starting {label}...")
    subprocess.run(
        [sys.executable, *args],
        cwd=".",
        check=False,
    )

if __name__ == "__main__":
    main()
