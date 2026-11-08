# Rehabilitation Performance Platform

Camera-based upper-limb rehabilitation assessment (Ontario Tech, MEng Mechatronics).
Current build: branch `main`.

## Get the code

```
git clone https://github.com/HRM04/Rehabilitation_Performance_Platform.git
cd Rehabilitation_Performance_Platform
```

The repository is private. Access must be granted on GitHub first.

## Install (Python 3.12 required)

```
python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The install is large (PyTorch). Notes:

- Use Python 3.12. The project declares `>=3.12`, and the pinned packages were tested on 3.12.
- `mediapipe` is pinned to 0.10.13 because the code uses `mp.solutions`. Do not upgrade it.
- `streamlit` is 1.60.0 because 1.66.1 needs a newer protobuf than mediapipe allows.
- The YOLO weights (`yolo11m.pt`, about 40 MB) download automatically the first time the object detector runs, so the first run needs internet.
- macOS asks for camera permission the first time. Allow it for the terminal or IDE.

## Run

```
python showcase_launcher.py                                  # menu: task and the two clinician dashboards
python -m src.capture.camera_capture                         # camera task
python -m streamlit run src/dashboard/clinician_session.py   # clinician session dashboard
python -m streamlit run src/dashboard/clinician_longitudinal.py
python -m streamlit run src/dashboard/patient_dashboard.py     # participant-facing dashboard
```

## Where recordings go

Nothing needs to be copied in. The first run creates the folders it writes to:

- `data/raw/video`, `data/raw/landmarks`, `data/raw/interactions`, `data/raw/objects`, `data/raw/pose`: one set of files per recorded session
- `data/trajectory_results`: per-attempt results
- `data/processed/unified_sessions`: the assessment file the dashboards read
- `data/session_configs`: the task chosen for each session

The dashboards read from `data/`, so they show data only after at least one session has been recorded.

`data/` is not tracked by git. Recordings stay on the machine that made them.
