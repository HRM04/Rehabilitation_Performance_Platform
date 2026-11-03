"""Minimal Streamlit interface for launching a camera assessment."""
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.platform_config.object_library import list_objects
from src.platform_config.session_config import SessionConfig, TRAJECTORY_PRESETS

st.set_page_config(page_title="Rehabilitation Assessment", layout="centered")
st.title("Rehabilitation Assessment")
st.write("Welcome. Follow the instructions to complete your movement assessment.")

objects = list_objects()
object_ids = list(objects)

with st.form("assessment_setup"):
    participant = st.text_input("Participant ID", value="demo_patient")
    object_id = st.selectbox(
        "Object",
        object_ids,
        format_func=lambda item: objects[item]["display_name"],
    )
    trajectory = st.selectbox(
        "Movement path",
        TRAJECTORY_PRESETS,
        format_func=lambda value: value.replace("_", " ").title(),
    )
    repetitions = st.number_input(
        "Number of repetitions", min_value=1, max_value=20, value=3
    )
    start = st.form_submit_button(
        "Start Assessment", use_container_width=True
    )

st.info(
    "When the camera opens, move your object to START, hold until "
    "the green indicator fills, and follow the on-screen instructions. "
    "At TARGET, hold until the green indicator fills again."
)

if start:
    participant = participant.strip()
    if not participant:
        st.error("Enter a participant ID before starting.")
        st.stop()

    config = SessionConfig(
        object_id=object_id,
        trajectory=trajectory,
        patient_id=participant,
        target_repetitions=int(repetitions),
    )
    config.validate()

    config_dir = ROOT / "data/session_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / f"{config.session_id}_session.json"
    if config_path.exists():
        st.error("Session ID already exists. Please try again.")
        st.stop()
    config_path.write_text(
        json.dumps(asdict(config), indent=2), encoding="utf-8"
    )

    # Launch the same camera and export pipeline as session_launcher.py.
    # The separate process keeps the Streamlit page responsive.
    worker = """
import subprocess
import sys
from src.assessment.export_unified_session import export_session

session_id = sys.argv[1]
result = subprocess.run(
    [sys.executable, "-m", "src.capture.camera_capture"],
    check=False,
)
if result.returncode == 0:
    export_session(session_id)
else:
    sys.exit(result.returncode)
"""

    env = os.environ.copy()
    env["REHAB_SESSION_CONFIG"] = str(config_path.resolve())

    log_dir = ROOT / "outputs/participant_interface_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{config.session_id}.log"

    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-c", worker, config.session_id],
            cwd=str(ROOT),
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    st.session_state["active_session"] = config.session_id
    st.session_state["active_pid"] = process.pid
    st.success("Assessment launched. Follow the camera window instructions.")

session_id = st.session_state.get("active_session")
if session_id:
    export_path = (
        ROOT / "data/processed/unified_sessions"
        / f"{session_id}_unified_assessment_v02.json"
    )
    st.write(f"**Session:** {session_id}")
    if export_path.exists():
        st.success("Session complete. Your assessment results are ready.")
    else:
        st.caption(
            "After the camera session closes, the assessment export "
            "may take a moment to finish."
        )

st.divider()
st.write("**View your results**")
st.code(
    "python -m streamlit run src/dashboard/patient_dashboard.py",
    language="bash",
)
