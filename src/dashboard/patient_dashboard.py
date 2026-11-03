"""Simple participant-facing assessment results dashboard."""
import json
from pathlib import Path
from datetime import datetime

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data/processed/unified_sessions"

st.set_page_config(page_title="My Rehabilitation Progress", layout="wide")
st.title("My Rehabilitation Progress")
st.caption("Your recorded movement assessment results")

sessions = []
for path in sorted(DATA_DIR.glob("*_unified_assessment_v02.json")):
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
        sessions.append(session)
    except (OSError, ValueError):
        continue

if not sessions:
    st.info("No completed assessments are available yet.")
    st.stop()

patients = sorted({
    str(s.get("patient_id"))
    for s in sessions if s.get("patient_id")
})
patient = st.selectbox("Participant ID", patients)
my_sessions = [
    s for s in sessions if str(s.get("patient_id")) == patient
]
my_sessions.sort(key=lambda s: str(s.get("session_id", "")))

selected = st.selectbox(
    "Assessment session",
    list(reversed(my_sessions)),
    format_func=lambda s: str(s.get("session_id", "")).replace("_", " "),
)
summary = selected.get("session_summary") or {}
task = selected.get("task") or {}
attempts = selected.get("attempts") or []

st.subheader("Session Overview")
a, b, c = st.columns(3)
a.metric("Completed repetitions", summary.get("completed_repetitions", 0))
b.metric("Total attempts", summary.get("total_attempts", len(attempts)))
c.metric("Target repetitions", task.get("target_repetitions", "—"))

st.write(
    f"**Object:** {str(task.get('object_id', '—')).title()}  |  "
    f"**Movement:** {str(task.get('trajectory', '—')).replace('_', ' ').title()}"
)

rows = []
for attempt in attempts:
    metrics = attempt.get("metrics") or {}
    duration = metrics.get("duration_ms")
    rows.append({
        "Repetition": attempt.get("repetition_number")
                      or attempt.get("attempt_number"),
        "Outcome": attempt.get("outcome", "Unknown"),
        "Time (seconds)": round(duration / 1000, 2)
                          if isinstance(duration, (int, float)) else None,
        "Movement efficiency": metrics.get("movement_efficiency"),
        "Path deviation (pixels)": metrics.get("mean_trajectory_deviation_px"),
        "Target error (pixels)": metrics.get("final_target_error_px"),
    })

st.subheader("Your Repetitions")
if rows:
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True)
    if df["Time (seconds)"].notna().any():
        st.markdown("**Time per repetition**")
        st.line_chart(df.set_index("Repetition")["Time (seconds)"])
else:
    st.info("No repetition measurements were recorded.")

st.subheader("Progress Across Sessions")
history = []
for s in my_sessions:
    attempts_s = s.get("attempts") or []
    durations = [
        a.get("metrics", {}).get("duration_ms")
        for a in attempts_s
        if isinstance(a.get("metrics", {}).get("duration_ms"), (int, float))
    ]
    history.append({
        "Session": str(s.get("session_id", "")),
        "Completed": (s.get("session_summary") or {}).get(
            "completed_repetitions", 0
        ),
        "Average time (seconds)": (
            round(sum(durations) / len(durations) / 1000, 2)
            if durations else None
        ),
    })

history_df = pd.DataFrame(history)
st.dataframe(history_df, hide_index=True, use_container_width=True)
if len(history_df) > 1:
    st.line_chart(
        history_df.set_index("Session")[
            ["Average time (seconds)"]
        ]
    )

st.caption(
    "These are prototype movement measurements, not clinical "
    "diagnoses or measures of recovery."
)
