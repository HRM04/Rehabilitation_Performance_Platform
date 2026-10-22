"""Read-only clinician session dashboard for rehabilitation assessments."""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.assessment.combined_assessment_summary import load_combined_session
from src.dashboard.trajectory_visualization import (
    render_trajectory_visualization,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/unified_sessions"

st.set_page_config(
    page_title="Clinician Session Dashboard",
    page_icon="📊",
    layout="wide",
)


def load_session(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_metric(attempt, name):
    """Read a metric from the unified trajectory assessment."""
    metrics = attempt.get("metrics") or {}
    return metrics.get(name)


st.title("Clinician Session Dashboard")
st.caption("Upper-limb rehabilitation | Prototype assessment interface")

session_files = sorted(
    DATA_DIR.glob("*_unified_assessment_v02.json"),
    reverse=True,
)

if not session_files:
    st.warning("No version 0.2 unified session assessments were found.")
    st.stop()

selected_file = st.selectbox(
    "Select rehabilitation session",
    session_files,
    format_func=lambda path: path.name.replace(
        "_unified_assessment_v02.json", ""
    ),
)

session = load_session(selected_file)
attempts = session.get("attempts", [])
summary = session.get("session_summary", {})
task = session.get("task", {})

st.subheader("Session Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Completed Repetitions", summary.get("completed_repetitions", 0))
col2.metric("Total Attempts", summary.get("total_attempts", len(attempts)))
col3.metric("Interrupted Attempts", summary.get("interrupted_attempts", 0))
col4.metric("Target Repetitions", task.get("target_repetitions", "N/A"))

st.write(
    f"**Patient:** {session.get('patient_id', 'Unknown')}  |  "
    f"**Object:** {task.get('object_id', 'Unknown')}  |  "
    f"**Trajectory:** {task.get('trajectory', 'Unknown')}"
)

st.divider()

# -------------------------------------------------------
# Repetition performance
# -------------------------------------------------------

st.subheader("Repetition Performance")

rows = []

for attempt in attempts:
    rows.append({
        "Attempt": attempt.get("attempt_number"),
        "Outcome": attempt.get("outcome"),
        "Duration (s)": (
            get_metric(attempt, "duration_ms") / 1000
            if get_metric(attempt, "duration_ms") is not None
            else None
        ),
        "Movement Efficiency": get_metric(
            attempt, "movement_efficiency"
        ),
        "Mean Deviation (px)": get_metric(
            attempt, "mean_trajectory_deviation_px"
        ),
        "Final Target Error (px)": get_metric(
            attempt, "final_target_error_px"
        ),
        "Average Speed (px/s)": get_metric(
            attempt, "average_speed_px_s"
        ),
    })

performance_df = pd.DataFrame(rows)

st.dataframe(
    performance_df,
    use_container_width=True,
    hide_index=True,
)

if not performance_df.empty:
    completed_df = performance_df[
        performance_df["Outcome"] == "COMPLETED"
    ].copy()

    if not completed_df.empty:
        chart1, chart2 = st.columns(2)

        with chart1:
            fig = px.bar(
                completed_df,
                x="Attempt",
                y="Duration (s)",
                title="Movement Duration by Attempt",
            )
            st.plotly_chart(fig, use_container_width=True)

        with chart2:
            fig = px.line(
                completed_df,
                x="Attempt",
                y="Movement Efficiency",
                markers=True,
                title="Movement Efficiency",
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()

# -------------------------------------------------------
# Interactive trajectory review
# -------------------------------------------------------

st.subheader("Trajectory Review")

if attempts:
    attempt_indices = list(range(len(attempts)))

    selected_index = st.selectbox(
        "Select attempt for trajectory review",
        attempt_indices,
        format_func=lambda index: (
            f"Attempt {attempts[index].get('attempt_number')} "
            f"— {attempts[index].get('outcome', 'UNKNOWN')}"
        ),
    )

    selected_attempt = attempts[selected_index]

    # Raw trajectory coordinates are stored separately from
    # the unified assessment.
    raw_session_path = (
        PROJECT_ROOT
        / "data"
        / "trajectory_results"
        / f"{session['session_id']}_trajectory_results.json"
    )

    if raw_session_path.is_file():
        raw_session = load_session(raw_session_path)

        matching_attempts = [
            item
            for item in raw_session.get("attempts", [])
            if item.get("attempt_number")
            == selected_attempt.get("attempt_number")
        ]

        if len(matching_attempts) == 1:
            render_trajectory_visualization(
                matching_attempts[0],
                raw_session.get(
                    "trajectory",
                    task.get("trajectory"),
                ),
                session_trajectory_config=raw_session.get(
                    "trajectory_config"
                ),
            )
        elif not matching_attempts:
            st.warning(
                "No matching raw trajectory recording was found "
                "for this attempt."
            )
        else:
            st.warning(
                "Multiple raw attempts share this attempt number. "
                "Trajectory matching requires review."
            )
    else:
        st.warning(
            "Original trajectory recording not found for this session."
        )
else:
    st.info("No attempts are available for trajectory review.")

st.divider()

# -------------------------------------------------------
# Consistency and observed performance change
# -------------------------------------------------------

st.subheader("Consistency and Performance Changes")

left, right = st.columns(2)

with left:
    st.markdown("**Repetition Consistency**")

    consistency = session.get("consistency") or {}

    consistency_rows = []

    for metric, values in consistency.items():
        if not isinstance(values, dict):
            continue

        consistency_rows.append({
            "Metric": metric,
            "Mean": values.get("mean"),
            "Coefficient of Variation": values.get("cv"),
            "Samples": values.get("sample_count"),
        })

    if consistency_rows:
        st.dataframe(
            pd.DataFrame(consistency_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Consistency measurements are unavailable.")

with right:
    st.markdown("**First-to-Last Observed Change**")

    changes = session.get("observed_performance_change") or {}

    change_rows = []

    for metric, value in changes.items():
        if not isinstance(value, dict):
            continue

        change_rows.append({
            "Metric": metric.replace("_", " ").title(),
            "First": value.get("first"),
            "Last": value.get("last"),
            "Change (%)": value.get("change_percent"),
        })

    if change_rows:
        st.dataframe(
            pd.DataFrame(change_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Performance-change measurements are unavailable.")

st.caption(
    "Observed changes describe performance within this session. "
    "They do not independently establish clinical improvement or fatigue."
)

st.divider()

# -------------------------------------------------------
# Joint movement quality
# -------------------------------------------------------

st.subheader("Upper-Limb Movement Quality")

joint_rows = []

for attempt in attempts:
    screened = attempt.get("pose_movement_quality_screened") or {}
    joints = screened.get("joints") or {}
    quality = attempt.get("joint_quality_control") or {}

    for joint in (
        "left_elbow",
        "right_elbow",
        "left_shoulder",
        "right_shoulder",
    ):
        joint_metrics = joints.get(joint) or {}
        joint_quality = quality.get(joint) or {}

        joint_rows.append({
            "Attempt": attempt.get("attempt_number"),
            "Joint": joint.replace("_", " ").title(),
            "Screened ROM (degrees)": joint_metrics.get("rom_deg"),
            "Quality Status": joint_quality.get("status", "UNKNOWN"),
            "Coverage (%)": joint_quality.get("coverage_percent"),
            "Flagged Transitions": joint_quality.get(
                "flagged_transition_count"
            ),
        })

if joint_rows:
    st.dataframe(
        pd.DataFrame(joint_rows),
        use_container_width=True,
        hide_index=True,
    )

st.caption(
    "Joint measurements that fail preliminary quality screening "
    "are withheld. These engineering checks are not clinically validated."
)

st.divider()


# -------------------------------------------------------
# Grasp and hand-object interaction assessment
# -------------------------------------------------------

st.subheader("Grasp and Hand-Object Interaction Assessment")

session_id = str(session.get("session_id") or "")

try:
    combined = load_combined_session(session_id)
except (FileNotFoundError, ValueError, OSError) as error:
    st.warning(f"Grasp observations unavailable: {error}")
else:
    grasp = combined["session_grasp_summary"]

    col1, col2 = st.columns(2)
    col1.metric("Hand Observations", grasp["hand_observations"])
    col2.metric("Detected Hand Sides", len(grasp["by_hand"]))

    left, right = st.columns(2)

    with left:
        st.markdown("**Detected Grasp Types**")
        counts = grasp["grasp_type_counts"]
        if counts:
            st.bar_chart(
                pd.Series(counts).sort_values(ascending=False)
            )
        else:
            st.info("No grasp classifications recorded.")

    with right:
        st.markdown("**Grasp Appropriateness Observations**")
        counts = grasp["appropriateness_counts"]
        if counts:
            st.bar_chart(pd.Series(counts))
        else:
            st.info("No appropriateness observations recorded.")

    st.markdown("**Hand-Specific Observations**")

    hand_rows = []
    for hand, data in grasp["by_hand"].items():
        counts = data["appropriateness_counts"]
        hand_rows.append({
            "Detected Hand": hand,
            "Observations": data["hand_observations"],
            "Appropriate": counts.get("APPROPRIATE", 0),
            "Inappropriate": counts.get("INAPPROPRIATE", 0),
            "Uncertain": counts.get("UNCERTAIN", 0),
            "Not Assessed": counts.get("NOT_ASSESSED", 0),
        })

    if hand_rows:
        st.dataframe(
            pd.DataFrame(hand_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.markdown("**Recorded Interaction Events**")

    events = grasp["interaction_event_counts"]
    if events:
        st.bar_chart(pd.Series(events))
    else:
        st.info("No interaction events recorded.")

    st.markdown("**Repetition-Level Grasp Observations**")

    attempt_rows = []
    for attempt in combined["attempts"]:
        attempt_grasp = attempt["grasp_summary"]
        counts = (
            attempt_grasp["appropriateness_counts"]
            if attempt_grasp else {}
        )

        attempt_rows.append({
            "Attempt": attempt["attempt_number"],
            "Outcome": attempt["outcome"],
            "Alignment": attempt["grasp_alignment_status"],
            "Hand Observations": (
                attempt_grasp["hand_observations"]
                if attempt_grasp else None
            ),
            "Appropriate": counts.get("APPROPRIATE"),
            "Inappropriate": counts.get("INAPPROPRIATE"),
            "Uncertain": counts.get("UNCERTAIN"),
            "Not Assessed": counts.get("NOT_ASSESSED"),
        })

    if attempt_rows:
        st.dataframe(
            pd.DataFrame(attempt_rows),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Grasp classifications and appropriateness labels are "
        "preliminary frame-level observations, not validated "
        "clinical scores or independent grasp success rates. "
        "Handedness labels are detector outputs, not confirmed "
        "affected-limb information. Missing attempt timestamps "
        "prevent repetition-level grasp alignment."
    )

st.divider()

# -------------------------------------------------------
# Data quality and attempt review
# -------------------------------------------------------

st.subheader("Data Quality and Attempt Review")

for attempt in attempts:
    attempt_number = attempt.get("attempt_number")
    outcome = attempt.get("outcome", "UNKNOWN")

    with st.expander(
        f"Attempt {attempt_number} — {outcome}",
        expanded=False,
    ):
        st.write(
            "**Active hand:**",
            attempt.get("active_hand", "UNKNOWN"),
        )

        st.write(
            "**Hand switch detected:**",
            attempt.get("hand_switch_detected"),
        )

        warnings = attempt.get("quality_warnings") or []

        if warnings:
            st.markdown("**Quality warnings**")
            for warning in warnings:
                st.warning(str(warning))
        else:
            st.success("No quality warnings recorded.")

        with st.expander("View raw assessment data"):
            st.json(attempt)

st.divider()

st.info(
    "Research prototype. Measurements are not validated clinical "
    "assessment scores and should not be used independently "
    "for clinical decisions."
)
